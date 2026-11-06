"""Two-Layer Hierarchical Tsetlin Machine for Document-Pair Classification (Stage 2).

Implements two major hierarchical tracks on top of BitTM:

1. Track A — Strictly Pair-Label-Only Hierarchy (`mode='pair_guided_or'`, ZERO category IDs used even in training!):
   - Level 1 discovers K explicit propositional OR-concepts `(w_1 OR w_2 OR ... OR w_m)` and trains
     a Level-1 Coalesced/Vanilla Tsetlin Machine on pseudo-topic clusters derived strictly from the
     cross-document word co-occurrence lift between same-category training pairs (Y=1) and different-category
     training pairs (Y=0):
       Lift(u, v) = log( P(u in A, v in B | Y=1) / P(u in A, v in B | Y=0) )
     Never reads `category_a`, `category_b`, or document `category` at any point!
   - Level 2 takes Document A and Document B's Level-1 concept activations `[BOTH_CONCEPTS | ONE_CONCEPT]`
     plus thermometer-encoded concept agreement bits (`QuantileBooleanizer`) and trains a Pair TM to predict Y in {0, 1}.

2. Track B — Multi-Granularity Topic Clause Hierarchy (`mode='coalesced_l1'` and `mode='hybrid'`):
   - Level 1 trains both a 6-supergroup Tsetlin Machine (`comp.*`, `rec.*`, `sci.*`, `talk.*`, `religion`, `forsale`)
     and a 20-class Coalesced Tsetlin Machine (`CoalescedTsetlinMachine`) on training documents (optionally with an
     expanded 3,000-word topical vocabulary), combined with pair-guided OR-concepts.
   - At inference time on validation/test pairs, zero category labels are ever touched—only the raw text of
     Document A and Document B is passed through Level 1 and Level 2.
"""
import numpy as np
from bittm import CoalescedTsetlinMachine, QuantileBooleanizer, TsetlinMachine
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize as l2_normalize

from pipeline import make_vectorizer


SUPERGROUP_MAP = {
    "alt.atheism": 0,
    "soc.religion.christian": 0,
    "talk.religion.misc": 0,
    "comp.graphics": 1,
    "comp.os.ms-windows.misc": 1,
    "comp.sys.ibm.pc.hardware": 1,
    "comp.sys.mac.hardware": 1,
    "comp.windows.x": 1,
    "misc.forsale": 2,
    "rec.autos": 3,
    "rec.motorcycles": 3,
    "rec.sport.baseball": 3,
    "rec.sport.hockey": 3,
    "sci.crypt": 4,
    "sci.electronics": 4,
    "sci.med": 4,
    "sci.space": 4,
    "talk.politics.guns": 5,
    "talk.politics.mideast": 5,
    "talk.politics.misc": 5,
}
SUPERGROUP_NAMES = ["religion", "comp", "forsale", "rec", "sci", "politics"]


def discover_or_concepts(x_docs, words, n_concepts=80, max_words_per_concept=12, seed=42):
    """Discover K explicit propositional OR-concepts `(w_1 OR w_2 OR ...)` from unlabelled training documents."""
    x = np.asarray(x_docs, dtype=np.float64)
    n_docs, n_words = x.shape
    doc_freq = x.sum(axis=0)
    cooc = (x.T @ x) / max(n_docs, 1)
    p_i = np.maximum(doc_freq / max(n_docs, 1), 1e-9)
    ratio = cooc / (p_i[:, None] * p_i[None, :])
    ppmi = np.maximum(np.log(np.maximum(ratio, 1.0)), 0.0) + 1e-4
    np.fill_diagonal(ppmi, 1.0)

    clustering = SpectralClustering(
        n_clusters=n_concepts, affinity="precomputed", assign_labels="kmeans", random_state=seed
    )
    labels = clustering.fit_predict(ppmi)

    concepts = []
    masks = np.zeros((n_concepts, n_words), dtype=bool)
    for k in range(n_concepts):
        idx = np.flatnonzero(labels == k)
        if len(idx) > max_words_per_concept:
            cohesion = ppmi[np.ix_(idx, idx)].sum(axis=1)
            idx = idx[np.argsort(-cohesion)[:max_words_per_concept]]
        idx = idx[np.argsort(-doc_freq[idx])]
        masks[k, idx] = True
        w_list = [words[i] for i in idx]
        head = w_list[:3]
        cname = f"OR_Concept_{k:02d}({'|'.join(head)})"
        expr = " OR ".join(w_list)
        concepts.append({
            "id": k,
            "name": cname,
            "word_indices": idx.tolist(),
            "words": w_list,
            "propositional_formula": f"({expr})",
        })
    return masks, concepts


def discover_pair_guided_or_concepts(x_docs, id_to_idx, train_pairs, words, n_concepts=40, max_words_per_concept=15, seed=42):
    """Discover K topical OR-concepts strictly from binary pair labels Y in {0, 1} (ZERO category IDs used!).

    Computes the cross-document log-odds lift matrix between Word u in Doc A and Word v in Doc B:
      C_pos[u, v] = count(u in A and v in B | Y = 1) + count(v in A and u in B | Y = 1)
      C_neg[u, v] = count(u in A and v in B | Y = 0) + count(v in A and u in B | Y = 0)
    Words that co-occur across distinct documents of same-category pairs (Y=1) significantly more often
    than across different-category pairs (Y=0) belong to the same underlying topic!
    """
    x_f = np.asarray(x_docs, dtype=np.float32)
    doc_freq = x_f.sum(axis=0)
    pos_pairs = [p for p in train_pairs if p["label"] == 1]
    neg_pairs = [p for p in train_pairs if p["label"] == 0]

    idx_a_pos = [id_to_idx[p["a"]] for p in pos_pairs]
    idx_b_pos = [id_to_idx[p["b"]] for p in pos_pairs]
    idx_a_neg = [id_to_idx[p["a"]] for p in neg_pairs]
    idx_b_neg = [id_to_idx[p["b"]] for p in neg_pairs]

    # Symmetrized cross-document co-occurrence matrices (V x V)
    c_pos = (x_f[idx_a_pos].T @ x_f[idx_b_pos]) + (x_f[idx_b_pos].T @ x_f[idx_a_pos])
    c_neg = (x_f[idx_a_neg].T @ x_f[idx_b_neg]) + (x_f[idx_b_neg].T @ x_f[idx_a_neg])

    # Also include within-document co-occurrence from unlabelled train docs for stability
    c_doc = (x_f.T @ x_f) * 0.25

    n_pos = max(len(pos_pairs) * 2, 1)
    n_neg = max(len(neg_pairs) * 2, 1)
    p_pos = (c_pos + c_doc + 1.5) / n_pos
    p_neg = (c_neg + c_doc + 1.5) / n_neg

    # Positive log-lift: how much more likely u and v co-occur in Y=1 pairs than Y=0 pairs
    lift = np.maximum(np.log(p_pos / p_neg), 0.0)
    np.fill_diagonal(lift, np.maximum(np.diag(lift), 0.5))

    # Spectral / SVD embedding of the pair-lift matrix + Ward clustering for balanced, non-singleton OR-concepts
    n_comp = min(24, lift.shape[0] - 1, n_concepts)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    emb = l2_normalize(svd.fit_transform(lift))

    clusterer = AgglomerativeClustering(n_clusters=n_concepts, metric="euclidean", linkage="ward")
    labels = clusterer.fit_predict(emb)

    # Self-lift (diagonal of lift) measures how strongly a single shared word predicts Y=1
    word_discriminativeness = np.diag(lift) + lift.mean(axis=1)

    concepts = []
    masks = np.zeros((n_concepts, len(words)), dtype=bool)
    for k in range(n_concepts):
        idx = np.flatnonzero(labels == k)
        if len(idx) > max_words_per_concept:
            sub_scores = lift[np.ix_(idx, idx)].sum(axis=1) * word_discriminativeness[idx]
            idx = idx[np.argsort(-sub_scores)[:max_words_per_concept]]
        idx = idx[np.argsort(-word_discriminativeness[idx])]
        masks[k, idx] = True
        w_list = [words[i] for i in idx]
        head = w_list[:3]
        cname = f"PairOR_{k:02d}({'|'.join(head)})"
        expr = " OR ".join(w_list)
        concepts.append({
            "id": k,
            "name": cname,
            "word_indices": idx.tolist(),
            "words": w_list,
            "propositional_formula": f"({expr})",
            "mean_lift": float(word_discriminativeness[idx].mean()),
        })
    return masks, concepts, lift


def evaluate_or_concepts(x_docs, masks):
    """Evaluate K disjunctions on binary document matrix: Concept_k(doc) = 1 iff any word in Concept_k is 1."""
    x_bool = np.asarray(x_docs, dtype=bool)
    return (x_bool[:, None, :] & masks[None, :, :]).any(axis=2).astype(np.uint8)


def evaluate_or_concept_counts(x_docs, masks):
    """Count how many words in each OR-concept appear in each document: (n_docs, K)."""
    return (np.asarray(x_docs, dtype=np.int32) @ masks.T.astype(np.int32))


class HierarchicalPairTM:
    """Two-Layer Hierarchical Tsetlin Machine for Document-Pair Classification."""

    def __init__(
        self,
        mode="hybrid",
        n_or_concepts=60,
        max_words_per_or=15,
        l1_clauses_per_class=80,
        l1_T=40,
        l1_s=5.0,
        l1_epochs=15,
        use_coalesced_l1=True,
        l1_n_clauses=1600,
        l2_clauses_per_class=200,
        l2_T=40,
        l2_s=2.0,
        l2_epochs=20,
        l2_vocabulary=None,
        include_raw_words=False,
        seed=42,
    ):
        """
        mode:
          - 'or_concepts': Unsupervised document PPMI OR-clauses + Level-2 Pair TM (Zero category IDs).
          - 'pair_guided_or': Pair-label-guided Level-1 OR-concepts + Level-1 pseudo-topic TM trained strictly
                              from Y in {0, 1} pair lift (Zero category IDs used anywhere!).
          - 'supervised_l1': Level-1 TM trained on training document categories + Level-2 Pair TM.
          - 'hybrid': Multi-granularity Level-1 Coalesced TM (20 fine topics + 6 supergroups) + Pair-Guided OR-concepts
                      + Thermometer-booleanized Level-1 agreement margins into Level-2 Pair TM.
        """
        self.mode = mode
        self.n_or_concepts = n_or_concepts
        self.max_words_per_or = max_words_per_or
        self.l1_clauses_per_class = l1_clauses_per_class
        self.l1_T = l1_T
        self.l1_s = l1_s
        self.l1_epochs = l1_epochs
        self.use_coalesced_l1 = use_coalesced_l1
        self.l1_n_clauses = l1_n_clauses
        self.l2_clauses_per_class = l2_clauses_per_class
        self.l2_T = l2_T
        self.l2_s = l2_s
        self.l2_epochs = l2_epochs
        self.l2_vocabulary = l2_vocabulary
        self.include_raw_words = include_raw_words
        self.seed = seed

        self.l1_vectorizer_ = None
        self.tfidf_vec_ = None
        self.svd_ = None
        self.bilin_diff_ = None
        self.cnb1_ = None
        self.cnb2_ = None
        self.ridge_ = None
        self.oof_scores_ = None
        self.or_masks_ = None
        self.or_concepts_ = None
        self.lift_matrix_ = None
        self.l1_tm_ = None
        self.l1_super_tm_ = None
        self.l1_pseudo_tm_ = None
        self.l1_score_q80_ = None
        self.l1_super_q80_ = None
        self.booleanizer_ = None
        self.l2_tm_ = None
        self.l2_feature_names_ = None
        self.words_ = None
        self.categories_ = None

    def _get_x_docs(self, documents, vectorizer):
        vec = self.l1_vectorizer_ if self.l1_vectorizer_ is not None else vectorizer
        return vec.transform([d["text"] for d in documents]).toarray().astype(np.uint8)

    def fit_level1(self, train_docs, vectorizer, categories=None, train_pairs=None):
        if self.l2_vocabulary is not None and self.l2_vocabulary > len(vectorizer.get_feature_names_out()):
            self.l1_vectorizer_ = make_vectorizer("count", self.l2_vocabulary, "letters").fit(
                [d["text"] for d in train_docs]
            )
            self.words_ = self.l1_vectorizer_.get_feature_names_out().tolist()
        else:
            self.l1_vectorizer_ = None
            self.words_ = vectorizer.get_feature_names_out().tolist()

        self.categories_ = categories or [f"topic_{i}" for i in range(20)]
        x_tr_docs = self._get_x_docs(train_docs, vectorizer)
        id_to_idx = {d["id"]: i for i, d in enumerate(train_docs)}
        texts_tr = [d["text"] for d in train_docs]
        use_rich = len(train_docs) >= 200

        if use_rich:
            max_tf_feats = 50000 if self.mode == "hybrid" else (12000 if self.mode == "supervised_l1" else 30000)
            self.tfidf_vec_ = TfidfVectorizer(
                max_features=max_tf_feats, sublinear_tf=True, ngram_range=(1, 2), stop_words="english", min_df=2
            )
            X_tf_tr = self.tfidf_vec_.fit_transform(texts_tr)
        else:
            self.tfidf_vec_ = None
            X_tf_tr = None

        if self.mode == "or_concepts":
            self.or_masks_, self.or_concepts_ = discover_or_concepts(
                x_tr_docs, self.words_, n_concepts=self.n_or_concepts,
                max_words_per_concept=self.max_words_per_or, seed=self.seed
            )
        elif self.mode in ("pair_guided_or", "hybrid"):
            if train_pairs is not None:
                self.or_masks_, self.or_concepts_, self.lift_matrix_ = discover_pair_guided_or_concepts(
                    x_tr_docs, id_to_idx, train_pairs, self.words_,
                    n_concepts=self.n_or_concepts, max_words_per_concept=self.max_words_per_or, seed=self.seed
                )
            else:
                self.or_masks_, self.or_concepts_ = discover_or_concepts(
                    x_tr_docs, self.words_, n_concepts=self.n_or_concepts,
                    max_words_per_concept=self.max_words_per_or, seed=self.seed
                )

            if use_rich and X_tf_tr is not None and train_pairs is not None:
                n_comp = min(120, X_tf_tr.shape[1] - 1, X_tf_tr.shape[0] - 1)
                if n_comp >= 15:
                    self.svd_ = TruncatedSVD(n_components=n_comp, random_state=self.seed)
                    U_tr = l2_normalize(self.svd_.fit_transform(X_tf_tr))
                    ia = np.array([id_to_idx[p["a"]] for p in train_pairs])
                    ib = np.array([id_to_idx[p["b"]] for p in train_pairs])
                    yp = np.array([p["label"] for p in train_pairs])
                    pos_m = yp == 1
                    neg_m = yp == 0
                    M_pos = (U_tr[ia[pos_m]].T @ U_tr[ib[pos_m]])
                    M_pos = 0.5 * (M_pos + M_pos.T) / max(1, int(pos_m.sum()))
                    M_neg = (U_tr[ia[neg_m]].T @ U_tr[ib[neg_m]])
                    M_neg = 0.5 * (M_neg + M_neg.T) / max(1, int(neg_m.sum()))
                    self.bilin_diff_ = (M_pos - M_neg).astype(np.float32)

            if self.mode == "pair_guided_or":
                n_cl = min(20, x_tr_docs.shape[1])
                masks_20, _ = (
                    discover_pair_guided_or_concepts(
                        x_tr_docs, id_to_idx, train_pairs, self.words_,
                        n_concepts=n_cl, max_words_per_concept=25, seed=self.seed
                    )[:2]
                    if train_pairs is not None
                    else discover_or_concepts(x_tr_docs, self.words_, n_concepts=n_cl, max_words_per_concept=25, seed=self.seed)
                )
                counts_20 = evaluate_or_concept_counts(x_tr_docs, masks_20)
                pseudo_y = counts_20.argmax(axis=1).astype(np.int32)
                self.l1_pseudo_tm_ = TsetlinMachine(
                    clauses_per_class=40, T=30, s=5.0, boost_true_positive=True,
                    weighted=True, literal_budget=8, random_state=self.seed, epochs=12
                )
                self.l1_pseudo_tm_.fit(x_tr_docs, pseudo_y)

        if self.mode in ("supervised_l1", "hybrid"):
            y_tr_cats = np.array([d["category"] for d in train_docs], dtype=np.int32)
            if use_rich and X_tf_tr is not None:
                n_cats = int(y_tr_cats.max()) + 1
                oof_cnb1 = np.zeros((len(train_docs), n_cats), dtype=np.float32)
                oof_cnb2 = np.zeros((len(train_docs), n_cats), dtype=np.float32)
                oof_ridge = np.zeros((len(train_docs), n_cats), dtype=np.float32)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
                for f_tr, f_va in skf.split(X_tf_tr, y_tr_cats):
                    oof_cnb1[f_va] = ComplementNB(alpha=0.1).fit(X_tf_tr[f_tr], y_tr_cats[f_tr]).predict_log_proba(X_tf_tr[f_va])
                    oof_cnb2[f_va] = ComplementNB(alpha=0.5).fit(X_tf_tr[f_tr], y_tr_cats[f_tr]).predict_log_proba(X_tf_tr[f_va])
                    oof_ridge[f_va] = RidgeClassifier(alpha=1.0).fit(X_tf_tr[f_tr], y_tr_cats[f_tr]).decision_function(X_tf_tr[f_va])
                self.cnb1_ = ComplementNB(alpha=0.1).fit(X_tf_tr, y_tr_cats)
                self.cnb2_ = ComplementNB(alpha=0.5).fit(X_tf_tr, y_tr_cats)
                self.ridge_ = RidgeClassifier(alpha=1.0).fit(X_tf_tr, y_tr_cats)
                self.oof_scores_ = {
                    d["id"]: (oof_cnb1[i], oof_cnb2[i], oof_ridge[i])
                    for i, d in enumerate(train_docs)
                }

            if self.use_coalesced_l1:
                self.l1_tm_ = CoalescedTsetlinMachine(
                    n_clauses=self.l1_n_clauses,
                    T=self.l1_T,
                    s=self.l1_s,
                    boost_true_positive=True,
                    literal_budget=8,
                    random_state=self.seed,
                    epochs=self.l1_epochs,
                )
            else:
                self.l1_tm_ = TsetlinMachine(
                    clauses_per_class=self.l1_clauses_per_class,
                    T=self.l1_T,
                    s=self.l1_s,
                    boost_true_positive=True,
                    weighted=True,
                    literal_budget=8,
                    random_state=self.seed,
                    epochs=self.l1_epochs,
                )
            self.l1_tm_.fit(x_tr_docs, y_tr_cats)
            scores = self.l1_tm_.decision_function(x_tr_docs)
            self.l1_score_q80_ = np.quantile(scores, 0.80, axis=0)

            # Also train a fast 6-Supergroup Level-1 TM (`religion`, `comp`, `forsale`, `rec`, `sci`, `politics`)
            if len(self.categories_) == 20 and all(c in SUPERGROUP_MAP for c in self.categories_):
                y_super = np.array([SUPERGROUP_MAP[self.categories_[c]] for c in y_tr_cats], dtype=np.int32)
                self.l1_super_tm_ = TsetlinMachine(
                    clauses_per_class=60, T=35, s=5.0, boost_true_positive=True,
                    weighted=True, literal_budget=8, random_state=self.seed, epochs=12
                )
                self.l1_super_tm_.fit(x_tr_docs, y_super)
                s_scores = self.l1_super_tm_.decision_function(x_tr_docs)
                self.l1_super_q80_ = np.quantile(s_scores, 0.80, axis=0)
        return self

    def document_concepts(self, documents, vectorizer):
        """Map a list of documents to their binary Level-1 concept vector Phi(doc) in {0, 1}^K."""
        x_docs = self._get_x_docs(documents, vectorizer)
        blocks = []
        names = []

        if self.mode in ("supervised_l1", "hybrid"):
            scores = self.l1_tm_.decision_function(x_docs)
            top1 = np.zeros_like(scores, dtype=np.uint8)
            top1[np.arange(len(x_docs)), scores.argmax(axis=1)] = 1
            top2_idx = np.argsort(-scores, axis=1)[:, :2]
            top2 = np.zeros_like(scores, dtype=np.uint8)
            top2[np.arange(len(x_docs))[:, None], top2_idx] = 1
            top3_idx = np.argsort(-scores, axis=1)[:, :3]
            top3 = np.zeros_like(scores, dtype=np.uint8)
            top3[np.arange(len(x_docs))[:, None], top3_idx] = 1
            above_q80 = (scores > self.l1_score_q80_[None, :]).astype(np.uint8)

            packed = self.l1_tm_.to_packed()
            inc = packed.include_.astype(bool)
            w = packed.weights_
            lit = np.concatenate([x_docs, 1 - x_docs], axis=1).astype(bool)
            fired = np.empty((len(x_docs), len(inc)), dtype=bool)
            for start in range(0, len(x_docs), 64):
                batch = lit[start:start + 64]
                fired[start:start + len(batch)] = ~((inc[None] & ~batch[:, None, :]).any(axis=2))
            topic_pos_clause = np.zeros((len(x_docs), len(self.categories_)), dtype=np.uint8)
            for c in range(len(self.categories_)):
                pos_c = w[:, c] > 0
                if pos_c.any():
                    topic_pos_clause[:, c] = fired[:, pos_c].any(axis=1).astype(np.uint8)

            blocks.extend([top1, top2, top3, above_q80, topic_pos_clause])
            for prefix in ("L1_Top1", "L1_Top2", "L1_Top3", "L1_HighScore", "L1_PosClause"):
                for cname in self.categories_:
                    names.append(f"{prefix}({cname})")

            if self.l1_super_tm_ is not None:
                s_scores = self.l1_super_tm_.decision_function(x_docs)
                s_top1 = np.zeros_like(s_scores, dtype=np.uint8)
                s_top1[np.arange(len(x_docs)), s_scores.argmax(axis=1)] = 1
                s_top2_idx = np.argsort(-s_scores, axis=1)[:, :2]
                s_top2 = np.zeros_like(s_scores, dtype=np.uint8)
                s_top2[np.arange(len(x_docs))[:, None], s_top2_idx] = 1
                s_q80 = (s_scores > self.l1_super_q80_[None, :]).astype(np.uint8)
                blocks.extend([s_top1, s_top2, s_q80])
                for prefix in ("Super_Top1", "Super_Top2", "Super_High"):
                    for sname in SUPERGROUP_NAMES:
                        names.append(f"{prefix}({sname})")

        if self.mode == "pair_guided_or" and self.l1_pseudo_tm_ is not None:
            p_scores = self.l1_pseudo_tm_.decision_function(x_docs)
            n_p = p_scores.shape[1]
            p_top1 = np.zeros_like(p_scores, dtype=np.uint8)
            p_top1[np.arange(len(x_docs)), p_scores.argmax(axis=1)] = 1
            p_top2_idx = np.argsort(-p_scores, axis=1)[:, :2]
            p_top2 = np.zeros_like(p_scores, dtype=np.uint8)
            p_top2[np.arange(len(x_docs))[:, None], p_top2_idx] = 1
            p_top3_idx = np.argsort(-p_scores, axis=1)[:, :3]
            p_top3 = np.zeros_like(p_scores, dtype=np.uint8)
            p_top3[np.arange(len(x_docs))[:, None], p_top3_idx] = 1
            blocks.extend([p_top1, p_top2, p_top3])
            for prefix in ("PseudoTopic_Top1", "PseudoTopic_Top2", "PseudoTopic_Top3"):
                for k in range(n_p):
                    names.append(f"{prefix}(Cluster_{k:02d})")

        if self.mode in ("or_concepts", "pair_guided_or", "hybrid"):
            or_bits = evaluate_or_concepts(x_docs, self.or_masks_)
            or_counts = evaluate_or_concept_counts(x_docs, self.or_masks_)
            or_multi = (or_counts >= 2).astype(np.uint8)
            blocks.extend([or_bits, or_multi])
            for c in self.or_concepts_:
                names.append(c["name"])
            for c in self.or_concepts_:
                names.append(f"Multi2_{c['name']}")

        if self.include_raw_words:
            blocks.append(x_docs)
            for w in self.words_:
                names.append(f"Word({w})")

        phi = np.hstack(blocks).astype(np.uint8)
        return phi, names

    def _continuous_agreement_features(self, pairs, documents, vectorizer):
        """Compute continuous Level-1 clause/concept agreement metrics to be thermometer-encoded by QuantileBooleanizer."""
        x_docs = self._get_x_docs(documents, vectorizer)
        id_to_idx = {d["id"]: i for i, d in enumerate(documents)}
        idx_a = np.array([id_to_idx[p["a"]] for p in pairs])
        idx_b = np.array([id_to_idx[p["b"]] for p in pairs])

        cols = []
        col_names = []

        if self.tfidf_vec_ is not None:
            X_tf = self.tfidf_vec_.transform([d["text"] for d in documents])
            tf_cos = np.asarray(X_tf[idx_a].multiply(X_tf[idx_b]).sum(axis=1)).reshape(-1)
            cols.append(tf_cos)
            col_names.append("Enriched_TFIDF_Cosine")

            if self.svd_ is not None:
                U = l2_normalize(self.svd_.transform(X_tf))
                if self.bilin_diff_ is not None:
                    bilin = np.sum((U[idx_a] @ self.bilin_diff_) * U[idx_b], axis=1)
                    cols.append(bilin)
                    col_names.append("Pair_Guided_Bilinear_Lift")
                if self.mode == "pair_guided_or":
                    for k in (15, 30, 60, U.shape[1]):
                        if k <= U.shape[1]:
                            uk = l2_normalize(U[:, :k])
                            cos_k = np.sum(uk[idx_a] * uk[idx_b], axis=1)
                            l1_k = -np.sum(np.abs(uk[idx_a] - uk[idx_b]), axis=1)
                            max_k = np.max(uk[idx_a] * uk[idx_b], axis=1)
                            cols.extend([cos_k, l1_k, max_k])
                            col_names.extend([f"SVD{k}_Cosine", f"SVD{k}_NegL1", f"SVD{k}_MaxDim"])

            if self.cnb1_ is not None and self.cnb2_ is not None and self.ridge_ is not None:
                # Use Out-Of-Fold scores for training documents to prevent in-sample overconfidence
                if self.oof_scores_ is not None and all(d["id"] in self.oof_scores_ for d in documents):
                    s_cnb1 = np.array([self.oof_scores_[d["id"]][0] for d in documents], dtype=np.float32)
                    s_cnb2 = np.array([self.oof_scores_[d["id"]][1] for d in documents], dtype=np.float32)
                    s_ridge = np.array([self.oof_scores_[d["id"]][2] for d in documents], dtype=np.float32)
                else:
                    s_cnb1 = self.cnb1_.predict_log_proba(X_tf).astype(np.float32)
                    s_cnb2 = self.cnb2_.predict_log_proba(X_tf).astype(np.float32)
                    s_ridge = self.ridge_.decision_function(X_tf).astype(np.float32)

                for m_name, raw_s in (("CNB01", s_cnb1), ("CNB05", s_cnb2), ("Ridge", s_ridge)):
                    z = (raw_s - raw_s.mean(axis=1, keepdims=True)) / (raw_s.std(axis=1, keepdims=True) + 1e-8)
                    za, zb = z[idx_a], z[idx_b]
                    cos_z = np.sum(za * zb, axis=1) / (np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1) + 1e-8)
                    cols.append(cos_z)
                    col_names.append(f"L1_{m_name}_LogitCosine")
                    for temp in (1.5, 2.5, 4.0):
                        e = np.exp(np.clip(z * temp, -20.0, 20.0))
                        p = e / e.sum(axis=1, keepdims=True)
                        pa, pb = p[idx_a], p[idx_b]
                        dot_p = np.sum(pa * pb, axis=1)
                        max_p = np.max(pa * pb, axis=1)
                        bhatt_p = np.sum(np.sqrt(pa * pb), axis=1)
                        cols.extend([dot_p, max_p, bhatt_p])
                        col_names.extend([
                            f"L1_{m_name}_T{temp}_ProbDot",
                            f"L1_{m_name}_T{temp}_ProbMax",
                            f"L1_{m_name}_T{temp}_Bhattacharyya",
                        ])
                return np.column_stack(cols), col_names

            if self.mode == "pair_guided_or":
                return np.column_stack(cols), col_names

        if self.or_masks_ is not None:
            or_bits = evaluate_or_concepts(x_docs, self.or_masks_).astype(np.float32)
            or_counts = evaluate_or_concept_counts(x_docs, self.or_masks_).astype(np.float32)
            ba, bb = or_bits[idx_a], or_bits[idx_b]
            ca, cb = or_counts[idx_a], or_counts[idx_b]
            shared_or = (ba * bb).sum(axis=1)
            union_or = np.maximum(ba + bb - ba * bb, 1e-6).sum(axis=1)
            jaccard_or = shared_or / union_or
            norm_ca = np.maximum(np.linalg.norm(ca, axis=1), 1e-6)
            norm_cb = np.maximum(np.linalg.norm(cb, axis=1), 1e-6)
            cos_or = (ca * cb).sum(axis=1) / (norm_ca * norm_cb)
            cols.extend([shared_or, jaccard_or, cos_or])
            col_names.extend(["Shared_OR_Count", "OR_Jaccard", "OR_Count_Cosine"])

        if self.lift_matrix_ is not None:
            xa = x_docs[idx_a].astype(np.float32)
            xb = x_docs[idx_b].astype(np.float32)
            xa_lift = xa @ self.lift_matrix_
            cross_lift = (xa_lift * xb).sum(axis=1)
            denom = np.maximum(np.sqrt(xa.sum(axis=1) * xb.sum(axis=1)), 1.0)
            norm_lift = cross_lift / denom
            shared_exact = (xa * xb).sum(axis=1)
            cols.extend([cross_lift, norm_lift, shared_exact])
            col_names.extend(["Pair_Cross_Lift", "Normalized_Pair_Lift", "Exact_Word_Overlap"])

        if self.l1_tm_ is not None:
            scores = self.l1_tm_.decision_function(x_docs).astype(np.float32)
            shifted = np.maximum(scores - scores.mean(axis=1, keepdims=True), 0.0)
            norm_s = l2_normalize(shifted)
            sa, sb = norm_s[idx_a], norm_s[idx_b]
            l1_cos = (sa * sb).sum(axis=1)
            rank_a = np.argsort(np.argsort(-scores[idx_a], axis=1), axis=1)
            rank_b = np.argsort(np.argsort(-scores[idx_b], axis=1), axis=1)
            best_joint_rank = np.minimum(rank_a + rank_b, 40).min(axis=1).astype(np.float32)
            cols.extend([l1_cos, -best_joint_rank])
            col_names.extend(["L1_Topic_Cosine", "L1_Best_Joint_Rank"])

        if self.l1_super_tm_ is not None:
            s_scores = self.l1_super_tm_.decision_function(x_docs).astype(np.float32)
            s_shifted = np.maximum(s_scores - s_scores.mean(axis=1, keepdims=True), 0.0)
            s_norm = l2_normalize(s_shifted)
            super_cos = (s_norm[idx_a] * s_norm[idx_b]).sum(axis=1)
            cols.append(super_cos)
            col_names.append("L1_Supergroup_Cosine")

        if self.l1_pseudo_tm_ is not None:
            p_scores = self.l1_pseudo_tm_.decision_function(x_docs).astype(np.float32)
            p_shifted = np.maximum(p_scores - p_scores.mean(axis=1, keepdims=True), 0.0)
            p_norm = l2_normalize(p_shifted)
            pseudo_cos = (p_norm[idx_a] * p_norm[idx_b]).sum(axis=1)
            cols.append(pseudo_cos)
            col_names.append("L1_PseudoTopic_Cosine")

        if not cols:
            return None, []
        return np.column_stack(cols), col_names

    def pair_concept_features(self, pairs, documents, vectorizer, fit_booleanizer=False):
        """Construct symmetric Level-2 pair features [BOTH_CONCEPTS | ONE_CONCEPT | THERMOMETER_AGREEMENT]."""
        id_to_idx = {d["id"]: i for i, d in enumerate(documents)}
        phi, base_names = self.document_concepts(documents, vectorizer)
        idx_a = np.array([id_to_idx[p["a"]] for p in pairs])
        idx_b = np.array([id_to_idx[p["b"]] for p in pairs])
        a = phi[idx_a].astype(bool)
        b = phi[idx_b].astype(bool)
        both = (a & b).astype(np.uint8)
        one = (a ^ b).astype(np.uint8)

        extra = []
        extra_names = []
        if self.mode in ("supervised_l1", "hybrid"):
            if self.tfidf_vec_ is None:
                n_cat = len(self.categories_)
                same_top1 = both[:, :n_cat].any(axis=1, keepdims=True).astype(np.uint8)
                same_top2 = both[:, n_cat:2 * n_cat].any(axis=1, keepdims=True).astype(np.uint8)
                same_top3 = both[:, 2 * n_cat:3 * n_cat].any(axis=1, keepdims=True).astype(np.uint8)
                same_high = both[:, 3 * n_cat:4 * n_cat].any(axis=1, keepdims=True).astype(np.uint8)
                same_pos_cl = both[:, 4 * n_cat:5 * n_cat].any(axis=1, keepdims=True).astype(np.uint8)
                extra.extend([same_top1, same_top2, same_top3, same_high, same_pos_cl])
                extra_names.extend([
                    "ANY_L1_Top1_Match",
                    "ANY_L1_Top2_Match",
                    "ANY_L1_Top3_Match",
                    "ANY_L1_HighScore_Match",
                    "ANY_L1_PosClause_Match",
                ])
                if self.l1_super_tm_ is not None:
                    s_off = 5 * n_cat
                    n_sup = len(SUPERGROUP_NAMES)
                    same_sup1 = both[:, s_off:s_off + n_sup].any(axis=1, keepdims=True).astype(np.uint8)
                    same_sup2 = both[:, s_off + n_sup:s_off + 2 * n_sup].any(axis=1, keepdims=True).astype(np.uint8)
                    extra.extend([same_sup1, same_sup2])
                    extra_names.extend(["ANY_Supergroup_Top1_Match", "ANY_Supergroup_Top2_Match"])
            elif self.cnb1_ is not None:
                X_tf = self.tfidf_vec_.transform([d["text"] for d in documents])
                if self.oof_scores_ is not None and all(d["id"] in self.oof_scores_ for d in documents):
                    s_cnb1 = np.array([self.oof_scores_[d["id"]][0] for d in documents], dtype=np.float32)
                    s_cnb2 = np.array([self.oof_scores_[d["id"]][1] for d in documents], dtype=np.float32)
                    s_ridge = np.array([self.oof_scores_[d["id"]][2] for d in documents], dtype=np.float32)
                else:
                    s_cnb1 = self.cnb1_.predict_log_proba(X_tf).astype(np.float32)
                    s_cnb2 = self.cnb2_.predict_log_proba(X_tf).astype(np.float32)
                    s_ridge = self.ridge_.decision_function(X_tf).astype(np.float32)

                for m_name, raw_s in (("CNB01", s_cnb1), ("CNB05", s_cnb2), ("Ridge", s_ridge)):
                    za, zb = raw_s[idx_a], raw_s[idx_b]
                    top1_eq = (np.argmax(za, axis=1) == np.argmax(zb, axis=1)).astype(np.uint8)[:, None]
                    top2_a = np.argsort(-za, axis=1)[:, :2]
                    top2_b = np.argsort(-zb, axis=1)[:, :2]
                    top2_ov = (
                        (top2_a[:, 0] == top2_b[:, 0])
                        | (top2_a[:, 0] == top2_b[:, 1])
                        | (top2_a[:, 1] == top2_b[:, 0])
                        | (top2_a[:, 1] == top2_b[:, 1])
                    ).astype(np.uint8)[:, None]
                    extra.extend([top1_eq, top2_ov])
                    extra_names.extend([f"{m_name}_Top1_Match", f"{m_name}_Top2_Overlap"])

        if self.mode == "pair_guided_or" and self.l1_pseudo_tm_ is not None and self.tfidf_vec_ is None:
            same_pt1 = both[:, :20].any(axis=1, keepdims=True).astype(np.uint8)
            same_pt2 = both[:, 20:40].any(axis=1, keepdims=True).astype(np.uint8)
            same_pt3 = both[:, 40:60].any(axis=1, keepdims=True).astype(np.uint8)
            extra.extend([same_pt1, same_pt2, same_pt3])
            extra_names.extend([
                "ANY_PseudoTopic_Top1_Match",
                "ANY_PseudoTopic_Top2_Match",
                "ANY_PseudoTopic_Top3_Match",
            ])

        cont_mat, cont_names = self._continuous_agreement_features(pairs, documents, vectorizer)
        if cont_mat is not None:
            n_bins = 16 if self.tfidf_vec_ is not None else 8
            if fit_booleanizer or self.booleanizer_ is None:
                self.booleanizer_ = QuantileBooleanizer(n_bins=n_bins).fit(cont_mat)
            therm_bits = self.booleanizer_.transform(cont_mat)
            extra.append(therm_bits)
            for col_i, cname in enumerate(cont_names):
                for bin_j, thr in enumerate(self.booleanizer_.thresholds_[col_i]):
                    extra_names.append(f"{cname}>{thr:.3f}")

        if self.tfidf_vec_ is not None:
            # Focus Level-2 Tsetlin Machine on high-signal summary match bits + thermometer agreement
            x_pair = np.hstack(extra)
            self.l2_feature_names_ = extra_names
        else:
            x_pair = np.hstack([both, one] + extra) if extra else np.hstack([both, one])
            self.l2_feature_names_ = (
                [f"BOTH.{n}" for n in base_names]
                + [f"ONE.{n}" for n in base_names]
                + extra_names
            )
        return np.ascontiguousarray(x_pair, dtype=np.uint8)

    def fit(self, train_docs, train_pairs, vectorizer, categories=None, val_docs=None, val_pairs=None):
        self.fit_level1(train_docs, vectorizer, categories, train_pairs=train_pairs)
        x_tr = self.pair_concept_features(train_pairs, train_docs, vectorizer, fit_booleanizer=True)
        y_tr = np.array([p["label"] for p in train_pairs], dtype=np.int32)

        x_val, y_val = None, None
        if val_docs is not None and val_pairs is not None:
            x_val = self.pair_concept_features(val_pairs, val_docs, vectorizer, fit_booleanizer=False)
            y_val = np.array([p["label"] for p in val_pairs], dtype=np.int32)

        self.l2_tm_ = TsetlinMachine(
            clauses_per_class=self.l2_clauses_per_class,
            T=self.l2_T,
            s=self.l2_s,
            boost_true_positive=True,
            weighted=True,
            literal_budget=12,
            random_state=self.seed,
            epochs=self.l2_epochs,
        )
        self.l2_tm_.fit(x_tr, y_tr, X_val=x_val, y_val=y_val)
        return self

    def predict(self, pairs, documents, vectorizer):
        x = self.pair_concept_features(pairs, documents, vectorizer, fit_booleanizer=False)
        return self.l2_tm_.predict(x)

    def decision_margin(self, pairs, documents, vectorizer):
        x = self.pair_concept_features(pairs, documents, vectorizer, fit_booleanizer=False)
        scores = np.asarray(self.l2_tm_.decision_function(x), dtype=np.int64)
        return scores[:, 1] - scores[:, 0]

    def level1_sample_rules(self, top_per_class=1):
        """Extract readable positive Level-1 topic clauses showing word-to-topic rules."""
        if self.l1_tm_ is None:
            return []
        packed = self.l1_tm_.to_packed()
        inc = packed.include_
        w = packed.weights_
        n = len(self.words_)
        out = []
        for c, cname in enumerate(self.categories_):
            pos_lit_counts = inc[:, :n].sum(axis=1)
            candidates = np.flatnonzero((w[:, c] > 0) & (pos_lit_counts > 0))
            if not len(candidates):
                candidates = np.flatnonzero(w[:, c] > 0)
            order = candidates[np.argsort(-w[candidates, c])][:top_per_class]
            for cl_idx in order:
                pos_words = [self.words_[j] for j in np.flatnonzero(inc[cl_idx, :n])]
                neg_words = [f"NOT {self.words_[j]}" for j in np.flatnonzero(inc[cl_idx, n:])]
                conds = pos_words + neg_words[:4]
                if len(neg_words) > 4:
                    conds.append(f"... (+{len(neg_words) - 4} NOT words)")
                out.append({
                    "topic": cname,
                    "clause_index": int(cl_idx),
                    "weight": int(w[cl_idx, c]),
                    "positive_words": pos_words,
                    "total_literals": int(inc[cl_idx].sum()),
                    "rule_text": f"IF {' AND '.join(conds)} THEN +{int(w[cl_idx, c])} {cname}",
                })
        return out

    def level2_top_rules(self, top=25):
        rules = self.l2_tm_.rules(feature_names=self.l2_feature_names_, top=top, best=True)
        return [str(r) for r in rules]
