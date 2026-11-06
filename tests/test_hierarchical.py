import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hierarchical import HierarchicalPairTM, discover_or_concepts, evaluate_or_concepts
from pipeline import make_vectorizer


def synthetic_corpus():
    docs = []
    # Topic 0: space orbit rocket; Topic 1: hockey goal puck; Topic 2: circuit battery voltage
    templates = {
        0: ["space orbit rocket", "space shuttle orbit", "rocket launch orbit", "nasa space shuttle"],
        1: ["hockey goal puck", "hockey team goal", "puck ice hockey", "team season puck"],
        2: ["circuit battery voltage", "pcb circuit voltage", "battery power circuit", "pcb layout battery"],
    }
    idx = 0
    for cat, phrases in templates.items():
        for rep in range(6):
            for text in phrases:
                docs.append({"id": f"d:{idx}", "text": text, "category": cat})
                idx += 1
    return docs


def test_discover_and_evaluate_or_concepts():
    docs = synthetic_corpus()
    vec = make_vectorizer("count", 20, "letters").fit([d["text"] for d in docs])
    words = vec.get_feature_names_out().tolist()
    x_docs = vec.transform([d["text"] for d in docs]).toarray().astype(np.uint8)
    masks, concepts = discover_or_concepts(x_docs, words, n_concepts=3, max_words_per_concept=5, seed=42)
    assert masks.shape == (3, len(words))
    assert len(concepts) == 3
    out = evaluate_or_concepts(x_docs, masks)
    assert out.shape == (len(docs), 3)
    assert set(np.unique(out)).issubset({0, 1})


def test_hierarchical_pair_tm_modes_and_symmetry():
    docs = synthetic_corpus()
    vec = make_vectorizer("count", 20, "letters").fit([d["text"] for d in docs])
    pairs = []
    for i in range(0, len(docs) - 1, 2):
        a, b = docs[i], docs[(i + 3) % len(docs)]
        pairs.append({
            "a": a["id"], "b": b["id"],
            "category_a": a["category"], "category_b": b["category"],
            "label": int(a["category"] == b["category"]),
        })
    reversed_pairs = [
        {"a": p["b"], "b": p["a"], "category_a": p["category_b"], "category_b": p["category_a"], "label": p["label"]}
        for p in pairs
    ]

    for mode in ("or_concepts", "supervised_l1", "hybrid"):
        htm = HierarchicalPairTM(
            mode=mode, n_or_concepts=3, max_words_per_or=4,
            l1_clauses_per_class=4, l1_T=4, l1_s=2.0, l1_epochs=3,
            l2_clauses_per_class=6, l2_T=4, l2_s=2.0, l2_epochs=3,
            include_raw_words=False, seed=42
        )
        htm.fit(docs, pairs, vec, categories=["space", "hockey", "electronics"])
        f_ab = htm.pair_concept_features(pairs, docs, vec)
        f_ba = htm.pair_concept_features(reversed_pairs, docs, vec)
        # Level-2 features must be 100% symmetric (order-independent)
        np.testing.assert_array_equal(f_ab, f_ba)
        pred_ab = htm.predict(pairs, docs, vec)
        pred_ba = htm.predict(reversed_pairs, docs, vec)
        np.testing.assert_array_equal(pred_ab, pred_ba)
        assert len(htm.level2_top_rules(top=5)) > 0
