"""Diagnostic studies on the validation split (without touching test or altering the frozen submission):
1. Overlap-bucket breakdown (0, 1, 2, 3, >=4 shared vocabulary words): TM vs TF-IDF Cosine on seed 42.
2. Unique-pairs vs orientation-augmented training (10,000 unique x 2 orientations vs 20,000 unique unordered pairs, 3 seeds).
3. Specificity s diagnostic curve (s = 1.0, 2.0, 5.0, 10.0) and literal_budget = 10 at s = 1.0 (3 seeds),
   measuring accuracy, positive vs negated literal counts, and median clause length.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bittm import TsetlinMachine
from bittm.model import PackedModel
import base64

from pipeline import auc, cosine_scores, make_pairs, make_vectorizer, metrics, pair_features


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "outputs/vanilla-v2"
RES_DIR = PROJECT / "docs/results"
FIG_DIR = PROJECT / "docs/figures"
SEEDS = (42, 43, 44)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_overlap_breakdown():
    """Compare frozen seed-42 TM and TF-IDF cosine across shared-word count buckets on the 4,000 validation pairs."""
    export = read_json(PROJECT / "docs/dashboard_data.json")
    saved = read_json(SOURCE / "validation_metrics.json")
    train_docs = read_json(SOURCE / "train_documents.json")
    val_docs = read_json(SOURCE / "validation_documents.json")
    val_pairs = read_json(SOURCE / "validation_pairs.json")
    words = export["words"]

    count = make_vectorizer("count", 1000, "letters")
    count.set_params(vocabulary=words)
    x_val = pair_features(val_pairs, val_docs, count, "sym").astype(np.uint8)
    y_val = np.array([p["label"] for p in val_pairs], dtype=np.int32)

    tfidf = make_vectorizer("tfidf", 1000, "letters").fit([d["text"] for d in train_docs])
    cos_val = cosine_scores(val_pairs, val_docs, tfidf)
    cutoff = saved["cosine_threshold"]
    cos_pred = (cos_val >= cutoff).astype(np.int32)

    mask = np.stack([
        np.unpackbits(np.frombuffer(base64.b64decode(c), dtype=np.uint8), count=2 * export["nFeatures"])
        for c in export["clauses"]
    ])
    weights = np.array(export["weights"])
    packed = PackedModel(mask, weights)
    tm_pred = packed.predict(x_val).astype(np.int32)

    shared_counts = x_val[:, :len(words)].sum(axis=1)
    buckets = [
        ("0 shared words", shared_counts == 0),
        ("1 shared word", shared_counts == 1),
        ("2 shared words", shared_counts == 2),
        ("3 shared words", shared_counts == 3),
        ("4+ shared words", shared_counts >= 4),
    ]
    rows = []
    for label, sel in buckets:
        idx = np.flatnonzero(sel)
        yb = y_val[idx]
        tmb = tm_pred[idx]
        cosb = cos_pred[idx]
        tm_m = metrics(yb, tmb)
        cos_m = metrics(yb, cosb)
        rows.append({
            "bucket": label,
            "pairs": int(len(idx)),
            "share_of_validation": float(len(idx) / len(y_val)),
            "true_same_rate": float(yb.mean()),
            "tm_predicted_same_rate": float(tmb.mean()),
            "cosine_predicted_same_rate": float(cosb.mean()),
            "tm": tm_m,
            "cosine": cos_m,
        })
    return rows


def run_unique_pairs_study(train_docs, val_docs, val_pairs, count_vec, x_val, y_val):
    """Compare 10,000 unique pairs x 2 orientations (20,000 rows) vs 20,000 unique unordered pairs (augment=False)."""
    pairs_20k_unique = make_pairs(train_docs, 20000, seed=42, augment=False)
    x_tr_unique = pair_features(pairs_20k_unique, train_docs, count_vec, "sym").astype(np.uint8)
    y_tr_unique = np.array([p["label"] for p in pairs_20k_unique], dtype=np.int32)

    train_pairs_aug = read_json(SOURCE / "train_pairs.json")
    x_tr_aug = pair_features(train_pairs_aug, train_docs, count_vec, "sym").astype(np.uint8)
    y_tr_aug = np.array([p["label"] for p in train_pairs_aug], dtype=np.int32)

    out = {}
    for name, x_tr, y_tr in (
        ("10k_unique_x2_orientations", x_tr_aug, y_tr_aug),
        ("20k_unique_unordered", x_tr_unique, y_tr_unique),
    ):
        runs = []
        for seed in SEEDS:
            tm = TsetlinMachine(
                clauses_per_class=100, T=25, s=1.0, boost_true_positive=True,
                weighted=False, random_state=seed, epochs=20
            )
            tm.fit(x_tr, y_tr)
            pred = tm.predict(x_val)
            scores = np.asarray(tm.decision_function(x_val), dtype=np.int64)
            margin = scores[:, 1] - scores[:, 0]
            m = metrics(y_val, pred)
            m["seed"] = seed
            m["roc_auc"] = auc(y_val, margin)
            runs.append(m)
        out[name] = {
            "runs": runs,
            "acc_mean": float(np.mean([r["accuracy"] for r in runs])),
            "acc_std": float(np.std([r["accuracy"] for r in runs])),
            "f1_mean": float(np.mean([r["f1"] for r in runs])),
            "auc_mean": float(np.mean([r["roc_auc"] for r in runs])),
        }

    # Also check best-validation-epoch checkpointing on the frozen 10k x 2 setup
    best_epoch_runs = []
    for seed in SEEDS:
        tm = TsetlinMachine(
            clauses_per_class=100, T=25, s=1.0, boost_true_positive=True,
            weighted=False, random_state=seed, epochs=20
        )
        tm.fit(x_tr_aug, y_tr_aug, X_val=x_val, y_val=y_val)
        pred = tm.predict(x_val)
        scores = np.asarray(tm.decision_function(x_val), dtype=np.int64)
        margin = scores[:, 1] - scores[:, 0]
        m = metrics(y_val, pred)
        m["seed"] = seed
        m["best_epoch"] = int(tm.best_epoch_)
        m["roc_auc"] = auc(y_val, margin)
        best_epoch_runs.append(m)
    out["10k_x2_best_epoch_checkpoint"] = {
        "runs": best_epoch_runs,
        "acc_mean": float(np.mean([r["accuracy"] for r in best_epoch_runs])),
        "acc_std": float(np.std([r["accuracy"] for r in best_epoch_runs])),
        "f1_mean": float(np.mean([r["f1"] for r in best_epoch_runs])),
        "auc_mean": float(np.mean([r["roc_auc"] for r in best_epoch_runs])),
    }
    return out


def run_specificity_and_budget_study(x_tr, y_tr, x_val, y_val):
    """Compare s in {1.0, 2.0, 3.0, 5.0, 10.0} and literal_budget in {None, 10} at s=1.0."""
    configs = [
        ("s=1.0 (frozen baseline)", dict(s=1.0, literal_budget=None)),
        ("s=2.0", dict(s=2.0, literal_budget=None)),
        ("s=3.0", dict(s=3.0, literal_budget=None)),
        ("s=5.0", dict(s=5.0, literal_budget=None)),
        ("s=10.0", dict(s=10.0, literal_budget=None)),
        ("s=1.0, literal_budget=10", dict(s=1.0, literal_budget=10)),
    ]
    n_feat = x_tr.shape[1]
    rows = []
    for label, kw in configs:
        accs, f1s, aucs, pos_counts, neg_counts, med_lens = [], [], [], [], [], []
        for seed in SEEDS:
            tm = TsetlinMachine(
                clauses_per_class=100, T=25, boost_true_positive=True,
                weighted=False, random_state=seed, epochs=20, **kw
            )
            tm.fit(x_tr, y_tr)
            pred = tm.predict(x_val)
            scores = np.asarray(tm.decision_function(x_val), dtype=np.int64)
            margin = scores[:, 1] - scores[:, 0]
            m = metrics(y_val, pred)
            accs.append(m["accuracy"])
            f1s.append(m["f1"])
            aucs.append(auc(y_val, margin))

            inc = tm.clauses_
            pos_lit = int(inc[:, :n_feat].sum())
            neg_lit = int(inc[:, n_feat:].sum())
            lens = inc.sum(axis=1)
            pos_counts.append(pos_lit)
            neg_counts.append(neg_lit)
            med_lens.append(float(np.median(lens)))
            print(f"{label} seed={seed}: acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, "
                  f"pos_lit={pos_lit}, neg_lit={neg_lit}, med_len={np.median(lens):.1f}", flush=True)
        rows.append({
            "label": label,
            "s": kw["s"],
            "literal_budget": kw["literal_budget"],
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "auc_mean": float(np.mean(aucs)),
            "positive_literals_seed42": pos_counts[0],
            "negated_literals_seed42": neg_counts[0],
            "median_clause_length_seed42": med_lens[0],
            "accs": accs,
        })
    return rows


def plot_overlap_breakdown(overlap_rows, out_path):
    labels = [r["bucket"].replace(" shared words", "").replace(" shared word", "") for r in overlap_rows]
    counts = [r["pairs"] for r in overlap_rows]
    tm_acc = [r["tm"]["accuracy"] for r in overlap_rows]
    cos_acc = [r["cosine"]["accuracy"] for r in overlap_rows]
    tm_rec = [r["tm"]["recall"] for r in overlap_rows]
    cos_rec = [r["cosine"]["recall"] for r in overlap_rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")

    b1 = axes[0].bar(x - w / 2, tm_acc, w, color="#16a34a", label="Vanilla TM (seed 42)")
    b2 = axes[0].bar(x + w / 2, cos_acc, w, color="#38bdf8", label="TF-IDF Cosine")
    axes[0].set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(labels, counts)])
    axes[0].set_ylim(0.4, 0.85)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_xlabel("Shared TM vocabulary words in pair")
    axes[0].set_title("Accuracy by shared-word count")
    axes[0].axhline(0.5, color="#64748b", ls="--", lw=1)
    axes[0].legend(loc="upper left")
    axes[0].bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    axes[0].bar_label(b2, fmt="%.2f", fontsize=8, padding=2)

    b3 = axes[1].bar(x - w / 2, tm_rec, w, color="#16a34a", label="Vanilla TM recall")
    b4 = axes[1].bar(x + w / 2, cos_rec, w, color="#38bdf8", label="TF-IDF Cosine recall")
    axes[1].set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(labels, counts)])
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_ylabel("Recall on true same-category pairs")
    axes[1].set_xlabel("Shared TM vocabulary words in pair")
    axes[1].set_title("Same-category recall by shared-word count")
    axes[1].legend(loc="upper left")
    axes[1].bar_label(b3, fmt="%.2f", fontsize=8, padding=2)
    axes[1].bar_label(b4, fmt="%.2f", fontsize=8, padding=2)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_s_diagnostic(s_rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    s_only = [r for r in s_rows if r["literal_budget"] is None]
    s_vals = [str(r["s"]) for r in s_only]
    means = [r["acc_mean"] for r in s_only]
    stds = [r["acc_std"] for r in s_only]

    axes[0].errorbar(s_vals, means, yerr=stds, marker="o", color="#16a34a", capsize=4, lw=2, label="Vanilla TM (3 seeds)")
    bud = next(r for r in s_rows if r["literal_budget"] is not None)
    axes[0].errorbar(["1.0\n(budget=10)"], [bud["acc_mean"]], yerr=[bud["acc_std"]], marker="s", color="#2563eb", capsize=4, label="s=1.0, budget=10")
    axes[0].axhline(0.6045, color="#dc2626", ls="-", label="TF-IDF Cosine (0.605)")
    axes[0].axhline(0.5, color="#64748b", ls="--", label="Chance (0.500)")
    axes[0].set_ylim(0.48, 0.63)
    axes[0].set_xlabel("Specificity s (and literal budget check)")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("Effect of specificity s on validation accuracy")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].grid(alpha=0.3)

    labels = [r["label"].replace(" (frozen baseline)", "").replace(", literal_", "\n") for r in s_rows]
    pos = [r["positive_literals_seed42"] for r in s_rows]
    neg = [r["negated_literals_seed42"] for r in s_rows]
    x = np.arange(len(labels))
    w = 0.38
    b1 = axes[1].bar(x - w / 2, pos, w, color="#16a34a", label="Positive literals (BOTH / ONE)")
    b2 = axes[1].bar(x + w / 2, neg, w, color="#dc2626", label="Negated literals (NOT)")
    axes[1].set_yscale("symlog", linthresh=10)
    axes[1].set_xticks(x, labels, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Total included literals across 200 clauses (log scale)")
    axes[1].set_title("Positive literals emerge only when s > 1.0")
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_docs = read_json(SOURCE / "train_documents.json")
    val_docs = read_json(SOURCE / "validation_documents.json")
    train_pairs = read_json(SOURCE / "train_pairs.json")
    val_pairs = read_json(SOURCE / "validation_pairs.json")
    words = read_json(SOURCE / "vocabulary.json")

    count_vec = make_vectorizer("count", 1000, "letters")
    count_vec.set_params(vocabulary=words)
    x_tr = pair_features(train_pairs, train_docs, count_vec, "sym").astype(np.uint8)
    y_tr = np.array([p["label"] for p in train_pairs], dtype=np.int32)
    x_val = pair_features(val_pairs, val_docs, count_vec, "sym").astype(np.uint8)
    y_val = np.array([p["label"] for p in val_pairs], dtype=np.int32)

    print("1. Computing overlap breakdown...")
    overlap_rows = compute_overlap_breakdown()
    plot_overlap_breakdown(overlap_rows, FIG_DIR / "overlap_breakdown.png")

    print("2. Running unique-pairs vs orientation-augmented study...")
    unique_study = run_unique_pairs_study(train_docs, val_docs, val_pairs, count_vec, x_val, y_val)

    print("3. Running specificity s and literal_budget diagnostic study...")
    s_rows = run_specificity_and_budget_study(x_tr, y_tr, x_val, y_val)
    plot_s_diagnostic(s_rows, FIG_DIR / "s_diagnostic.png")

    report = {
        "overlap_breakdown_seed42": overlap_rows,
        "training_diversity_and_checkpoint_study": unique_study,
        "specificity_and_budget_study": s_rows,
    }
    out_file = RES_DIR / "diagnostic_studies.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
