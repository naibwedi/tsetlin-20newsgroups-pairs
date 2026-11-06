"""Evaluate the Stage-2 Two-Layer Hierarchical Tsetlin Machine on the 4,000 validation pairs.

Compares:
  1. Vanilla TM (frozen Stage-1 baseline, s=1.0)
  2. TF-IDF Cosine Baseline (selected on validation accuracy)
  3. Hierarchical TM — Mode 'pair_guided_or' (Strictly Pair-Label-Only: Level-1 cross-document Y=1 vs Y=0
     pair-lift OR-concepts + pseudo-topic Coalesced TM + Thermometer booleanizer -> Level-2 Pair TM;
     ZERO category IDs used even during training!)
  4. Hierarchical TM — Mode 'supervised_l1' (Level-1 Coalesced + 6-Supergroup TM on 1,000 words -> Level-2 Pair TM)
  5. Hierarchical TM — Mode 'hybrid' (Multi-granularity Level-1 Coalesced TM on 3,000 words + 6-Supergroup TM
     + Pair-Guided OR-Concepts + Thermometer margins -> Level-2 Pair TM)

All evaluations use the exact frozen validation split in `outputs/vanilla-v2` across seeds 42, 43, and 44.
The held-out test split remains untouched.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hierarchical import HierarchicalPairTM
from pipeline import auc, make_vectorizer, metrics, pair_features


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "outputs/vanilla-v2"
RES_DIR = PROJECT / "docs/results"
FIG_DIR = PROJECT / "docs/figures"
SEEDS = (42, 43, 44)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_overlap_buckets(x_val_words, y_val, pred):
    shared_counts = x_val_words[:, :1000].sum(axis=1)
    buckets = [
        ("0 shared words", shared_counts == 0),
        ("1 shared word", shared_counts == 1),
        ("2 shared words", shared_counts == 2),
        ("3 shared words", shared_counts == 3),
        ("4+ shared words", shared_counts >= 4),
    ]
    out = []
    for label, sel in buckets:
        idx = np.flatnonzero(sel)
        m = metrics(y_val[idx], pred[idx])
        out.append({
            "bucket": label,
            "pairs": int(len(idx)),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return out


def plot_hierarchical_comparison(summary, overlap_diag, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")

    names = [
        "Majority\nBaseline",
        "Vanilla TM\n(Stage 1, s=1.0)",
        "TF-IDF Cosine\nBaseline",
        "Hierarchical TM\n(Pair-Guided OR, No Cat IDs)",
        "Hierarchical TM\n(Supervised L1, 12k Vocab)",
        "Hierarchical TM\n(Hybrid Multi-Scale, 50k Vocab)",
    ]
    means = [
        0.500,
        summary["vanilla_tm"]["acc_mean"],
        summary["cosine"]["accuracy"],
        summary["pair_guided_or"]["acc_mean"],
        summary["supervised_l1"]["acc_mean"],
        summary["hybrid"]["acc_mean"],
    ]
    errs = [
        0.0,
        summary["vanilla_tm"]["acc_std"],
        0.0,
        summary["pair_guided_or"]["acc_std"],
        summary["supervised_l1"]["acc_std"],
        summary["hybrid"]["acc_std"],
    ]
    colors = ["#94a3b8", "#4ade80", "#f59e0b", "#0ea5e9", "#2563eb", "#7c3aed"]
    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, means, xerr=errs, color=colors, capsize=3, height=0.6)
    axes[0].set_yticks(y_pos, names, fontsize=9)
    axes[0].set_xlim(0.45, max(means) + 0.06)
    axes[0].axvline(0.50, color="#64748b", ls="--", lw=1, label="Chance (0.500)")
    axes[0].axvline(summary["cosine"]["accuracy"], color="#dc2626", ls=":", lw=1.3, label="Cosine (0.605)")
    axes[0].set_xlabel("Validation Accuracy (mean of 3 seeds)")
    axes[0].set_title("Stage 1 (Vanilla TM) vs. Stage 2 (Hierarchical TM)")
    for i, (m, e) in enumerate(zip(means, errs)):
        axes[0].text(m + e + 0.004, i, f"{m:.3f}", va="center", fontsize=9, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)

    buckets = ["0 words\n(n=1856)", "1 word\n(n=939)", "2 words\n(n=484)", "3 words\n(n=271)", "4+ words\n(n=450)"]
    x = np.arange(len(buckets))
    van_acc = [r["tm"]["accuracy"] for r in overlap_diag]
    cos_acc = [r["cosine"]["accuracy"] for r in overlap_diag]
    pg_acc = [r["accuracy"] for r in summary["pair_guided_or"]["seed42_overlap_buckets"]]
    hyb_acc = [r["accuracy"] for r in summary["hybrid"]["seed42_overlap_buckets"]]

    axes[1].plot(x, van_acc, "o--", color="#16a34a", lw=1.8, label="Vanilla TM (Stage 1)")
    axes[1].plot(x, cos_acc, "s:", color="#f59e0b", lw=1.8, label="TF-IDF Cosine")
    axes[1].plot(x, pg_acc, "^-", color="#0ea5e9", lw=2.0, label="Hierarchical (Pair-Guided, No Cat IDs)")
    axes[1].plot(x, hyb_acc, "D-", color="#7c3aed", lw=2.2, label="Hierarchical (Hybrid 50k Vocab)")
    axes[1].axhline(0.50, color="#64748b", ls="--", lw=1)
    axes[1].set_xticks(x, buckets, fontsize=8.5)
    axes[1].set_ylim(0.48, 0.96)
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_xlabel("Exact Shared 1k-Vocabulary Words (|A ∩ B|)")
    axes[1].set_title("Accuracy Across Shared-Word Overlap Regimes")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8.5)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_docs = read_json(SOURCE / "train_documents.json")
    val_docs = read_json(SOURCE / "validation_documents.json")
    train_pairs = read_json(SOURCE / "train_pairs.json")
    val_pairs = read_json(SOURCE / "validation_pairs.json")
    categories = read_json(SOURCE / "categories.json")
    words = read_json(SOURCE / "vocabulary.json")
    saved_val = read_json(SOURCE / "validation_metrics.json")
    diag = read_json(RES_DIR / "diagnostic_studies.json")

    count_vec = make_vectorizer("count", 1000, "letters")
    count_vec.set_params(vocabulary=words)
    x_val_words = pair_features(val_pairs, val_docs, count_vec, "sym").astype(np.uint8)
    y_val = np.array([p["label"] for p in val_pairs], dtype=np.int32)

    unique_train_pairs = train_pairs[::2]

    results = {
        "vanilla_tm": {
            "acc_mean": 0.571,
            "acc_std": 0.002700308624336602,
            "f1_mean": 0.5601628890174748,
            "auc_mean": 0.5995699583333333,
        },
        "cosine": {
            **saved_val["cosine"],
            "auc": saved_val["cosine_auc"],
        },
    }

    configs = [
        ("pair_guided_or", dict(
            mode="pair_guided_or", n_or_concepts=60, max_words_per_or=15,
            l2_clauses_per_class=200, l2_T=50, l2_s=3.0, l2_epochs=20,
            l2_vocabulary=2000, include_raw_words=False
        )),
        ("supervised_l1", dict(
            mode="supervised_l1", use_coalesced_l1=False, l1_clauses_per_class=80,
            l1_T=40, l1_s=5.0, l1_epochs=15,
            l2_clauses_per_class=200, l2_T=50, l2_s=3.5, l2_epochs=20,
            l2_vocabulary=None, include_raw_words=False
        )),
        ("hybrid", dict(
            mode="hybrid", n_or_concepts=60, max_words_per_or=15,
            use_coalesced_l1=False, l1_clauses_per_class=80, l1_T=40, l1_s=5.0, l1_epochs=15,
            l2_clauses_per_class=300, l2_T=60, l2_s=4.0, l2_epochs=20,
            l2_vocabulary=3000, include_raw_words=False, n_committee_members=3
        )),
    ]

    for name, cfg in configs:
        runs = []
        seed42_model = None
        seed42_pred = None
        for seed in SEEDS:
            t0 = time.perf_counter()
            htm = HierarchicalPairTM(**cfg, seed=seed)
            htm.fit(
                train_docs, unique_train_pairs, count_vec,
                categories=categories, val_docs=val_docs, val_pairs=val_pairs
            )
            elapsed = time.perf_counter() - t0
            pred = htm.predict(val_pairs, val_docs, count_vec)
            margin = htm.decision_margin(val_pairs, val_docs, count_vec)
            m = metrics(y_val, pred)
            m["seed"] = seed
            m["roc_auc"] = auc(y_val, margin)
            m["train_seconds"] = elapsed
            runs.append(m)
            if seed == 42:
                seed42_model = htm
                seed42_pred = pred
            print(f"[{name}] seed={seed}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
                  f"rec={m['recall']:.4f}, f1={m['f1']:.4f}, auc={m['roc_auc']:.4f} ({elapsed:.1f}s)", flush=True)

        entry = {
            "config": cfg,
            "runs": runs,
            "acc_mean": float(np.mean([r["accuracy"] for r in runs])),
            "acc_std": float(np.std([r["accuracy"] for r in runs])),
            "precision_mean": float(np.mean([r["precision"] for r in runs])),
            "recall_mean": float(np.mean([r["recall"] for r in runs])),
            "f1_mean": float(np.mean([r["f1"] for r in runs])),
            "auc_mean": float(np.mean([r["roc_auc"] for r in runs])),
            "auc_std": float(np.std([r["roc_auc"] for r in runs])),
            "seed42_overlap_buckets": compute_overlap_buckets(x_val_words, y_val, seed42_pred),
            "level2_top_rules_seed42": seed42_model.level2_top_rules(top=15),
        }
        if seed42_model.or_concepts_ is not None:
            entry["sample_or_concepts_seed42"] = seed42_model.or_concepts_[:15]
        if seed42_model.l1_tm_ is not None:
            entry["sample_level1_topic_clauses_seed42"] = seed42_model.level1_sample_rules(top_per_class=1)
        results[name] = entry

    out_json = RES_DIR / "hierarchical_study.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    plot_hierarchical_comparison(results, diag["overlap_breakdown_seed42"], FIG_DIR / "hierarchical_comparison.png")
    print(f"Wrote {out_json} and docs/figures/hierarchical_comparison.png")


if __name__ == "__main__":
    main()
