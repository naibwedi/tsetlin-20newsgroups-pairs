"""Trace two seed-42 validation decisions from the frozen dashboard clause masks."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bittm.model import PackedModel

from pipeline import cosine_scores, feature_names, make_vectorizer, metrics, pair_features


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def explain(index, pair, row, cosine, cutoff, mask, weights, words, categories, native):
    literals = np.r_[row, 1 - row].astype(bool)
    fired = mask.any(axis=1) & np.all(~mask | literals, axis=1)
    score = weights[fired].sum(axis=0)
    np.testing.assert_array_equal(score, native)
    swing = weights[:, 1] - weights[:, 0]
    names = feature_names(words, "sym")
    names += ["NOT " + name for name in names]
    clauses = []
    for direction in (1, -1):
        candidates = np.flatnonzero(fired & (np.sign(swing) == direction))
        if candidates.size:
            clause = int(candidates[0])
            conditions = np.flatnonzero(mask[clause])
            clauses.append(dict(clause=clause, weights=weights[clause].tolist(),
                                signed_contribution=int(swing[clause]), length=int(conditions.size),
                                first_conditions=[names[i] for i in conditions[:3]]))
    blocked = None
    for clause in np.flatnonzero(~fired):
        failing = np.flatnonzero(mask[clause] & ~literals)
        if failing.size:
            blocked = dict(clause=int(clause), first_false_condition=names[failing[0]])
            break
    return dict(manifest_index=index, a=pair["a"], b=pair["b"],
                category_a=categories[pair["category_a"]], category_b=categories[pair["category_b"]],
                truth=pair["label"], shared_tm_words=[words[i] for i in np.flatnonzero(row[:len(words)])],
                exclusive_tm_word_count=int(row[len(words):].sum()), cosine=float(cosine),
                cosine_threshold=cutoff, cosine_prediction=int(cosine >= cutoff),
                scores=score.tolist(), margin=int(score[1] - score[0]), tm_prediction=int(score.argmax()),
                fired_clauses=int(fired.sum()), votes_toward_same=int(swing[fired & (swing > 0)].sum()),
                votes_toward_different=int(-swing[fired & (swing < 0)].sum()),
                example_fired_clauses=clauses, example_blocked_clause=blocked)


def plot(records, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, layout="constrained")
    all_scores = [score for record in records for score in record["scores"]]
    for axis, record in zip(axes, records):
        bars = axis.bar(["Different", "Same"], record["scores"], color=["#496783", "#00816d"], width=0.55)
        axis.axhline(0, color="#3c4958", linewidth=0.7)
        axis.bar_label(bars, fmt="%+d", padding=3)
        axis.set_ylim(min(all_scores)-1.5, max(all_scores)+1.5)
        truth = "same" if record["truth"] else "different"
        axis.set_title(f"Validation row {record['manifest_index']} · actual {truth}\n"
                       f"TM margin +{record['margin']} → same")
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Signed class score (votes)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(project, output):
    source = project / "outputs/vanilla-v2"
    export = read(project / "docs/dashboard_data.json")
    saved = read(source / "validation_metrics.json")
    cfg = saved["config"]
    if cfg["seed"] != 42 or cfg["features"] != "sym" or any(
        cfg[key] != value for key, value in export["config"].items()
    ):
        raise ValueError("Dashboard and validation run must have the same seed-42 symmetric configuration")
    train = read(source / "train_documents.json")
    docs = read(source / "validation_documents.json")
    pairs = read(source / "validation_pairs.json")
    categories = read(source / "categories.json")
    count = make_vectorizer("count", cfg["vocabulary"], cfg["vocab_mode"])
    count.set_params(vocabulary=export["words"])
    x = pair_features(pairs, docs, count, "sym").astype(np.uint8)
    tfidf = make_vectorizer("tfidf", cfg["vocabulary"], cfg["vocab_mode"]).fit([d["text"] for d in train])
    cosine = cosine_scores(pairs, docs, tfidf)
    cutoff = saved["cosine_threshold"]
    mask = np.stack([np.unpackbits(np.frombuffer(base64.b64decode(c), dtype=np.uint8),
                                   count=2 * export["nFeatures"]) for c in export["clauses"]])
    weights = np.array(export["weights"])
    packed = PackedModel(mask, weights)
    scores = packed.decision_function(x)
    pred = packed.predict(x)
    labels = [p["label"] for p in pairs]
    for name, guesses in (("tm", pred), ("cosine", cosine >= cutoff)):
        if metrics(labels, guesses)["confusion_matrix"] != saved[name]["confusion_matrix"]:
            raise ValueError(f"{name} decisions no longer reproduce the frozen validation confusion matrix")
    words = export["words"]
    shared = x[:, :len(words)].sum(axis=1)
    records = []
    for label in (1, 0):
        eligible = (pred == 1) & (cosine < cutoff) & (np.asarray(labels) == label) & (shared == 1)
        index = int(np.flatnonzero(eligible)[0])
        record = explain(index, pairs[index], x[index], cosine[index], cutoff, mask.astype(bool),
                         weights, words, categories, scores[index])
        records.append(record)
    for folder in ("results", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    evidence = dict(config=cfg, selection="First validation manifest row of each true label with exactly one shared "
                    "TM vocabulary word, TM predicting same, cosine predicting different.", examples=records)
    (output / "results/worked_pairs.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    plot(records, output / "figures/worked_pair_votes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Defaults to the project's docs directory")
    args = parser.parse_args()
    main(args.project, args.output or args.project / "docs")
