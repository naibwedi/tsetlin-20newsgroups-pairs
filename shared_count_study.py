"""Compare the frozen vanilla TM with three extra shared-word count bits.

From the project root, run: python shared_count_study.py
Inputs are outputs/vanilla-v2's saved training and validation manifests.
The experiment reads those manifests in their saved order without resampling.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import bittm
import numpy as np
from bittm import TsetlinMachine

from pipeline import auc, make_vectorizer, metrics, pair_features


SEEDS = (42, 43, 44)
COUNT_CUTOFFS = (1, 2, 5)
TM_SETTINGS = dict(clauses_per_class=100, T=25, s=1.0, weighted=False,
                   boost_true_positive=True, epochs=20)


def append_count_bits(symmetric_features, vocabulary_size):
    """The first feature block contains one bit per distinct shared word."""
    if symmetric_features.shape[1] != vocabulary_size * 2:
        raise ValueError("Expected BOTH and ONE blocks of equal vocabulary size")
    shared = symmetric_features[:, :vocabulary_size].sum(axis=1)
    bits = shared[:, None] >= np.asarray(COUNT_CUTOFFS)
    return np.column_stack((symmetric_features, bits))


def run(manifests):
    hashes = {}

    def load_file(name):
        raw = (manifests / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    train_docs = load_file("train_documents.json")
    vectorizer = make_vectorizer("count", 1000, "letters")
    vectorizer.fit([doc["text"] for doc in train_docs])
    words = vectorizer.get_feature_names_out().tolist()
    if words != load_file("vocabulary.json"):
        raise ValueError("Training vocabulary differs from the saved frozen vocabulary")
    matrices, labels = {}, {}
    for split, documents in (("train", train_docs), ("validation", load_file("validation_documents.json"))):
        pairs = load_file(f"{split}_pairs.json")
        matrices[split] = pair_features(pairs, documents, vectorizer, mode="sym").astype(np.uint8)
        labels[split] = np.array([pair["label"] for pair in pairs], dtype=np.int32)

    truth = labels["validation"]
    results = {}
    base_predictions = {}
    for variant in ("sym", "sym_plus_counts"):
        train = matrices["train"]
        validation = matrices["validation"]
        if variant == "sym_plus_counts":
            train = append_count_bits(train, len(words))
            validation = append_count_bits(validation, len(words))
        runs = []
        for seed in SEEDS:
            tm = TsetlinMachine(**TM_SETTINGS, random_state=seed)
            start = perf_counter()
            tm.fit(train, labels["train"])
            duration = perf_counter() - start
            prediction = tm.predict(validation)
            scores = np.asarray(tm.decision_function(validation), dtype=np.int64)
            margin = scores[:, 1] - scores[:, 0]
            if not np.array_equal(margin > 0, prediction):
                raise ValueError("Native predictions disagree with signed score margin")
            row = metrics(truth, prediction)
            row.update(seed=seed, roc_auc=auc(truth, margin), training_seconds=duration)
            if variant == "sym":
                base_predictions[seed] = prediction
            else:
                base_correct = base_predictions[seed] == truth
                row["formerly_wrong_now_correct"] = int(np.sum(~base_correct & (prediction == truth)))
                row["formerly_correct_now_wrong"] = int(np.sum(base_correct & (prediction != truth)))
            runs.append(row)
            print(f"{variant} seed={seed}: accuracy={row['accuracy']:.5f}, "
                  f"F1={row['f1']:.5f}, matrix={row['confusion_matrix']}", flush=True)
        summary = {key: {"mean": float(np.mean([r[key] for r in runs])),
                         "std_population": float(np.std([r[key] for r in runs]))}
                   for key in ("accuracy", "f1", "roc_auc")}
        results[variant] = {"input_bits": train.shape[1], "runs": runs, "summary": summary}
    return {"split": "validation", "seed_policy": "fixed documents and pairs, varied TM seed",
            "seeds": SEEDS, "tm_settings": TM_SETTINGS, "bittm_version": bittm.__version__,
            "vocabulary": {"size": len(words), "mode": "letters", "fit_split": "train"},
            "base_features": "[BOTH | ONE]", "extra_bits": [f"shared_words >= {n}" for n in COUNT_CUTOFFS],
            "rows": {name: len(y) for name, y in labels.items()}, "manifest_sha256": hashes,
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=Path("outputs/vanilla-v2"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/shared_count_study.json"))
    args = parser.parse_args()
    report = run(args.manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
