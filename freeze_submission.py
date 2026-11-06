"""Record the November vanilla configuration and validate its cosine comparison.

Run from the project root after generating outputs/vanilla-v2 with run.py.
Test manifests are hashed as bytes. Only validation pairs are scored.
"""
import hashlib
import json
from pathlib import Path

from pipeline import auc, best_cosine_threshold, cosine_scores, make_vectorizer, metrics


SOURCE = Path("outputs/vanilla-v2")
OUT = Path("docs/results/submission_freeze.json")
BASE_SHA = "29deafe26bb3fb0e431f476f757bb8b26ba24ea9"
SIMPLIFY_SHA = "d660d4933714882f2c5c827b3d38ef69058819e3"


def read(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = read("validation_metrics.json")
    settings = dict(features="sym", vocab_mode="letters", vocabulary=1000, clauses=100,
                    threshold=25, epochs=20, boost=1, s=1.0, seed=42)
    if run["split"] != "validation" or any(run["config"][k] != v for k, v in settings.items()):
        raise ValueError("Expected the frozen seed-42 validation run")
    texts = [item["text"] for item in read("train_documents.json")]
    docs, pairs = read("validation_documents.json"), read("validation_pairs.json")
    labels = [row["label"] for row in pairs]
    words = read("vocabulary.json")
    controls = {}
    for name in ("original", "matched_to_tm"):
        vec = make_vectorizer("tfidf", 1000, "letters")
        if name == "matched_to_tm":
            vec.set_params(vocabulary=words)
        vec.fit(texts)
        scores = cosine_scores(pairs, docs, vec)
        cutoff = best_cosine_threshold(scores, labels)
        controls[name] = {
            "words_shared_with_tm": len(set(words).intersection(vec.vocabulary_)),
            "validation_cutoff": cutoff,
            "validation_auc": auc(labels, scores),
            **metrics(labels, scores >= cutoff),
        }
    if abs(controls["original"]["validation_cutoff"] - run["cosine_threshold"]) > 1e-12:
        raise ValueError("Original cosine cutoff differs from saved validation result")
    filenames = ["categories.json", "vocabulary.json"]
    for partition in ("train", "validation", "test"):
        filenames.extend([f"{partition}_documents.json", f"{partition}_pairs.json"])
    record = {
        "selected_model": "unsimplified vanilla TM",
        "model": settings, "weighted_clauses": False,
        "library": {"training_commit": BASE_SHA, "optional_simplify_commit": SIMPLIFY_SHA},
        "data_seed": 42, "comparison_model_seeds": [42, 43, 44],
        "train_rows": 20000, "validation_rows": len(pairs), "test_rows_planned": 4000,
        "primary_metric": "accuracy", "cosine_for_final_evaluation": "original",
        "cosine_comparison": controls,
        "manifest_sha256": {name: digest(SOURCE / name) for name in filenames},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
