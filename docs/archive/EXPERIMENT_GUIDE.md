> **Update:** this guide describes the *first* run. It contains a baseline error (the TF-IDF cosine cut-off was chosen by F1, which reported 0.500; the correct value is about 0.605). The current method, corrected numbers, figures and explanations are in [`docs/`](docs/README.md). Start with [`docs/03_results_and_evidence.md`](docs/03_results_and_evidence.md).

# Vanilla TM experiment: first measured validation results

## What the model is asked to learn

Each example is a pair of 20 Newsgroups posts. Its target is 1 if their original category IDs match and 0 otherwise. The model receives only word-presence bits from the two posts. It never receives their category IDs. We split documents before pairing them to prevent the same post appearing in training and evaluation.

The required `s=1.0` is used in `bittm.TsetlinMachine`, with 100 clauses **per class** (200 total), `T=50`, 20 epochs, 1,000 vocabulary words, unweighted votes, and `boost_true_positive=False`. The 20,000 training rows contain both orientations of 10,000 unique unordered pairs. Validation has 4,000 held-out pairs. The test split was prepared but has **not** been scored or used to choose settings.

## How the code works

`pipeline.py` loads the standard dataset partitions, removes headers, footers and quotes, holds out 20% of training documents for validation, removes exact duplicates across partitions, and samples balanced positive/negative pairs separately. `CountVectorizer(binary=True)` learns the vocabulary from training documents only. `pair_features` concatenates two vectors: `[A.word_1 ... A.word_n | B.word_1 ... B.word_n]`. A 1,000-word vocabulary yields 2,000 Boolean features.

`run.py` trains BitTM's vanilla class. Each class owns 100 clauses; the first half vote for that class and the second half vote against it. Each clause is an AND of included literals, where a literal can mean that a word is present or absent in A or B. BitTM combines fired clause votes to choose class 0 or 1. The script also measures majority prediction, TF-IDF cosine similarity, a confusion matrix, runtime, and consistency when A and B are swapped. The cosine threshold is selected on validation accuracy, so that score is optimistic and should not be treated as an untouched evaluation.

## First full validation result

| Method | Accuracy | Precision for same category | Recall for same category | F1 |
|---|---:|---:|---:|---:|
| Vanilla BitTM, `s=1.0`, boost off | 0.501 | 0.504 | 0.135 | 0.213 |
| Majority class | 0.500 | 0 | 0 | 0 |
| TF-IDF cosine, threshold selected on validation F1 (old output) | 0.500 | 0.500 | 1.000 | 0.667 |

The TM predicted “different category” for 3,464 of 4,000 validation pairs. It correctly found 270 of 2,000 positive pairs, and mislabeled 266 negative pairs as positive. This is a **weak baseline**, close to chance accuracy. It does not support a claim of useful document-pair learning yet. The original cosine threshold selected as 0 because F1 favored predicting every pair as positive. Recomputing the same saved validation scores with an accuracy-selected threshold gave 0.60425 accuracy at threshold 0.04788, with ROC AUC 0.60957. This shows useful, though modest, discrimination. The old JSON and figure retain the earlier F1-selected baseline and must not be cited as the corrected comparison. Pair-order consistency was 0.7725, so about 23% of predictions changed when pair order was reversed.

![Measured validation accuracy](outputs/validation/accuracy.png)

(`outputs/` is git-ignored; the maintained figures are in `docs/figures/`.)

![Measured TM confusion matrix](outputs/validation/confusion_matrix.png)

These charts and values come from `outputs/validation/validation_metrics.json`. The dataset's 20 category labels are coarse; same-category pairs can use different words, while different-category pairs can share words. This makes the task harder than predicting one document's topic. Before changing the model, check the actual learned clauses, positive/negative pair sampling, and whether the vocabulary retains enough topical words. Future parameter experiments must use the same validation pairs and leave test untouched.

## Reproducing

Run `pytest tests -q` first. Set `SCIKIT_LEARN_DATA` to a writable project data folder. Run `python run.py --output outputs/validation`, then `python figures.py outputs/validation/validation_metrics.json`. The output path holds manifests, vocabulary, metrics and figures. Use the exact dependency commit in `requirements.txt`. Training results are reproducible for a fixed BitTM seed and environment, but document any system or version differences when comparing runs.

## One controlled follow-up: boost true positives

With the same documents, pairs, vocabulary, `s=1.0`, clauses, `T`, and epochs, setting `boost_true_positive=True` produced validation accuracy **0.51275**, recall **0.4445**, and F1 **0.4771**. The original setting produced accuracy 0.501, recall 0.135, and F1 0.2129. Boosting improved this run, but performance is still near chance accuracy and pair-order consistency fell from 0.7725 to 0.74325. This is one seed and one validation comparison; it is not a winning result or a final test result.

![Boosted run validation accuracy](outputs/validation_boost/accuracy.png)

![Boosted run confusion matrix](outputs/validation_boost/confusion_matrix.png)

The boosted run's first 30 readable clauses are in `outputs/validation_boost/validation_rules.txt`. These are real learned rules, selected in BitTM's default rule order. Check whether they contain topical words, negations, and both document positions before claiming interpretability.

## Correction to cosine baseline

Claude identified a real evaluation mistake. The old `validation_metrics.json` files chose the cosine threshold by F1, causing an all-positive predictor. The current `run.py` chooses it by validation accuracy and records AUC. Rerun the validation command to regenerate corrected JSON and charts before citing them. The threshold is still tuned on validation data, so its validation accuracy is optimistic.

## Corrected measured comparison (boost on)

The rerun in `outputs/validation_corrected` uses the same 4,000 validation pairs and chooses the cosine cutoff by **validation accuracy**. BitTM accuracy is **0.51275** (F1 **0.47706**). TF-IDF cosine accuracy is **0.60475** (F1 **0.52047**), threshold **0.04000**, and ROC AUC **0.60954**. Thus the simple similarity baseline beats this TM configuration by about **9.2 percentage points** on validation. This comparison does not establish test performance. Use the corrected folder's charts for presentations; the older charts show the flawed F1-selected cosine baseline.

![Corrected accuracy comparison](outputs/validation_corrected/accuracy.png)

![Corrected TM confusion matrix](outputs/validation_corrected/confusion_matrix.png)
