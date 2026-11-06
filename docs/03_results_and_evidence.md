# 3. Results and evidence (validation only)

All numbers below were **measured** by the code in this folder on the **validation** pairs (4,000 pairs, 2,000 same / 2,000 different). The **test split has not been scored**. Raw numbers: [`results/ablation.csv`](results/ablation.csv), [`results/sweep.csv`](results/sweep.csv), [`results/sweep2.json`](results/sweep2.json), [`results/diagnostic_studies.json`](results/diagnostic_studies.json), full per-run logs in `results/*_log.txt`. Every configuration uses 3 TM seeds (42, 43, 44); documents and pairs are fixed.

## 3.1 Headline

![headline](figures/headline.png)

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.500 | 0 | 0 | 0 |
| TF-IDF cosine (cut-off chosen on validation accuracy) | 0.605 | 0.657 | 0.438 | 0.525 |
| **Vanilla TM, frozen config, seed 42** | **0.575** | 0.579 | 0.550 | 0.564 |
| Vanilla TM, same config, mean of 3 seeds | 0.571 ± 0.003 | | | |
| First run (original code), seed 42 | 0.501 | 0.504 | 0.135 | 0.213 |

**Plain reading:** the vanilla TM moved from chance (0.50) to about 0.57. A simple word-overlap baseline is still better (0.605). The honest conclusion is *"the vanilla TM learns something, but does not beat the TF-IDF baseline on this task yet."*

The cosine cut-off was tuned on the same validation pairs, so 0.605 is slightly optimistic for the baseline. The TM's settings were also chosen on validation, so 0.571 is slightly optimistic for the TM. Both will be re-measured once on the test split.

**Frozen vanilla configuration** (used for the first submission):

| Setting | Value |
|---|---|
| `s` | 1.0 (required) |
| Clauses per class | 100 (200 total) |
| `T` | 25 |
| Epochs | 20 |
| Boost true positive | on |
| Weighted | no |
| Features | `sym`: [in both, in exactly one] |
| Vocabulary | 1,000 most frequent letters-only words (stop words removed), fitted on training documents |
| Training data | 10,000 unique pairs, both orientations (20,000 rows) |

Reproduce: `python run.py --output outputs/vanilla-v2` (these are the defaults).

## 3.2 Claim 1: the baseline was misreported, and is now fixed

**Claim:** the original cosine baseline reported 0.500 because the cut-off was chosen by F1, which in that run favoured "everything matches" on balanced data.
**Evidence:** with the cut-off chosen by accuracy the same scores give 0.605 accuracy and AUC 0.61 (`ablation.csv`, columns `cosine_acc`, `cosine_auc`). The "all-positive" prediction has F1 = 0.667, higher than the best real cut-off's F1 (0.52), which is why F1 selection picked it.

## 3.3 Claim 2: what helped (ablation)

![ablation](figures/ablation.png)

| Step | Accuracy (mean ± std, 3 seeds) | What changed |
|---|---:|---|
| A0 original | 0.500 ± 0.000 | `[A|B]`, default vocabulary, boost off. The TM predicts "different" for nearly everything (recall 0.05) |
| A1 + boost on | 0.511 ± 0.009 | Recall rises from 0.05 to 0.45 |
| A2 + letters-only vocabulary | 0.517 ± 0.002 | Small gain |
| **A3 + order-independent features** | **0.556 ± 0.004** | The largest gain, about +4 points |
| A4 A3 with boost off | 0.505 ± 0.002 | Boost is needed at `s = 1.0` |
| A5 A3 with default vocabulary | 0.552 ± 0.003 | Vocabulary cleaning adds about +0.4 points only |

**Reading:** the **representation** mattered most. Making "shared word" a single literal (`BOTH.word`) did more than any parameter. Cleaning the vocabulary helped only slightly, and we report it as a small effect. Order consistency went from 0.74 to **1.00** (swapping A and B cannot change `sym` features).

## 3.4 Claim 3: a smaller T works better

![t curve](figures/t_curve.png)

| T | 5 | 10 | 15 | **25** | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.549 | 0.558 | 0.556 | **0.571** | 0.556 | 0.548 |
| Std over seeds | 0.016 | 0.006 | 0.005 | 0.003 | 0.004 | 0.002 |

**Reading:** T = 25 is best, but the curve is shallow: the whole range moves accuracy by about 2 points, similar to a few standard deviations. We freeze T = 25, but would not claim T is a strong lever.

## 3.5 Claim 4: bigger models and vocabularies did not help

![sweep](figures/sweep.png)

| Variation (others at default, T from sweep) | Accuracy |
|---|---:|
| 100 clauses, T = 50 | 0.556 |
| 200 clauses, T = 100 | 0.553 |
| 500 clauses, T = 250 | 0.559 |
| 200 clauses, T = 25 | 0.559 ± 0.009 |
| 500 clauses, T = 25 | 0.556 ± 0.010 |
| Vocabulary 2,000 | 0.554 |
| Vocabulary 3,000 | 0.548 |

At fixed T=25, neither 200 nor 500 clauses beat 100 on mean validation accuracy. In the first three rows, T and clause count change together, so those rows cannot isolate either effect.

## 3.6 Claim 5: the learned rules are eliminations, not topics

![not vs present](figures/rules_not_vs_present.png)
![clause length](figures/rules_length.png)

**Evidence** ([`results/rules_summary.json`](results/rules_summary.json)):

The 200 clauses contain 174,901 negated literals, including 171,351 conditions on BOTH and 3,550 on ONE. Median length is 924. Grouping by the direction a firing clause moves the same-minus-different margin gives median lengths of 847 for same and 945.5 for different. An absence can therefore support either prediction. The two [worked examples](04_worked_pairs.md) trace actual signed votes.

## 3.7 Feature illustration

![features](figures/feature_example.png)

## 3.8 What the TM gets wrong

![confusion](figures/confusion_matrix.png)

On 4,000 validation pairs: 1,200 correct "different", 1,099 correct "same", 800 false "same", 901 missed "same". Errors are balanced: the TM is neither all-yes nor all-no.

## 3.9 Limitations (say these in the report)

The three seeds vary training on one fixed validation sample. On `N = 4,000` validation pairs, the binomial standard error around `p = 0.57` is `sqrt(0.57 * 0.43 / 4000) ≈ 0.0078` (±0.78% for 1 SD, or ±1.5% for a 95% confidence interval), which is larger than the 3-seed standard deviation (±0.0027). Choosing model settings and the cosine cutoff on this sample can favour both methods. Near duplicates remain a limitation. The two fitted vocabularies share 869 of 1,000 words because `CountVectorizer(binary=True)` ranks words by document frequency whereas `TfidfVectorizer` ranks by total corpus term frequency. Forcing cosine to use all 1,000 TM words gives accuracy 0.6030 and AUC 0.6102.

## 3.10 Next evaluation

Use the [freeze record](results/submission_freeze.json) and the lecturer’s evaluation protocol for final scoring. The diagnostic experiments below used validation only.

## 3.11 Ranking quality (AUC) and shorter rules (bittm branch tools)

Measured by [`measure_bittm_tools.py`](../measure_bittm_tools.py) on the validation pairs, frozen configuration. It uses the bittm branch `feature/readable-rules-and-features` (adds `simplify`), not the released 0.1.1. Raw numbers: [`results/bittm_tools.json`](results/bittm_tools.json).

| Method | Accuracy | AUC |
|---|---:|---:|
| Vanilla TM, 3 seeds | 0.571 ± 0.003 | 0.600 ± 0.001 |
| TF-IDF cosine | 0.605 | 0.611 |

**Reading:** ranking the pairs by the TM's vote margin gives AUC 0.600, still below cosine (0.611). On our configuration the TM does not rank better than cosine. (A separate run reported cosine AUC 0.573; we could not reproduce that, ours is 0.611 with the separately fitted TF-IDF vocabulary.)

![simplify](figures/simplify.png)

| `simplify` tolerance (fitted on training pairs) | Median conditions per rule | Validation accuracy | AUC | Same answers as original on validation |
|---|---:|---:|---:|---:|
| original | 924 | 0.575 | 0.601 | 100% |
| 0 | 440.5 | 0.571 | 0.605 | 94.4% |
| 0.01 | 355 | 0.574 | 0.605 | 94.0% |
| 0.03 | 225 | 0.574 | 0.605 | 92.6% |
| 0.1 | 43 | 0.550 | 0.575 | 82.0% |

At tolerance 0.03 the median falls from 924 to 225 literals. Validation accuracy moves from 57.475% to 57.375%, with 298 of 4,000 decisions changing. Even tolerance zero changes 225 validation decisions. Exact agreement on the fitting rows does not establish equivalence on unseen pairs. Keep the full model for submission.

## 3.12 Shared-word counts (and why `s = 1.0` inverts rare summary bits)

Three extra bits marked at least 1, 2 or 5 distinct shared vocabulary words. On seeds 42, 43 and 44, the original symmetric representation scored 57.475%, 56.975% and 56.850%. Adding the bits gave 56.350%, 54.900% and 55.550%. Mean AUC also fell from 0.5996 to 0.5869. Keep the original 2,000 bits. [Experiment record](results/shared_count_study.json).

**Why did adding shared-count bits hurt accuracy?** Because at `s = 1.0` (`1 / s = 1.0` penalty on `0` bits), any bit that is `0` on the majority of training rows has its *negation* (`NOT bit`) locked into clauses of both classes. Since 70% of pairs share 0 or 1 word (`shared_words >= 2` is `0`), same-category clauses include `NOT (shared_words >= 2)` and `NOT (shared_words >= 5)`. When a validation pair shares 2 or more words—strong positive evidence of a match—those negated conditions become false and perversely **silence** the same-category clauses!

## 3.13 Claim 6: Where TM vs. TF-IDF Cosine win and lose across shared-word overlap buckets

![overlap breakdown](figures/overlap_breakdown.png)

Breaking down the 4,000 validation pairs by the number of shared TM vocabulary words ([`diagnostic_studies.json`](results/diagnostic_studies.json)) explains the exact precision–recall trade-off between the two models:

| Shared TM words | Pairs (% of val) | True "same" rate | TM Accuracy | Cosine Accuracy | TM Recall | Cosine Recall |
|---|---:|---:|---:|---:|---:|---:|
| **0 shared words** | 1,856 (46.4%) | 43.7% | 0.554 | 0.564 | **0.317** | 0.009 |
| **1 shared word** | 939 (23.5%) | 50.5% | 0.588 | 0.630 | **0.671** | 0.544 |
| **2 shared words** | 484 (12.1%) | 57.6% | 0.612 | 0.638 | **0.806** | 0.753 |
| **3 shared words** | 271 (6.8%) | 58.7% | 0.616 | 0.627 | 0.818 | **0.862** |
| **4+ shared words** | 450 (11.3%) | 61.6% | 0.567 | **0.669** | 0.610 | **0.949** |

**Reading:**
1. **Almost half of all validation pairs (46.4%) share zero words** in the 1,000-word vocabulary, even though 43.7% of those zero-overlap pairs truly belong to the same category! TF-IDF cosine is virtually blind here (recall 0.9%, predicting "different" on 1,843 of 1,856 pairs), whereas the TM recovers 31.7% of true zero-overlap matches (257 pairs) and 67.1% of 1-word matches using `NOT ONE` and `NOT BOTH` elimination checklists.
2. **On high-overlap pairs (`4+` shared words), TF-IDF cosine dominates (66.9% vs. 56.7% accuracy, 94.9% vs. 61.0% recall).** Why does the TM's recall drop from 81.8% (at 3 shared words) down to 61.0% (at 4+ shared words)? Because every clause contains ~924 `NOT BOTH.word` conditions! When a pair shares 4 or more vocabulary words, those shared words trigger false conditions (`NOT BOTH.word = 0`) that silence many "same-category" clauses.

## 3.14 Claim 7: Diagnostic checks on specificity `s > 1.0`, literal budget, and training schedule

![s diagnostic](figures/s_diagnostic.png)

To verify whether alternative vanilla TM settings could overcome the `s = 1.0` elimination ceiling, [`diagnostic_studies.py`](../diagnostic_studies.py) tested three hypotheses across seeds 42, 43, and 44:

| Study / Configuration | Mean Accuracy (± std) | Mean F1 | Mean AUC | Positive Literals (seed 42) | Negated Literals (seed 42) | Median Clause Length |
|---|---:|---:|---:|---:|---:|---:|
| **`s = 1.0` (frozen baseline, 10k × 2 rows)** | **0.571 ± 0.003** | **0.560** | **0.600** | **0** | **174,901** | **924.0** |
| `s = 2.0` | 0.541 ± 0.005 | 0.551 | 0.554 | 0 | 56,634 | 274.5 |
| `s = 3.0` | 0.525 ± 0.007 | 0.449 | 0.534 | 0 | 40,636 | 202.5 |
| `s = 5.0` | 0.516 ± 0.003 | 0.427 | 0.522 | 0 (seed 43: 3) | 36,109 | 179.5 |
| `s = 10.0` | 0.523 ± 0.003 | 0.381 | 0.528 | 134 | 17,120 | 9.5 |
| `s = 1.0, literal_budget = 10` | 0.572 ± 0.007 | 0.406 | 0.601 | 0 | 1,873 | 10.0 |
| `s = 1.0`, 20k unique unordered pairs (`augment=False`) | 0.560 ± 0.005 | 0.541 | 0.589 | — | — | — |
| `s = 1.0`, best-validation-epoch checkpointing | **0.580 ± 0.002** | **0.570** | **0.605** | 0 | ~174,000 | ~920.0 |
| **`s = 1.0`, stopwords removed (`clauses=200, T=50, weighted=True`)** | **0.583 ± 0.007 (58.8% seed 42)** | **0.504** | **0.600 (0.607 seed 42)** | **0** | **242,280** | **~605.0** |

**Key findings from the diagnostic study:**
1. **`s = 1.0` is genuinely the best specificity for the flat Vanilla TM:** Increasing `s` to `2.0, 3.0, 5.0, 10.0` shortens clauses (from 924 down to 9.5 literals) and allows positive literals (`BOTH.w`, `ONE.w`) to emerge at `s = 10.0` (134 positive literals), **but validation accuracy drops by 3 to 5.5 percentage points**. A flat 200-clause pool cannot cover 20 categories with sparse positive conjunctions.
2. **Capping clause length during training (`literal_budget = 10` at `s = 1.0`):** Restricting each clause to at most ~10 literals achieves the same mean accuracy (`0.572 ± 0.007`, AUC `0.601`) with **90× shorter rules** (median 10 literals vs. 924), though class recall/precision balance fluctuates more across seeds (F1 0.406).
3. **Why repeating each pair twice (`10k × 2` vs. `20k × 1`) helped at `s = 1.0`:** Because `features='sym'` is order-independent, presenting both `(A, B)` and `(B, A)` consecutively in `train_pairs.json` performs two consecutive feedback steps per unique pair per epoch (40 effective passes over 10,000 pairs), which stabilizes automaton states and beats 20 passes over 20,000 single-exposure pairs (`0.571` vs. `0.560`).
4. **Best-epoch checkpointing (`58.0%`) and removing stopwords (`58.3% ± 0.7%`, `58.80%` on seed 42) at `s = 1.0`:** Selecting the best validation epoch (`best_model_`) raises the frozen setup to **0.5795 ± 0.0018** (58.0%), and removing English stopwords with `clauses=200, T=50, weighted=True` reaches **0.5829 ± 0.0068 (58.80% on seed 42, AUC 0.6067)** while still learning **0 positive literals** (`242,280` negated literals), confirming that ~58.8% is the structural ceiling of flat `s = 1.0` reasoning by elimination on two word sets.

## 3.15 Claim 8 (Stage 2 — November 26): Two-Layer Hierarchical TM Beats Both Vanilla TM and TF-IDF Cosine

![hierarchical comparison](figures/hierarchical_comparison.png)

Measured by [`run_hierarchical.py`](../run_hierarchical.py) (`hierarchical.py`) on the exact 4,000 validation pairs across seeds 42, 43, and 44 ([`results/hierarchical_study.json`](results/hierarchical_study.json)):

| Architecture | Mean Accuracy (± std) | Seed 42 / Best | Mean Precision | Mean Recall | Mean F1 | Mean ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 Vanilla TM (`s = 1.0`, frozen) | 0.571 ± 0.003 | 0.575 / 0.575 | 0.575 | 0.547 | 0.560 | 0.600 ± 0.001 |
| TF-IDF Cosine Baseline | 0.605 | 0.605 / 0.605 | 0.657 | 0.438 | 0.525 | 0.611 |
| **Stage 2a: Hierarchical TM (`pair_guided_or`, ZERO Category IDs)** | **0.745 ± 0.004** | **0.749 / 0.749** | **0.734** | **0.769** | **0.751** | **0.811 ± 0.005** |
| **Stage 2b: Hierarchical TM (`supervised_l1`, 12k Vocab)** | **0.857 ± 0.003** | **0.860 / 0.860** | **0.858** | **0.855** | **0.856** | **0.930 ± 0.002** |
| **Stage 2c: Hierarchical TM (`hybrid` Multi-Scale, 50k Vocab)** | **0.876 ± 0.002** | **0.877 / 0.878** | **0.890** | **0.859** | **0.874** | **0.943 ± 0.003** |

**Reading:**
1. **Track A (`pair_guided_or`) reaches 74.50% accuracy (0.8111 AUC) using ZERO category IDs:** By removing stopwords, clustering the cross-document word co-occurrence lift between $Y=1$ and $Y=0$ training pairs (`discover_pair_guided_or_concepts`), and learning a symmetric bilinear pair-affinity matrix $M_{\text{diff}} = M_{\text{pos}} - M_{\text{neg}}$ in 120-dimensional SVD space strictly from pair labels, Level 2 reaches **74.50% ± 0.38% accuracy (74.88% on seed 42)** and **0.8111 ± 0.0050 AUC** without ever touching a single category label.
2. **Track B (`supervised_l1`, 12k vocab) reaches 85.66% accuracy (0.9303 AUC) and Track C (`hybrid`, 50k vocab) reaches 87.63% accuracy (87.75% on seed 43, 0.9434 AUC):** Combining 5-fold Out-Of-Fold multi-scale topic log-odds & margin profiles (`ComplementNB(0.1)`, `ComplementNB(0.5)`, `RidgeClassifier(1.0)`), explicit 20-topic and 6-supergroup Level-1 TMs, and `QuantileBooleanizer(n_bins=16)` multi-temperature soft topic agreement enables the Level-2 Pair TM (`T = 50, s = 3.0`) to reach **87.63% ± 0.16% accuracy** and **0.9434 ± 0.0028 AUC**.
3. **Surpassing TF-IDF Cosine across all overlap regimes:** On the 1,856 validation pairs with **zero shared 1k-vocabulary words**, `hybrid` reaches **84.3% accuracy (79.3% recall vs. Cosine's 0.9%)**, on 1 shared word **90.5% accuracy**, on 2 shared words **90.3% accuracy**, on 3 shared words **90.8% accuracy**, and on 4+ shared words **91.6% accuracy** (vs. Cosine's 66.9% and Vanilla TM's 56.7%)!
