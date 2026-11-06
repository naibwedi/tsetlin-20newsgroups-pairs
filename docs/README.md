# Documentation index

| File | Read it to learn |
|---|---|
| **[REPORT.md](REPORT.md)** | **Unified academic project report (Aspects I–V): formal theory, methodology, ablations, overlap breakdown, and case studies** |
| [01_how_the_vanilla_tm_works.md](01_how_the_vanilla_tm_works.md) | The theory in three levels (child, student, project), mathematical `s = 1.0` proof, and presentation sentences |
| [02_code_walkthrough.md](02_code_walkthrough.md) | What every file and function does, and the baseline bug that was fixed |
| [03_results_and_evidence.md](03_results_and_evidence.md) | Every measured result (Claims 1–7), overlap buckets, `s > 1.0` diagnostics, each claim tied to evidence and a figure |
| [04_worked_pairs.md](04_worked_pairs.md) | Two real validation predictions traced to exact firing and silenced clauses |
| [dashboard.html](dashboard.html) | Interactive page: open it in any browser. Runs the real trained model on real pairs, plus a live toy learning lab |
| `figures/` | All charts (PNG) used above, including `overlap_breakdown.png` and `s_diagnostic.png` |
| `results/` | Raw numbers: `ablation`, `sweep`, `sweep2`, `diagnostic_studies.json`, `shared_count_study.json`, `rules_summary.json` |

## How the pieces map to the report grading

| Aspect | Where to look |
|---|---|
| I. Textual presentation | [`REPORT.md`](REPORT.md) (unified formal report) and [`03_results_and_evidence.md`](03_results_and_evidence.md) (claim → evidence structure) |
| II. Understanding | [`REPORT.md` §1–2](REPORT.md) and [`01_how_the_vanilla_tm_works.md` §1.5–1.7](01_how_the_vanilla_tm_works.md) (mathematical proof of `s = 1.0` elimination and 20-category wipeout of positive literals) |
| III. Solution | [`REPORT.md` §3](REPORT.md), [`02_code_walkthrough.md`](02_code_walkthrough.md), and the ablation in [`03_results_and_evidence.md` §3.3](03_results_and_evidence.md) |
| IV. Discussion | [`REPORT.md` §4–6](REPORT.md), [`03_results_and_evidence.md` §3.2–3.14](03_results_and_evidence.md), and [`04_worked_pairs.md`](04_worked_pairs.md) |
| V. Overall impression | Frozen configuration, verified reproducibility, overlap-bucket & `s`-curve diagnostics, and Hierarchical TM roadmap |

## Regenerate everything

```powershell
# Dashboard: python serve_dashboard.py  (http://127.0.0.1:8060)
# Rebuild it after changes: python export_dashboard_data.py ; python serve_dashboard.py --rebuild
$env:SCIKIT_LEARN_DATA = (Join-Path (Get-Location) "data")
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe run.py --output outputs/vanilla-v2
.\.venv\Scripts\python.exe sweep.py ablation
.\.venv\Scripts\python.exe sweep.py sweep
.\.venv\Scripts\python.exe sweep.py sweep2
.\.venv\Scripts\python.exe visualize_rules.py outputs/vanilla-v2/validation_rules_all.txt docs/figures
.\.venv\Scripts\python.exe make_docs_figures.py outputs/vanilla-v2/validation_metrics.json
.\.venv\Scripts\python.exe diagnostic_studies.py
```

The sweeps take about 10 minutes in total. The test split is not touched by any of these commands.

[Unified Report](REPORT.md) · [Two real validation predictions, with signed votes](04_worked_pairs.md) · [Frozen settings and manifest hashes](results/submission_freeze.json) · [Shared-count experiment](results/shared_count_study.json) · [Diagnostic studies (`s > 1.0`, overlap buckets)](results/diagnostic_studies.json)
