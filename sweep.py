"""Validation-only experiment groups, each config repeated over 3 TM seeds.

  python sweep.py ablation   # which change helped? (features / vocabulary / boost)
  python sweep.py sweep      # one-factor-at-a-time settings sweep around the best base
  python sweep.py sweep2     # follow-up: smaller T and bigger models at T=25

Documents and pairs are fixed (data seed 42); only the TM's random start changes
with the seed. The test split is never scored here.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from pipeline import prepare
from run import run_experiment

SEEDS = (42, 43, 44)
BASE = dict(features='sym', vocab_mode='letters', boost=1, clauses=100, threshold=50, vocabulary=1000)

GROUPS = {
    'ablation': [
        ('A0 original (A|B, default vocab, boost off)', dict(features='ab', vocab_mode='default', boost=0)),
        ('A1 + boost on', dict(features='ab', vocab_mode='default', boost=1)),
        ('A2 + letters-only vocab', dict(features='ab', vocab_mode='letters', boost=1)),
        ('A3 + order-independent features (new default)', dict(features='sym', vocab_mode='letters', boost=1)),
        ('A4 new default but boost off', dict(features='sym', vocab_mode='letters', boost=0)),
        ('A5 order-independent, default vocab', dict(features='sym', vocab_mode='default', boost=1)),
    ],
    'sweep': [
        ('base: 100 clauses, T=50, vocab 1000', {}),
        ('200 clauses, T=100', dict(clauses=200, threshold=100)),
        ('500 clauses, T=250', dict(clauses=500, threshold=250)),
        ('100 clauses, T=25', dict(threshold=25)),
        ('100 clauses, T=100', dict(threshold=100)),
        ('vocab 2000', dict(vocabulary=2000)),
        ('vocab 3000', dict(vocabulary=3000)),
    ],
    # Follow-up: the first sweep showed smaller T is better, so look further down.
    'sweep2': [
        ('T=5', dict(threshold=5)),
        ('T=10', dict(threshold=10)),
        ('T=15', dict(threshold=15)),
        ('T=25 (from sweep 1)', dict(threshold=25)),
        ('200 clauses, T=25', dict(clauses=200, threshold=25)),
        ('500 clauses, T=25', dict(clauses=500, threshold=25)),
    ],
}


def main(group):
    out = Path('docs/results'); out.mkdir(parents=True, exist_ok=True)
    parts, pairs = prepare('outputs/sweep-data')
    rows = []
    for label, change in GROUPS[group]:
        accs, f1s, recs, cos = [], [], [], None
        for seed in SEEDS:
            cfg = {**BASE, **change, 'seed': seed}
            r, _ = run_experiment(cfg, parts, pairs, out=None)
            accs.append(r['tm']['accuracy']); f1s.append(r['tm']['f1']); recs.append(r['tm']['recall'])
            cos = (r['cosine']['accuracy'], r['cosine_auc'])
            print(f"{label} seed={seed} acc={accs[-1]:.4f} rec={recs[-1]:.3f} "
                  f"train={r['train_seconds']:.0f}s order={r['order_consistency']:.3f}", flush=True)
        rows.append({'name': label, **{k: v for k, v in {**BASE, **change}.items()},
                     'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                     'recall_mean': float(np.mean(recs)), 'f1_mean': float(np.mean(f1s)),
                     'accs': accs, 'cosine_acc': cos[0], 'cosine_auc': cos[1]})
        (out / f'{group}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with open(out / f'{group}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'accs'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print('done', group)


if __name__ == '__main__':
    main(sys.argv[1])
