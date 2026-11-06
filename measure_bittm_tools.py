"""Re-measure ranking quality (AUC) and rule shortening on the frozen vanilla configuration.

Needs the bittm branch feature/readable-rules-and-features (it adds simplify()). Until it is released, put a copy of its
Python files, plus the compiled _core extension, first on PYTHONPATH, then:

  python measure_bittm_tools.py

Validation only. Writes docs/results/bittm_tools.json.
"""
import json
from pathlib import Path
import numpy as np
from pipeline import prepare, pair_features, make_vectorizer, auc
from run import run_experiment, DEFAULTS

OUT = Path('docs/results/bittm_tools.json')


def margin(packed, x):
    """Score for "same" minus score for "different" (higher = more likely same), computed from the packed model."""
    inc = np.asarray(packed.include_).astype(bool)
    w = np.asarray(packed.weights_).astype(np.int64)
    lit = np.concatenate([x, 1 - x], axis=1).astype(bool)
    fired = ~((inc[None] & ~lit[:, None, :]).any(axis=2)) & inc.any(axis=1)[None]
    s = fired.astype(np.int64) @ w
    return (s[:, 1] - s[:, 0]).astype(float)


def main():
    parts, pairs = prepare('outputs/dashboard-data')
    cfg = dict(DEFAULTS)
    count = make_vectorizer('count', cfg['vocabulary'], cfg['vocab_mode']).fit([d['text'] for d in parts['train']])
    xtr = pair_features(pairs['train'], parts['train'], count, cfg['features']).astype(np.uint8)
    xva = pair_features(pairs['validation'], parts['validation'], count, cfg['features']).astype(np.uint8)
    yva = np.array([p['label'] for p in pairs['validation']])

    res = {'seeds': {}, 'config': cfg}
    models = {}
    for seed in (42, 43, 44):
        r, model = run_experiment({**cfg, 'seed': seed}, parts, pairs, out=None)
        packed = model.to_packed()
        models[seed] = (model, packed)
        res['seeds'][seed] = {'accuracy': r['tm']['accuracy'], 'auc': auc(yva, margin(packed, xva)),
                              'cosine_acc': r['cosine']['accuracy'], 'cosine_auc': r['cosine_auc']}
        print(seed, res['seeds'][seed])
    for k in ('accuracy', 'auc', 'cosine_acc', 'cosine_auc'):
        v = [res['seeds'][s][k] for s in res['seeds']]
        res[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}

    # Rule shortening on the frozen model (seed 42). X for simplify is the training data.
    model, packed = models[42]
    base_len = np.asarray(packed.include_).sum(axis=1)
    rows = [{'tolerance': None, 'median_literals': float(np.median(base_len)), 'mean_literals': float(base_len.mean()),
             'accuracy': res['seeds'][42]['accuracy'], 'auc': res['seeds'][42]['auc'], 'same_answers_on_val': 1.0}]
    base_pred = margin(packed, xva) > 0
    for tol in (0.0, 0.01, 0.03, 0.1):
        small = packed.simplify(xtr, tolerance=tol)
        ln = np.asarray(small.include_).sum(axis=1)
        m = margin(small, xva)
        rows.append({'tolerance': tol, 'median_literals': float(np.median(ln)), 'mean_literals': float(ln.mean()),
                     'accuracy': float(((m > 0) == yva.astype(bool)).mean()), 'auc': auc(yva, m),
                     'same_answers_on_val': float(((m > 0) == base_pred).mean())})
        print(rows[-1])
    res['simplify'] = rows
    OUT.write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
