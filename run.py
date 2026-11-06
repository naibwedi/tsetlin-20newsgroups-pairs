"""Run validation experiments and, after selection, a separate final test.

Defaults are the best settings found on validation (T=25, order-independent features,
letters-only vocabulary, boost on). Use --features ab --vocab-mode default
--boost 0 --threshold 50 to reproduce the very first run.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pipeline import (prepare, pair_features, feature_names, make_vectorizer,
                      cosine_scores, best_cosine_threshold, auc, metrics, reverse_pair)

DEFAULTS = dict(vocabulary=1000, clauses=100, threshold=25, epochs=20, boost=1, s=1.0,
                features='sym', vocab_mode='letters', seed=42)


def run_experiment(cfg, parts, pairs, out=None, final_test=False, save_rules=True):
    """Train one vanilla TM and evaluate it. `cfg` is a dict like DEFAULTS.

    `seed` only changes the TM's random start; documents and pairs stay fixed so
    runs are comparable.
    """
    from bittm import TsetlinMachine
    import bittm
    cfg = {**DEFAULTS, **cfg}
    count = make_vectorizer('count', cfg['vocabulary'], cfg['vocab_mode'])
    count.fit([d['text'] for d in parts['train']])
    words = count.get_feature_names_out().tolist()
    mode = cfg['features']

    x = pair_features(pairs['train'], parts['train'], count, mode)
    y = np.array([p['label'] for p in pairs['train']], dtype=np.int32)
    val_y = np.array([p['label'] for p in pairs['validation']])

    model = TsetlinMachine(clauses_per_class=cfg['clauses'], T=cfg['threshold'], s=cfg['s'],
                           boost_true_positive=bool(cfg['boost']), weighted=False,
                           random_state=cfg['seed'], epochs=cfg['epochs'])
    start = time.perf_counter()
    model.fit(x, y)
    train_seconds = time.perf_counter() - start

    # Cosine baseline: the cut-off is always chosen on VALIDATION pairs.
    tfidf = make_vectorizer('tfidf', cfg['vocabulary'], cfg['vocab_mode'])
    tfidf.fit([d['text'] for d in parts['train']])
    val_scores = cosine_scores(pairs['validation'], parts['validation'], tfidf)
    cosine_threshold = best_cosine_threshold(val_scores, val_y)

    name = 'test' if final_test else 'validation'
    ep, ed = pairs[name], parts[name]
    ex = pair_features(ep, ed, count, mode)
    ey = np.array([p['label'] for p in ep])
    start = time.perf_counter()
    pred = model.predict(ex)
    inference = time.perf_counter() - start
    reversed_x = pair_features([reverse_pair(p) for p in ep], ed, count, mode)
    scores = cosine_scores(ep, ed, tfidf)
    result = {'config': cfg, 'split': name, 'tm': metrics(ey, pred),
              'majority': metrics(ey, np.full(len(ey), np.bincount(y).argmax())),
              'cosine': metrics(ey, scores >= cosine_threshold),
              'cosine_auc': auc(ey, scores),
              'cosine_threshold': float(cosine_threshold),
              'train_seconds': train_seconds, 'inference_seconds': inference,
              'order_consistency': float(np.mean(pred == model.predict(reversed_x))),
              'clauses_per_class': cfg['clauses'], 'total_clauses': len(model.clauses_),
              'bittm_version': bittm.__version__}
    if out is not None:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / 'vocabulary.json').write_text(json.dumps(words, indent=2), encoding='utf-8')
        if save_rules:
            names = feature_names(words, mode)
            rules = model.rules(feature_names=names)
            (out / f'{name}_rules_all.txt').write_text('\n'.join(map(str, rules)), encoding='utf-8')
            (out / f'{name}_rules.txt').write_text(
                '\n'.join(str(r) for r in model.rules(feature_names=names, top=30)), encoding='utf-8')
        (out / f'{name}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--final-test', action='store_true', help='evaluate chosen settings on the held-out test set')
    p.add_argument('--vocabulary', type=int, default=DEFAULTS['vocabulary'])
    p.add_argument('--clauses', type=int, default=DEFAULTS['clauses'])
    p.add_argument('--threshold', type=int, default=DEFAULTS['threshold'])
    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--boost', type=int, choices=[0, 1], default=DEFAULTS['boost'])
    p.add_argument('--s', type=float, default=DEFAULTS['s'])
    p.add_argument('--features', choices=['ab', 'sym'], default=DEFAULTS['features'],
                   help="'ab' = [A|B]; 'sym' = [in both | in exactly one] (order independent)")
    p.add_argument('--vocab-mode', choices=['letters', 'default'], default=DEFAULTS['vocab_mode'])
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'], help='TM random seed only')
    args = p.parse_args()
    try:
        import bittm  # noqa: F401
    except ImportError as exc:
        raise SystemExit('BitTM is unavailable. Install the pinned repository dependency in requirements.txt.') from exc
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    sizes = (200, 40, 40) if args.smoke else (20000, 4000, 4000)
    parts, pairs = prepare(out, *sizes)
    cfg = {k: getattr(args, k.replace('-', '_')) for k in DEFAULTS}
    cfg['smoke'] = args.smoke
    result, _ = run_experiment(cfg, parts, pairs, out, args.final_test)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
