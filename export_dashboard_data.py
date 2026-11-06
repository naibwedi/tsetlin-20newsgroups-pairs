"""Export the trained vanilla TM, sample validation pairs and measured results for docs/dashboard.html.

  python export_dashboard_data.py

Retrains the frozen configuration (seed 42, about 10 s), checks that a NumPy re-implementation of the
voting matches model.predict on all 4,000 validation pairs, then writes docs/dashboard_data.json.
"""
import base64
import json
import re
from pathlib import Path
import numpy as np
from pipeline import prepare, pair_features, make_vectorizer
from run import run_experiment, DEFAULTS

OUT = Path('docs/dashboard_data.json')
EMAIL = re.compile(r'\S+@\S+')
URL = re.compile(r'https?://\S+')


def snippet(text, n=300):
    t = re.sub(r'\s+', ' ', text).strip()
    t = URL.sub('[link]', EMAIL.sub('[email]', t))
    return t if len(t) <= n else t[:n].rsplit(' ', 1)[0] + ' …'


def clause_scores(x, include, weights):
    """Reconstruct signed class votes, bounding memory by scoring 32 pairs at a time."""
    scores = np.empty((len(x), weights.shape[1]), dtype=np.int64)
    include = np.asarray(include, dtype=bool)
    nonempty = include.any(axis=1)
    weights = np.asarray(weights, dtype=np.int64)
    for start in range(0, len(x), 32):
        batch = x[start:start + 32].astype(bool)
        literals = np.concatenate((batch, ~batch), axis=1)
        blocked = (include[None] & ~literals[:, None]).any(axis=2)
        scores[start:start + len(batch)] = (~blocked & nonempty) @ weights
    return scores


def main():
    parts, pairs = prepare('outputs/dashboard-data')
    cfg = dict(DEFAULTS)
    result, model = run_experiment(cfg, parts, pairs, out=None)
    count = make_vectorizer('count', cfg['vocabulary'], cfg['vocab_mode']).fit([d['text'] for d in parts['train']])
    words = count.get_feature_names_out().tolist()

    packed_model = model.to_packed()
    include = np.asarray(packed_model.include_).astype(np.uint8)   # (clauses, 2 * features)
    weights = np.asarray(packed_model.weights_).astype(int)        # (clauses, classes)
    n_feat = include.shape[1] // 2

    # Re-implement voting exactly as bittm's explain() does and compare with predict().
    x = pair_features(pairs['validation'], parts['validation'], count, cfg['features']).astype(np.uint8)
    scores = clause_scores(x, include, weights)
    np.testing.assert_array_equal(scores, model.decision_function(x))
    mine = scores.argmax(axis=1)
    theirs = np.asarray(model.predict(x.astype(np.uint32)))
    assert (mine == theirs).all(), 'JS-style voting does not match model.predict'
    print('voting re-implementation matches model.predict on', len(mine), 'pairs')

    docs = {d['id']: d for d in parts['validation']}
    categories = json.loads(Path('outputs/dashboard-data/categories.json').read_text(encoding='utf-8'))
    y = np.array([p['label'] for p in pairs['validation']])
    rng = np.random.default_rng(7)
    picks = []
    for lab, pred in ((1, 1), (0, 0), (0, 1), (1, 0)):   # true positive, true negative, false positive, false negative
        idx = [i for i in range(len(y)) if y[i] == lab and mine[i] == pred
               and len(docs[pairs['validation'][i]['a']]['text']) > 150 and len(docs[pairs['validation'][i]['b']]['text']) > 150]
        picks += list(rng.choice(idx, 8, replace=False))
    rng.shuffle(picks)
    sample = []
    vocab_index = {w: i for i, w in enumerate(words)}
    for i in picks:
        p = pairs['validation'][i]
        a, b = docs[p['a']], docs[p['b']]
        present = lambda d: sorted(set(vocab_index[w] for w in count.build_analyzer()(d['text']) if w in vocab_index))
        sample.append({'a': snippet(a['text']), 'b': snippet(b['text']), 'catA': categories[a['category']],
                       'catB': categories[b['category']], 'label': int(p['label']), 'pred': int(mine[i]),
                       'scores': scores[i].tolist(), 'wa': present(a), 'wb': present(b)})

    def packed(row):
        return base64.b64encode(np.packbits(row).tobytes()).decode()

    results = {k: json.loads(Path(f'docs/results/{k}.json').read_text(encoding='utf-8')) for k in ('ablation', 'sweep', 'sweep2')}
    for rows in results.values():
        for r in rows:
            r.pop('accs', None)
    data = {'words': words, 'nFeatures': n_feat,
            'clauses': [packed(r) for r in include], 'weights': weights.tolist(),
            'sample': sample, 'results': results,
            'headline': {'tm': result['tm'], 'cosine': result['cosine'], 'cosine_auc': result['cosine_auc'],
                         'majority': result['majority'], 'order_consistency': result['order_consistency']},
            'rules': json.loads(Path('docs/results/rules_summary.json').read_text(encoding='utf-8')),
            'config': {k: cfg[k] for k in ('vocabulary', 'clauses', 'threshold', 'epochs', 'boost', 's', 'features', 'vocab_mode')}}
    OUT.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    print('wrote', OUT, round(OUT.stat().st_size / 1024), 'KB;', len(sample), 'sample pairs')
    print('sample correct:', sum(s['label'] == s['pred'] for s in sample), 'of', len(sample))


if __name__ == '__main__':
    main()
