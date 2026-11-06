"""Reproducible document splitting, pairing, features, and baselines."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split


def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def split_documents(train, test, seed=42):
    def records(dataset, prefix):
        return [{"id": f"{prefix}:{i}", "text": text, "category": int(label)}
                for i, (text, label) in enumerate(zip(dataset.data, dataset.target)) if normalize(text)]
    tr, te = records(train, "train"), records(test, "test")
    keep, val = train_test_split(tr, test_size=.2, random_state=seed,
                                 stratify=[d['category'] for d in tr])
    # Priority test > validation > training. Keep duplicates within a partition.
    def digest(d): return hashlib.sha256(normalize(d['text']).encode('utf-8')).hexdigest()
    test_hash = {digest(d) for d in te}
    val = [d for d in val if digest(d) not in test_hash]
    held = test_hash | {digest(d) for d in val}
    keep = [d for d in keep if digest(d) not in held]
    return {'train': keep, 'validation': val, 'test': te}


def make_pairs(documents, count, seed=42, augment=False):
    if count % (4 if augment else 2):
        raise ValueError('count must permit equal positive and negative pairs and both orientations')
    rng = np.random.default_rng(seed)
    by_category = {}
    for d in documents: by_category.setdefault(d['category'], []).append(d)
    categories = sorted(by_category)
    eligible = [c for c in categories if len(by_category[c]) >= 2]
    unique_count = count // 2 if augment else count
    seen, pairs = set(), []
    for label in (0, 1):
        attempts = 0
        while sum(p['label'] == label for p in pairs) < unique_count // 2:
            attempts += 1
            if attempts > unique_count * 1000: raise ValueError('not enough unique pairs')
            if label:
                c = int(rng.choice(eligible)); a, b = rng.choice(by_category[c], 2, replace=False)
            else:
                c1, c2 = rng.choice(categories, 2, replace=False)
                a, b = rng.choice(by_category[c1]), rng.choice(by_category[c2])
            key = tuple(sorted((a['id'], b['id'])))
            if key in seen: continue
            seen.add(key)
            if rng.integers(2): a, b = b, a
            pairs.append({'a': a['id'], 'b': b['id'], 'category_a': a['category'],
                          'category_b': b['category'], 'label': label})
    rng.shuffle(pairs)
    if augment:
        pairs = [q for p in pairs for q in (p, reverse_pair(p))]
    return pairs


def reverse_pair(p):
    return {'a': p['b'], 'b': p['a'], 'category_a': p['category_b'],
            'category_b': p['category_a'], 'label': p['label']}


FEATURE_MODES = ('ab', 'sym')
LETTERS_ONLY = r'(?u)\b[a-zA-Z]{3,}\b'


def make_vectorizer(kind, vocabulary, vocab_mode='letters'):
    """Build a CountVectorizer ('count', binary) or TfidfVectorizer ('tfidf').

    vocab_mode 'letters' keeps only words of 3+ letters (drops numbers like 1990);
    'default' is scikit-learn's standard tokenizer (the original behaviour).
    """
    kw = {'max_features': vocabulary, 'stop_words': 'english'}
    if vocab_mode == 'letters':
        kw['token_pattern'] = LETTERS_ONLY
    return CountVectorizer(binary=True, **kw) if kind == 'count' else TfidfVectorizer(**kw)


def pair_features(pairs, documents, vectorizer, mode='ab'):
    """Boolean TM input for each pair.

    'ab'  : [A words | B words]            (depends on which document is A)
    'sym' : [in BOTH | in exactly ONE]     (A and B can be swapped freely)
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f'mode must be one of {FEATURE_MODES}')
    docs = {d['id']: d for d in documents}
    a = vectorizer.transform([docs[p['a']]['text'] for p in pairs]).toarray()
    b = vectorizer.transform([docs[p['b']]['text'] for p in pairs]).toarray()
    if mode == 'sym':
        a, b = (a > 0), (b > 0)
        return np.ascontiguousarray(np.hstack([a & b, a ^ b]), dtype=np.uint32)
    return np.ascontiguousarray(np.hstack([a, b]), dtype=np.uint32)


def feature_names(words, mode='ab'):
    """Readable names matching the columns of pair_features."""
    if mode == 'sym':
        return [f'BOTH.{w}' for w in words] + [f'ONE.{w}' for w in words]
    return [f'A.{w}' for w in words] + [f'B.{w}' for w in words]


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'precision': float(p),
            'recall': float(r), 'f1': float(f), 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def cosine_scores(pairs, documents, vectorizer):
    docs = {d['id']: d for d in documents}
    a = vectorizer.transform([docs[p['a']]['text'] for p in pairs])
    b = vectorizer.transform([docs[p['b']]['text'] for p in pairs])
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def best_cosine_threshold(scores, y):
    """Select the highest validation accuracy, breaking ties toward the lower cutoff."""
    scores = np.asarray(scores); y = np.asarray(y)
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
    return float(max(candidates, key=lambda t: (np.mean((scores >= t) == y), -t)))


def auc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def prepare(output, train_count=20000, validation_count=4000, test_count=4000, seed=42):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
    test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
    parts = split_documents(train, test, seed)
    pairs = {name: make_pairs(docs, n, seed, name == 'train') for (name, docs), n in
             zip(parts.items(), (train_count, validation_count, test_count))}
    for name in parts:
        (out / f'{name}_documents.json').write_text(json.dumps(parts[name], ensure_ascii=False), encoding='utf-8')
        (out / f'{name}_pairs.json').write_text(json.dumps(pairs[name], indent=2), encoding='utf-8')
    (out / 'categories.json').write_text(json.dumps(train.target_names, indent=2), encoding='utf-8')
    return parts, pairs
