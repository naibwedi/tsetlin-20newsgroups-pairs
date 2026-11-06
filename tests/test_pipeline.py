import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline import (make_pairs, reverse_pair, metrics, split_documents, make_vectorizer,
                      pair_features, feature_names, best_cosine_threshold)

def docs():
    return [{'id': f'{c}-{i}', 'text': f'word {c} item {i}', 'category': c}
            for c in range(3) for i in range(20)]

def test_pairs_balanced_unique_and_reproducible():
    pairs = make_pairs(docs(), 40, seed=42)
    assert pairs == make_pairs(docs(), 40, seed=42)
    assert sum(p['label'] for p in pairs) == 20
    assert len({tuple(sorted((p['a'], p['b']))) for p in pairs}) == 40
    assert all(p['a'] != p['b'] and p['label'] == (p['category_a'] == p['category_b']) for p in pairs)

def test_augmentation_has_reverse_once():
    pairs = make_pairs(docs(), 40, seed=42, augment=True)
    assert all(reverse_pair(pairs[i]) == pairs[i+1] for i in range(0, 40, 2))
    assert sum(p['label'] for p in pairs) == 20

def test_metrics():
    m = metrics([0,0,1,1], [0,1,1,1])
    assert m['accuracy'] == .75 and m['recall'] == 1
    assert m['confusion_matrix'] == [[1,1],[0,2]]

def test_split_removes_cross_split_exact_duplicates():
    train = SimpleNamespace(data=[f'doc {c} {i}' for c in range(2) for i in range(20)], target=[c for c in range(2) for i in range(20)])
    test = SimpleNamespace(data=['DOC 0 0', 'test other'], target=[0,1])
    parts = split_documents(train, test)
    assert all(d['text'].lower() != 'doc 0 0' for name in ('train','validation') for d in parts[name])
    ids = [{d['id'] for d in v} for v in parts.values()]
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i+1,3))


def small_setup():
    documents = [{'id': 'x', 'text': 'space orbit rocket', 'category': 0},
                 {'id': 'y', 'text': 'space hockey 1990', 'category': 1}]
    vec = make_vectorizer('count', 10, 'letters').fit([d['text'] for d in documents])
    return documents, vec


def test_sym_features_do_not_depend_on_order():
    documents, vec = small_setup()
    p = {'a': 'x', 'b': 'y', 'category_a': 0, 'category_b': 1, 'label': 0}
    assert (pair_features([p], documents, vec, 'sym') == pair_features([reverse_pair(p)], documents, vec, 'sym')).all()
    assert not (pair_features([p], documents, vec, 'ab') == pair_features([reverse_pair(p)], documents, vec, 'ab')).all()


def test_sym_features_mean_both_and_one():
    documents, vec = small_setup()
    words = vec.get_feature_names_out().tolist()
    p = {'a': 'x', 'b': 'y', 'category_a': 0, 'category_b': 1, 'label': 0}
    row = pair_features([p], documents, vec, 'sym')[0]
    named = dict(zip(feature_names(words, 'sym'), row))
    assert named['BOTH.space'] == 1 and named['ONE.space'] == 0
    assert named['ONE.orbit'] == 1 and named['BOTH.orbit'] == 0
    assert named['ONE.hockey'] == 1


def test_letters_vocabulary_drops_numbers():
    _, vec = small_setup()
    assert '1990' not in vec.get_feature_names_out()


def test_cosine_threshold_uses_accuracy_not_all_positive():
    scores = np.array([0.0, 0.0, 0.1, 0.5, 0.6, 0.7])
    y = np.array([0, 0, 0, 1, 1, 1])
    t = best_cosine_threshold(scores, y)
    assert 0.1 < t <= 0.5
    assert ((scores >= t) == y).all()
