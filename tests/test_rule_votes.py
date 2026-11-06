from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_dashboard_data import clause_scores
from visualize_rules import parse


def test_signed_binary_vote_direction(tmp_path):
    rules = tmp_path / 'rules.txt'
    rules.write_text(
        'IF BOTH.a THEN -1 0\nIF ONE.b THEN +1 0\n'
        'IF NOT BOTH.a THEN -1 1\nIF NOT ONE.b THEN +1 1\n'
        'IF ONE.a THEN +1 0, +1 1\n',
        encoding='utf-8',
    )
    assert [clause['supports'] for clause in parse(rules)] == [1, 0, 0, 1, None]


def test_extracted_scores_handle_empty_negative_and_tied_votes():
    # Include columns stand for x and NOT x. A clause with neither never fires.
    include = np.array([[0, 0], [1, 0], [0, 1], [1, 0]])
    weights = np.array([[9, 9], [-1, 0], [0, 1], [0, -1]])
    x = np.array([[0], [1]], dtype=np.uint8)
    scores = clause_scores(x, include, weights)
    assert scores.tolist() == [[0, 1], [-1, -1]]
    assert scores.argmax(axis=1).tolist() == [1, 0]
