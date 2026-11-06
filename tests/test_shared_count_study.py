import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_count_study import append_count_bits


def test_count_cutoffs_use_only_distinct_shared_words():
    # Each row has exactly five words across the pair. Only their overlap varies.
    shared = np.array([[1] * k + [0] * (5 - k) for k in range(6)], dtype=np.uint8)
    inputs = np.column_stack((shared, 1 - shared))
    extended = append_count_bits(inputs, vocabulary_size=5)
    np.testing.assert_array_equal(extended[:, :10], inputs)
    np.testing.assert_array_equal(extended[:, 10:], [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1],
    ])
