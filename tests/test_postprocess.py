"""Generic numeric post-processing: infer each real column's format
(decimals, range) and apply it to synthetic data.

Run:  python -m pytest tests/test_postprocess.py -v
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from run_flip_demo import infer_numeric_schema, apply_numeric_schema


def _real():
    rng = np.random.default_rng(0)
    age = rng.integers(17, 90, 500).astype(float)            # whole
    price = np.round(rng.uniform(0, 100, 500), 2)            # cents
    amount = rng.integers(0, 5000, 500).astype(float)        # >= 0, whole
    return np.stack([age, price, amount], 1), ['age', 'price', 'amount']


def test_infer_schema_finds_decimals_and_bounds():
    X, names = _real()
    s = infer_numeric_schema(X, names)
    assert s['age']['decimals'] == 0 and s['amount']['decimals'] == 0
    assert s['price']['decimals'] == 2
    assert s['amount']['lo'] >= 0


def test_infer_schema_tolerates_float_error():
    X, names = _real()
    noisy = X + 1e-9 * np.random.default_rng(1).normal(size=X.shape)
    assert infer_numeric_schema(noisy, names)['age']['decimals'] == 0


def test_apply_schema_rounds_and_clips():
    X, names = _real()
    s = infer_numeric_schema(X, names)
    syn = pd.DataFrame({'age': [36.98, 120.4], 'price': [3.14159, -2.0],
                        'amount': [-50.3, 12.6], 'other': ['a', 'b']})
    out = apply_numeric_schema(syn, s)
    assert out['age'].tolist() == [37, int(s['age']['hi'])]
    assert out['age'].dtype.kind == 'i'
    assert out['price'].tolist() == [3.14, s['price']['lo']]
    assert out['amount'].tolist() == [int(s['amount']['lo']), 13]
    assert out['other'].tolist() == ['a', 'b']        # untouched


def test_apply_schema_without_bounds_only_rounds():
    X, names = _real()
    s = infer_numeric_schema(X, names)
    out = apply_numeric_schema(pd.DataFrame({'amount': [-50.3]}), s,
                               bounds=False)
    assert out['amount'].tolist() == [-50]
