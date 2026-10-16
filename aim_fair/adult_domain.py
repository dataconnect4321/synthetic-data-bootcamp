"""Discretize UCI Adult for AIM / Private-PGM and map synthetic rows back.

Marginal-based mechanisms need every attribute discrete. Numeric columns are
binned with FIXED, PUBLIC edges (the UCI documentation's ranges and round
numbers), not quantiles of the data: data-dependent edges would be an extra
release outside the privacy accounting. Category lists are the public schema.
Decoding draws an integer uniformly inside each bin (exact for one-value
bins such as capital-gain = 0, hours = 40 or each education-num value).
"""
import numpy as np
import pandas as pd

NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
           'hours-per-week']
CATEGORICAL = ['sex', 'income', 'workclass', 'marital-status', 'occupation',
               'relationship', 'race', 'native-country']

# Right-open bin edges [lo, hi). Public ranges: age 17-90, fnlwgt ~1e4-1.5e6,
# education-num 1-16, capital-gain 0-99999, capital-loss 0-4356, hours 1-99.
EDGES = {
    'age': [17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 91],
    'fnlwgt': [10000, 50000, 75000, 100000, 125000, 150000, 175000, 200000,
               225000, 250000, 300000, 350000, 400000, 500000, 1600000],
    'education-num': list(range(1, 18)),
    'capital-gain': [0, 1, 3000, 5000, 7500, 10000, 15000, 25000, 100000],
    'capital-loss': [0, 1, 1500, 1800, 2000, 2300, 4400],
    'hours-per-week': [1, 20, 30, 35, 40, 41, 45, 50, 55, 60, 100],
}


def domain_sizes(categories):
    """{attribute: number of values}; categories: {col: list of labels}."""
    sizes = {c: len(EDGES[c]) - 1 for c in NUMERIC}
    sizes.update({c: len(categories[c]) for c in CATEGORICAL})
    return sizes


def encode(df, categories):
    """Raw Adult frame -> {attribute: int array} for mbi.Dataset."""
    out = {}
    for c in NUMERIC:
        e = np.asarray(EDGES[c])
        x = np.clip(df[c].to_numpy(), e[0], e[-1] - 1)
        out[c] = (np.searchsorted(e, x, side='right') - 1).astype(np.int64)
    for c in CATEGORICAL:
        lookup = {str(v): i for i, v in enumerate(categories[c])}
        out[c] = df[c].astype(str).map(lookup).to_numpy().astype(np.int64)
    return out


def decode(codes, categories, seed=0):
    """{attribute: int array} -> raw Adult frame (numerics uniform in bin)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    for c in NUMERIC:
        e = np.asarray(EDGES[c])
        k = np.asarray(codes[c])
        lo, hi = e[k], e[k + 1]
        df[c] = rng.integers(lo, np.maximum(hi, lo + 1))
    for c in CATEGORICAL:
        df[c] = np.asarray(categories[c], dtype=object)[np.asarray(codes[c])]
    return df[NUMERIC + CATEGORICAL]
