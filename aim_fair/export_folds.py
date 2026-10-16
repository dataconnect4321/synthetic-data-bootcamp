"""Write the Adult 3-fold CV splits used by run_adult_flip.py as raw CSVs,
so the AIM environment (aim_fair/.venv, no torch) trains on exactly the same
training rows. Run with the MAIN environment:

    python aim_fair/export_folds.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tabsyn-main', 'tabsyn-main'))
sys.path.insert(0, ROOT)
import run_adult_flip as R  # noqa: E402


def to_frame(num, cat, data):
    df = pd.DataFrame(num * data['stds'].values + data['means'].values,
                      columns=R.NUMERIC).round().astype(int)
    for j, c in enumerate(R.CATEGORICAL):
        df[c] = data['encoders'][c][cat[:, j]]
    return df[R.NUMERIC + R.CATEGORICAL]


if __name__ == '__main__':
    out = os.path.join(HERE, 'data')
    os.makedirs(out, exist_ok=True)
    df = R.read_adult()
    for f in range(3):
        data = R.prepare(df, f, 3, 42)
        to_frame(data['train_num'], data['train_cat'], data).to_csv(
            os.path.join(out, f'fold{f}_train.csv'), index=False)
        to_frame(data['test_num'], data['test_cat'], data).to_csv(
            os.path.join(out, f'fold{f}_test.csv'), index=False)
        print(f'fold {f}: {len(data["train_num"])} train / '
              f'{len(data["test_num"])} test')
