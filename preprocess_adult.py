"""Preprocess UCI Adult data: merge adult.data and adult.test, create binary
target (income >50K), and drop rows with missing values ('?')."""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "Data", "adult")
OUTPUT_PATH = os.path.join(_HERE, "Data", "adult_preprocessed.csv")

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]


def main():
    read = dict(names=COLUMNS, skipinitialspace=True, na_values="?")
    train = pd.read_csv(os.path.join(DATA_DIR, "adult.data"), **read)
    # adult.test starts with a '|1x3 Cross validator' line
    test = pd.read_csv(os.path.join(DATA_DIR, "adult.test"), skiprows=1, **read)
    df = pd.concat([train, test], ignore_index=True)
    initial_rows = len(df)

    # Create 0/1 target: 1 = income >50K (adult.test labels end in '.')
    df["target"] = df["income"].str.startswith(">50K").astype(int)
    df = df.drop(columns="income")

    # Exclude rows with any missing value
    df = df.dropna()

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Initial rows:        {initial_rows}")
    print(f"Preprocessed rows:   {len(df)}")
    print(f"Rows removed:        {initial_rows - len(df)}")
    print(f"Target distribution:\n{df['target'].value_counts().sort_index()}")
    print(f"\nSex distribution:\n{df['sex'].value_counts()}")
    print(f"\nRow total of new dataset: {len(df)}")


if __name__ == "__main__":
    main()
