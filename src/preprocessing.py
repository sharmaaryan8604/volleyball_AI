import pandas as pd
import numpy as np


def load_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def clean_dataset(df):

    numeric_cols = [
        "receive_location",
        "digger_location",
        "pass_land_location",
        "hitter_location",
        "set_location",
        "hit_land_location"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["hit_land_location"])

    df.loc[:, "hit_land_location"] = (
        df["hit_land_location"]
        .round()
        .astype(int)
    )

    df = df[df["hit_land_location"].between(1, 25)]

    return df