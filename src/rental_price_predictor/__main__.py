import numpy as np
import pandas as pd


def main():
    raw_df = pd.read_csv("data/raw/immo_data.csv")

    # Keep Munich city listings only.
    df = raw_df.loc[raw_df["regio2"].eq("München")].copy()

    # Keep the baseline features and target.
    model_df = df[["livingSpace", "noRooms", "baseRent"]].dropna()

    # Remove obvious data-quality errors found during inspection.
    model_df = model_df.loc[
        (model_df["baseRent"] >= 100) & (model_df["livingSpace"] <= 500)
    ].copy()

    X = model_df[["livingSpace", "noRooms"]].to_numpy()

    y = model_df["baseRent"].to_numpy()

    rng = np.random.default_rng(42)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_index = int(0.8 * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_train_design = np.column_stack([np.ones(X_train.shape[0]), X_train])

    X_test_design = np.column_stack([np.ones(X_test.shape[0]), X_test])

    print("Clean rows:", len(model_df))
    print("X_train_design:", X_train_design.shape)
    print("X_test_design:", X_test_design.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)


if __name__ == "__main__":
    main()
