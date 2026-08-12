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

    # Reproducible 80/20 train-test split.
    rng = np.random.default_rng(42)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_index = int(0.8 * len(X))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    # X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    # Mean baseline.
    baseline_prediction = y_train.mean()

    y_test_baseline_predictions = np.full(
        shape=y_test.shape,
        fill_value=baseline_prediction,
    )

    baseline_mae = np.mean(np.abs(y_test_baseline_predictions - y_test))

    baseline_rmse = np.sqrt(np.mean((y_test_baseline_predictions - y_test) ** 2))

    print("Baseline MAE:", baseline_mae)
    print("Baseline RMSE:", baseline_rmse)

    # Scale features using training-set statistics only.
    feature_means = X_train.mean(axis=0)
    feature_stds = X_train.std(axis=0)

    X_train_scaled = (X_train - feature_means) / feature_stds
    # X_test_scaled = (X_test - feature_means) / feature_stds

    # Add intercept column.
    X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])

    # X_test_design = np.column_stack([np.ones(X_test_scaled.shape[0]), X_test_scaled])

    # Start linear regression with all weights at zero.
    weights = np.zeros(X_train_design.shape[1])

    # Initial prediction and loss.
    y_train_predictions = X_train_design @ weights
    train_errors = y_train_predictions - y_train
    initial_train_mse = np.mean(train_errors**2)

    # Compute vectorized gradient.
    n = len(y_train)

    gradient = (2 / n) * (X_train_design.T @ train_errors)

    print("Initial weights:", weights)
    print("Initial gradient:", gradient)
    print("Initial train MSE:", initial_train_mse)

    # Perform one gradient-descent update.
    learning_rate = 0.001

    weights = weights - learning_rate * gradient

    # Recalculate predictions and loss using the updated weights.
    y_train_predictions = X_train_design @ weights
    train_errors = y_train_predictions - y_train
    updated_train_mse = np.mean(train_errors**2)

    print("Updated weights:", weights)
    print("Train MSE after one update:", updated_train_mse)


if __name__ == "__main__":
    main()
