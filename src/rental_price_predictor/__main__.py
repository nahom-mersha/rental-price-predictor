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

    # Add intercept column.
    X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])

    # Gradient descent solution.
    weights = np.zeros(X_train_design.shape[1])

    learning_rate = 0.01
    iterations = 5000
    n = len(y_train)

    for _ in range(iterations):
        y_train_predictions = X_train_design @ weights

        train_errors = y_train_predictions - y_train

        gradient = (2 / n) * (X_train_design.T @ train_errors)

        weights = weights - learning_rate * gradient

    gradient_descent_predictions = X_train_design @ weights

    gradient_descent_errors = gradient_descent_predictions - y_train

    gradient_descent_mse = np.mean(gradient_descent_errors**2)

    # Normal equation solution.
    normal_weights = (
        np.linalg.inv(X_train_design.T @ X_train_design) @ X_train_design.T @ y_train
    )

    normal_predictions = X_train_design @ normal_weights

    normal_errors = normal_predictions - y_train

    normal_mse = np.mean(normal_errors**2)

    print("\nGradient descent weights:", weights)
    print("Gradient descent train MSE:", gradient_descent_mse)

    print("\nNormal equation weights:", normal_weights)
    print("Normal equation train MSE:", normal_mse)

    # Pseudoinverse solution.
    pinv_weights = np.linalg.pinv(X_train_design) @ y_train

    pinv_predictions = X_train_design @ pinv_weights
    pinv_errors = pinv_predictions - y_train
    pinv_mse = np.mean(pinv_errors**2)

    # Least-squares solution.
    lstsq_weights = np.linalg.lstsq(
        X_train_design,
        y_train,
        rcond=None,
    )[0]

    lstsq_predictions = X_train_design @ lstsq_weights
    lstsq_errors = lstsq_predictions - y_train
    lstsq_mse = np.mean(lstsq_errors**2)

    print("\nPseudoinverse weights:", pinv_weights)
    print("Pseudoinverse train MSE:", pinv_mse)

    print("\nLeast-squares weights:", lstsq_weights)
    print("Least-squares train MSE:", lstsq_mse)


if __name__ == "__main__":
    main()
