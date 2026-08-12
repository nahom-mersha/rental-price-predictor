import matplotlib.pyplot as plt
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

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.3]
    iterations = 5000

    n = len(y_train)

    all_loss_histories = {}

    for learning_rate in learning_rates:
        weights = np.zeros(X_train_design.shape[1])
        loss_history = []

        for _ in range(iterations):
            y_train_predictions = X_train_design @ weights

            train_errors = y_train_predictions - y_train

            train_mse = np.mean(train_errors**2)

            gradient = (2 / n) * (X_train_design.T @ train_errors)

            weights = weights - learning_rate * gradient

            loss_history.append(train_mse)

        all_loss_histories[learning_rate] = loss_history

        print(f"Learning rate {learning_rate}: final MSE = {loss_history[-1]}")

    for learning_rate, loss_history in all_loss_histories.items():
        plt.plot(
            range(iterations),
            loss_history,
            label=f"lr={learning_rate}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.title("Learning Rate Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
