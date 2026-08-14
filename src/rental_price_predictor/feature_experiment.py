import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from rental_price_predictor.sklearn_models import (
    build_model_pipeline,
    cross_validated_mae,
    load_and_split_data,
)

BASE_NUMERICAL_FEATURES = [
    "livingSpace",
    "noRooms",
]

BASE_CATEGORICAL_FEATURES = [
    "typeOfFlat",
    "regio3",
]

ENHANCED_NUMERICAL_FEATURES = [
    "livingSpace",
    "noRooms",
    "yearConstructed",
    "floor",
]

ENHANCED_CATEGORICAL_FEATURES = [
    "typeOfFlat",
    "regio3",
    "interiorQual",
    "condition",
    "balcony",
    "garden",
    "lift",
    "hasKitchen",
    "cellar",
]

ALPHA_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


def select_ridge_alpha(
    X_train,
    y_train,
    numerical_features,
    categorical_features,
):
    """Choose Ridge alpha using training-set cross-validation only."""
    results = []

    for alpha in ALPHA_VALUES:
        mae_scores = cross_validated_mae(
            Ridge(alpha=alpha),
            X_train,
            y_train,
            numerical_features,
            categorical_features,
        )

        results.append(
            {
                "alpha": alpha,
                "mean_cv_mae": mae_scores.mean(),
            }
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["mean_cv_mae"].idxmin()]

    return best_row["alpha"], results_df


def evaluate_ridge(
    X_train,
    X_test,
    y_train,
    y_test,
    alpha,
    numerical_features,
    categorical_features,
):
    """Fit Ridge and return held-out test metrics and predictions."""
    pipeline = build_model_pipeline(
        Ridge(alpha=alpha),
        numerical_features,
        categorical_features,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "predictions": predictions,
    }


def main():
    # Load once with enhanced columns. The baseline columns are then selected
    # from the same rows, guaranteeing the identical train/test split.
    X_train, X_test, y_train, y_test = load_and_split_data(
        ENHANCED_NUMERICAL_FEATURES,
        ENHANCED_CATEGORICAL_FEATURES,
    )

    base_features = BASE_NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES

    X_train_baseline = X_train[base_features]
    X_test_baseline = X_test[base_features]

    print("Enhanced Ridge feature experiment")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # The baseline alpha=10 was selected in the earlier model-selection step.
    baseline_results = evaluate_ridge(
        X_train_baseline,
        X_test_baseline,
        y_train,
        y_test,
        alpha=10.0,
        numerical_features=BASE_NUMERICAL_FEATURES,
        categorical_features=BASE_CATEGORICAL_FEATURES,
    )

    best_alpha, cv_results = select_ridge_alpha(
        X_train,
        y_train,
        ENHANCED_NUMERICAL_FEATURES,
        ENHANCED_CATEGORICAL_FEATURES,
    )

    print("\nEnhanced Ridge cross-validation:")
    print(cv_results.to_string(index=False))
    print(f"\nSelected enhanced alpha: {best_alpha}")

    enhanced_results = evaluate_ridge(
        X_train,
        X_test,
        y_train,
        y_test,
        alpha=best_alpha,
        numerical_features=ENHANCED_NUMERICAL_FEATURES,
        categorical_features=ENHANCED_CATEGORICAL_FEATURES,
    )

    # Use the same actual high-rent listings for both models: top 10%.
    high_rent_threshold = y_test.quantile(0.90)
    high_rent_mask = y_test >= high_rent_threshold

    baseline_high_rent_mae = mean_absolute_error(
        y_test[high_rent_mask],
        baseline_results["predictions"][high_rent_mask],
    )

    enhanced_high_rent_mae = mean_absolute_error(
        y_test[high_rent_mask],
        enhanced_results["predictions"][high_rent_mask],
    )

    print("\nHeld-out test comparison:")
    print(f"Baseline Ridge (alpha=10) MAE: {baseline_results['mae']:.2f}")
    print(f"Enhanced Ridge (alpha={best_alpha}) MAE: {enhanced_results['mae']:.2f}")
    print(f"Baseline Ridge RMSE: {baseline_results['rmse']:.2f}")
    print(f"Enhanced Ridge RMSE: {enhanced_results['rmse']:.2f}")
    print(f"Baseline Ridge R²: {baseline_results['r2']:.3f}")
    print(f"Enhanced Ridge R²: {enhanced_results['r2']:.3f}")

    print(f"\nHigh-rent test subset (top 10%, rent ≥ €{high_rent_threshold:.2f}):")
    print(f"Baseline high-rent MAE: {baseline_high_rent_mae:.2f}")
    print(f"Enhanced high-rent MAE: {enhanced_high_rent_mae:.2f}")


if __name__ == "__main__":
    main()
