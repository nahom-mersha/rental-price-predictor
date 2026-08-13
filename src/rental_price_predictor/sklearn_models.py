import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = "neg_mean_absolute_error"

NUMERICAL_FEATURES = [
    "livingSpace",
    "noRooms",
]

CATEGORICAL_FEATURES = [
    "typeOfFlat",
    "regio3",
]

TARGET = "baseRent"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_and_split_data():
    """Load Munich rental data, clean it, and create train/test sets."""
    raw_df = pd.read_csv("data/raw/immo_data.csv")

    # Keep Munich city listings only.
    munich_df = raw_df.loc[raw_df["regio2"].eq("München")].copy()

    # Keep only the features and target used by the professional models.
    model_df = munich_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

    # Remove obvious data-quality problems.
    model_df = model_df.loc[
        (model_df["baseRent"] >= 100) & (model_df["livingSpace"] <= 500)
    ].copy()

    X = model_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    y = model_df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def build_preprocessor():
    """Create leakage-safe preprocessing for numerical and categorical data."""
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaling",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "one_hot_encoding",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(model):
    """Combine preprocessing and a regression model."""
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validated_mae(
    model,
    X_train,
    y_train,
):
    """Evaluate a model using leakage-safe cross-validation."""
    pipeline = build_model_pipeline(model)

    negative_mae_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
    )

    # scikit-learn returns negative MAE because larger scores are treated
    # as better. Convert the values back to normal positive MAE.
    return -negative_mae_scores


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def evaluate_linear_regression(
    X_train,
    y_train,
):
    """Evaluate ordinary Linear Regression using cross-validation."""
    mae_scores = cross_validated_mae(
        LinearRegression(),
        X_train,
        y_train,
    )

    print("\nLinear Regression")
    print(f"Mean CV MAE: {mae_scores.mean():.2f}")


def evaluate_ridge_regression(
    X_train,
    y_train,
):
    """Compare Ridge regularization strengths using cross-validation."""
    alpha_values = [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    print("\nRidge Regression")

    for alpha in alpha_values:
        mae_scores = cross_validated_mae(
            Ridge(alpha=alpha),
            X_train,
            y_train,
        )

        print(f"alpha={alpha}: mean CV MAE = {mae_scores.mean():.2f}")


def run_tree_grid_search(
    X_train,
    y_train,
):
    """Tune Decision Tree hyperparameters using GridSearchCV."""
    tree_pipeline = build_model_pipeline(
        DecisionTreeRegressor(random_state=RANDOM_STATE)
    )

    parameter_grid = {
        "model__max_depth": [
            5,
            10,
            15,
            None,
        ],
        "model__min_samples_leaf": [
            5,
            10,
            20,
        ],
        "model__min_samples_split": [
            2,
            20,
            50,
        ],
    }

    grid_search = GridSearchCV(
        estimator=tree_pipeline,
        param_grid=parameter_grid,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print("\nDecision Tree GridSearchCV")
    print(
        "Best parameters:",
        grid_search.best_params_,
    )
    print(f"Best mean CV MAE: {-grid_search.best_score_:.2f}")

    return grid_search


# ---------------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test_set(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Fit a selected model and evaluate it on the held-out test set."""
    pipeline = build_model_pipeline(model)

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"\n{name} — Final Test")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")


def evaluate_tree_on_test_set(
    tree_grid_search,
    X_test,
    y_test,
):
    """
    Evaluate the best Decision Tree selected by GridSearchCV.

    GridSearchCV refits the best model on the full training set by default,
    so another fit is not needed here.
    """
    predictions = tree_grid_search.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\nDecision Tree — Final Test")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("Dataset split")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # -----------------------------------------------------------------------
    # Mean baseline
    # -----------------------------------------------------------------------

    baseline_prediction = y_train.mean()

    baseline_predictions = [baseline_prediction] * len(y_test)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    baseline_rmse = (
        mean_squared_error(
            y_test,
            baseline_predictions,
        )
        ** 0.5
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions,
    )

    print("\nMean Baseline")
    print(f"MAE: {baseline_mae:.2f}")
    print(f"RMSE: {baseline_rmse:.2f}")
    print(f"R²: {baseline_r2:.3f}")

    # -----------------------------------------------------------------------
    # Model selection using training data only
    # -----------------------------------------------------------------------

    evaluate_linear_regression(
        X_train,
        y_train,
    )

    evaluate_ridge_regression(
        X_train,
        y_train,
    )

    tree_grid_search = run_tree_grid_search(
        X_train,
        y_train,
    )

    # -----------------------------------------------------------------------
    # Final evaluation on the held-out test set
    # -----------------------------------------------------------------------

    evaluate_on_test_set(
        "Linear Regression",
        LinearRegression(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    evaluate_on_test_set(
        "Ridge Regression",
        Ridge(alpha=10.0),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    evaluate_tree_on_test_set(
        tree_grid_search,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
