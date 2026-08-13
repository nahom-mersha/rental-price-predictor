import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
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
    """
    Combine preprocessing and a regression model.

    Every model goes through the same preprocessing workflow.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validated_mae(model, X_train, y_train):
    """Evaluate a model using leakage-safe cross-validation."""
    pipeline = build_model_pipeline(model)

    negative_mae_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
    )

    # scikit-learn returns negative MAE because larger scores are
    # considered better. Convert them back to normal positive MAE values.
    return -negative_mae_scores


# ---------------------------------------------------------------------------
# Model experiments
# ---------------------------------------------------------------------------


def evaluate_linear_regression(X_train, y_train):
    """Evaluate ordinary linear regression."""
    mae_scores = cross_validated_mae(
        LinearRegression(),
        X_train,
        y_train,
    )

    print("\nLinear Regression")
    print("CV MAE scores:", mae_scores)
    print("Mean CV MAE:", mae_scores.mean())


def evaluate_ridge_regression(X_train, y_train):
    """Compare several Ridge regularization strengths."""
    alpha_values = [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]

    ridge_results = []

    for alpha in alpha_values:
        mae_scores = cross_validated_mae(
            Ridge(alpha=alpha),
            X_train,
            y_train,
        )

        ridge_results.append((alpha, mae_scores.mean()))

    print("\nRidge Regression")

    for alpha, mean_mae in ridge_results:
        print(f"alpha={alpha}: mean CV MAE = {mean_mae}")


def run_tree_split_experiment(X_train, y_train):
    """
    Learning experiment:

    Keep depth and leaf size fixed while changing min_samples_split.

    This demonstrates how one tree hyperparameter affects
    cross-validation performance.
    """
    split_values = [
        2,
        5,
        10,
        20,
        50,
    ]

    tree_results = []

    for min_samples_split in split_values:
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
        )

        mae_scores = cross_validated_mae(
            model,
            X_train,
            y_train,
        )

        tree_results.append(
            (
                min_samples_split,
                mae_scores.mean(),
            )
        )

    print("\nDecision Tree min_samples_split experiment")

    for min_samples_split, mean_mae in tree_results:
        print(f"min_samples_split={min_samples_split}: mean CV MAE = {mean_mae}")


def run_tree_grid_search(X_train, y_train):
    """
    Search combinations of Decision Tree hyperparameters using 5-fold CV.

    Unlike the manual experiment above, GridSearchCV tests combinations
    of hyperparameters instead of changing only one at a time.
    """
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
    print(
        "Best mean CV MAE:",
        -grid_search.best_score_,
    )

    return grid_search


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    print("Dataset split")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training targets: {len(y_train)}")
    print(f"Test targets: {len(y_test)}")

    # Compare professional regression models using only the training set.
    evaluate_linear_regression(
        X_train,
        y_train,
    )

    evaluate_ridge_regression(
        X_train,
        y_train,
    )

    # Keep this manual experiment because it documents the learning process.
    run_tree_split_experiment(
        X_train,
        y_train,
    )

    # Systematically search combinations of tree hyperparameters.
    run_tree_grid_search(
        X_train,
        y_train,
    )


if __name__ == "__main__":
    main()
