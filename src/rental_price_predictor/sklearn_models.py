import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def main():
    raw_df = pd.read_csv("data/raw/immo_data.csv")

    # Keep Munich city listings only.
    df = raw_df.loc[raw_df["regio2"].eq("München")].copy()

    # Keep the selected features and target.
    model_df = df[
        [
            "livingSpace",
            "noRooms",
            "typeOfFlat",
            "regio3",
            "baseRent",
        ]
    ].copy()

    # Remove obvious data-quality problems.
    model_df = model_df.loc[
        (model_df["baseRent"] >= 100) & (model_df["livingSpace"] <= 500)
    ].copy()

    numerical_features = [
        "livingSpace",
        "noRooms",
    ]

    categorical_features = [
        "typeOfFlat",
        "regio3",
    ]

    X = model_df[numerical_features + categorical_features]
    y = model_df["baseRent"]

    # Reserve the test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training targets: {len(y_train)}")
    print(f"Test targets: {len(y_test)}")

    # Numerical preprocessing: fill missing values, then standardize.
    numerical_pipeline = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("scaling", StandardScaler()),
        ]
    )

    # Categorical preprocessing: fill missing values, then one-hot encode.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(strategy="constant", fill_value="missing"),
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Linear Regression.
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    linear_cv_scores = cross_val_score(
        linear_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    linear_mae_scores = -linear_cv_scores

    print("\nLinear Regression")
    print("CV MAE scores:", linear_mae_scores)
    print("Mean CV MAE:", linear_mae_scores.mean())

    # Ridge Regression: compare several regularization strengths.
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_results = []

    for alpha in alpha_values:
        ridge_pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", Ridge(alpha=alpha)),
            ]
        )

        ridge_cv_scores = cross_val_score(
            ridge_pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
        )

        ridge_mae_scores = -ridge_cv_scores
        mean_ridge_mae = ridge_mae_scores.mean()

        ridge_results.append((alpha, mean_ridge_mae))

    print("\nRidge Regression")

    for alpha, mean_mae in ridge_results:
        print(f"alpha={alpha}: mean CV MAE = {mean_mae}")

    split_values = [2, 5, 10, 20, 50]
    tree_split_results = []

    for min_samples_split in split_values:
        tree_pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=10,
                        min_samples_leaf=10,
                        min_samples_split=min_samples_split,
                        random_state=42,
                    ),
                ),
            ]
        )

        tree_cv_scores = cross_val_score(
            tree_pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
        )

        tree_mae_scores = -tree_cv_scores
        mean_tree_mae = tree_mae_scores.mean()

        tree_split_results.append((min_samples_split, mean_tree_mae))

    print("\nDecision Tree min_samples_split experiment")

    for min_samples_split, mean_mae in tree_split_results:
        print(f"min_samples_split={min_samples_split}: mean CV MAE = {mean_mae}")

    tree_grid_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )
    tree_param_grid = {
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_leaf": [5, 10, 20],
        "model__min_samples_split": [2, 20, 50],
    }

    tree_grid_search = GridSearchCV(
        estimator=tree_grid_pipeline,
        param_grid=tree_param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    tree_grid_search.fit(X_train, y_train)

    print("\nDecision Tree GridSearchCV")
    print("Best parameters:", tree_grid_search.best_params_)
    print("Best mean CV MAE:", -tree_grid_search.best_score_)


if __name__ == "__main__":
    main()
