import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    X = model_df[
        [
            "livingSpace",
            "noRooms",
            "typeOfFlat",
            "regio3",
        ]
    ]
    y = model_df["baseRent"]

    numerical_features = [
        "livingSpace",
        "noRooms",
    ]

    categorical_features = [
        "typeOfFlat",
        "regio3",
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("scaling", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "one_hot_encoding",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    linear_model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    linear_cv_scores = cross_val_score(
        linear_model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
    )

    linear_mae_scores = -linear_cv_scores
    print("Linear Regression CV MAE scores:", linear_mae_scores)
    print("Linear Regression mean CV MAE:", linear_mae_scores.mean())

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

    for alpha, mean_mae in ridge_results:
        print(f"Ridge alpha={alpha}: mean CV MAE = {mean_mae}")


if __name__ == "__main__":
    main()
