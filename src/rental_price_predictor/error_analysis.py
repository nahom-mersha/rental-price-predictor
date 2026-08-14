from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge

from rental_price_predictor.sklearn_models import (
    build_model_pipeline,
    load_and_split_data,
)


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    ridge = build_model_pipeline(Ridge(alpha=10.0))
    ridge.fit(X_train, y_train)

    predictions = ridge.predict(X_test)

    residuals = y_test - predictions
    absolute_errors = abs(residuals)

    analysis_df = X_test.copy()
    analysis_df["actual_rent"] = y_test
    analysis_df["predicted_rent"] = predictions
    analysis_df["residual"] = residuals
    analysis_df["absolute_error"] = absolute_errors

    analysis_df = analysis_df.sort_values(
        "absolute_error",
        ascending=False,
    )

    print(
        analysis_df[
            [
                "livingSpace",
                "noRooms",
                "regio3",
                "actual_rent",
                "predicted_rent",
                "residual",
                "absolute_error",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    neighbourhood_errors = (
        analysis_df.groupby("regio3")
        .agg(
            test_samples=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
        )
        .loc[lambda df: df["test_samples"] >= 10]
        .sort_values("mae", ascending=False)
    )

    print("\nNeighbourhood MAE (at least 10 test listings):")
    print(neighbourhood_errors.to_string())
    preprocessor = ridge.named_steps["preprocessing"]
    model = ridge.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "absolute_coefficient": abs(model.coef_),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\nRidge feature effects:")
    print(coefficients_df.to_string(index=False))

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    plt.xlabel("Predicted monthly rent (€)")
    plt.ylabel("Residual (€)")
    plt.title("Ridge residuals vs. predicted rent")
    plt.tight_layout()
    figures_directory = Path("reports/figures")
    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        figures_directory / "ridge_residuals_vs_predicted_rent.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
