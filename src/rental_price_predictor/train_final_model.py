from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from rental_price_predictor.sklearn_models import (
    ENHANCED_CATEGORICAL_FEATURES,
    ENHANCED_NUMERICAL_FEATURES,
    build_model_pipeline,
    load_clean_model_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "enhanced_ridge.joblib"


def train_final_model() -> Pipeline:
    """Train the selected enhanced Ridge model on all cleaned data."""
    X, y = load_clean_model_data(
        ENHANCED_NUMERICAL_FEATURES,
        ENHANCED_CATEGORICAL_FEATURES,
    )

    model = build_model_pipeline(
        Ridge(alpha=10.0),
        ENHANCED_NUMERICAL_FEATURES,
        ENHANCED_CATEGORICAL_FEATURES,
    )

    model.fit(X, y)

    return model


def main() -> None:
    """Train and save the final deployable model."""
    model = train_final_model()

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
