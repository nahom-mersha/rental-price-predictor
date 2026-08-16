import logging
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

logger = logging.getLogger(__name__)
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting final model training.")

    model = train_final_model()

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)
    logger.info("Saved trained model to: %s", MODEL_PATH)


if __name__ == "__main__":
    main()
