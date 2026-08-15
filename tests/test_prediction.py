from typing import cast

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from rental_price_predictor.prediction import predict_rent


class RecordingModel:
    """Small fake model that records the data passed to predict()."""

    def __init__(self) -> None:
        self.input_df: pd.DataFrame | None = None

    def predict(self, input_df: pd.DataFrame) -> list[float]:
        self.input_df = input_df.copy()
        return [1234.56]


def test_predict_rent_returns_float_and_builds_expected_input():
    recording_model = RecordingModel()
    model = cast(Pipeline, recording_model)
    prediction = predict_rent(
        model,
        living_space=70.0,
        no_rooms=2.0,
        neighbourhood="Schwabing",
        balcony=False,
        garden=False,
        lift=True,
        has_kitchen=True,
        cellar=True,
    )

    assert prediction == 1234.56
    assert isinstance(prediction, float)

    assert recording_model.input_df is not None

    assert list(recording_model.input_df.columns) == [
        "livingSpace",
        "noRooms",
        "yearConstructed",
        "floor",
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

    assert recording_model.input_df.loc[0, "livingSpace"] == 70.0
    assert recording_model.input_df.loc[0, "regio3"] == "Schwabing"
    assert pd.isna(recording_model.input_df.loc[0, "yearConstructed"])


def test_predict_rent_rejects_invalid_living_space():
    recording_model = RecordingModel()
    model = cast(Pipeline, recording_model)

    with pytest.raises(
        ValueError,
        match="living_space must be greater than 0",
    ):
        predict_rent(
            model,
            living_space=-10.0,
            no_rooms=2.0,
            neighbourhood="Schwabing",
            balcony=False,
            garden=False,
            lift=False,
            has_kitchen=True,
            cellar=True,
        )


def test_predict_rent_rejects_non_boolean_amenity():
    recording_model = RecordingModel()
    model = cast(Pipeline, recording_model)

    with pytest.raises(
        ValueError,
        match="balcony must be True or False",
    ):
        predict_rent(
            model,
            living_space=70.0,
            no_rooms=2.0,
            neighbourhood="Schwabing",
            balcony="yes",  # type: ignore[arg-type]
            garden=False,
            lift=False,
            has_kitchen=True,
            cellar=True,
        )
