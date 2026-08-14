import math
from numbers import Real

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_rent(
    model: Pipeline,
    living_space: float,
    no_rooms: float,
    neighbourhood: str,
    *,
    balcony: bool,
    garden: bool,
    lift: bool,
    has_kitchen: bool,
    cellar: bool,
    flat_type: str | None = None,
    year_constructed: float | None = None,
    floor: float | None = None,
    interior_quality: str | None = None,
    condition: str | None = None,
) -> float:
    """Predict monthly cold rent in euros for one Munich apartment."""

    required_numbers = {
        "living_space": living_space,
        "no_rooms": no_rooms,
    }

    for field_name, value in required_numbers.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{field_name} must be a finite number.")

    if living_space <= 0 or living_space > 500:
        raise ValueError(
            "living_space must be greater than 0 and at most 500 square metres."
        )

    if no_rooms <= 0:
        raise ValueError("no_rooms must be greater than 0.")

    if not isinstance(neighbourhood, str) or not neighbourhood.strip():
        raise ValueError("neighbourhood must be a non-empty string.")

    optional_numbers = {
        "year_constructed": year_constructed,
        "floor": floor,
    }

    for field_name, value in optional_numbers.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{field_name} must be a finite number or None.")

    optional_text = {
        "flat_type": flat_type,
        "interior_quality": interior_quality,
        "condition": condition,
    }

    for field_name, value in optional_text.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{field_name} must be a non-empty string or None.")

    amenities = {
        "balcony": balcony,
        "garden": garden,
        "lift": lift,
        "has_kitchen": has_kitchen,
        "cellar": cellar,
    }

    for field_name, value in amenities.items():
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} must be True or False.")

    input_row = {
        "livingSpace": living_space,
        "noRooms": no_rooms,
        "yearConstructed": year_constructed,
        "floor": floor,
        "typeOfFlat": flat_type,
        "regio3": neighbourhood.strip(),
        "interiorQual": interior_quality,
        "condition": condition,
        "balcony": balcony,
        "garden": garden,
        "lift": lift,
        "hasKitchen": has_kitchen,
        "cellar": cellar,
    }

    input_df = pd.DataFrame([input_row]).replace({None: float("nan")})
    prediction = model.predict(input_df)

    return float(prediction[0])
