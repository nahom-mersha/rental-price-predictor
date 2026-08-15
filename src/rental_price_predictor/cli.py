import argparse

import joblib

from rental_price_predictor.prediction import predict_rent
from rental_price_predictor.train_final_model import MODEL_PATH


def main() -> None:
    """Run a rental-price prediction from terminal arguments."""
    parser = argparse.ArgumentParser(
        description="Predict monthly cold rent for a Munich apartment."
    )

    parser.add_argument(
        "--living-space",
        type=float,
        required=True,
        help="Apartment living space in square metres.",
    )
    parser.add_argument(
        "--no-rooms",
        type=float,
        required=True,
        help="Number of rooms, for example 2.5.",
    )
    parser.add_argument(
        "--neighbourhood",
        required=True,
        help="Munich neighbourhood, for example Schwabing.",
    )

    parser.add_argument(
        "--flat-type",
        help="Optional apartment type.",
    )
    parser.add_argument(
        "--year-constructed",
        type=float,
        help="Optional construction year.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        help="Optional floor number.",
    )
    parser.add_argument(
        "--interior-quality",
        help="Optional interior-quality category.",
    )
    parser.add_argument(
        "--condition",
        help="Optional property-condition category.",
    )

    parser.add_argument(
        "--balcony",
        action="store_true",
        help="Include if the apartment has a balcony.",
    )
    parser.add_argument(
        "--garden",
        action="store_true",
        help="Include if the apartment has a garden.",
    )
    parser.add_argument(
        "--lift",
        action="store_true",
        help="Include if the building has a lift.",
    )
    parser.add_argument(
        "--has-kitchen",
        action="store_true",
        help="Include if the apartment has a kitchen.",
    )
    parser.add_argument(
        "--cellar",
        action="store_true",
        help="Include if the apartment has a cellar.",
    )

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        parser.error(
            f"Trained model not found at: {MODEL_PATH}. Run train_final_model first."
        )

    model = joblib.load(MODEL_PATH)

    try:
        prediction = predict_rent(
            model,
            living_space=args.living_space,
            no_rooms=args.no_rooms,
            neighbourhood=args.neighbourhood,
            balcony=args.balcony,
            garden=args.garden,
            lift=args.lift,
            has_kitchen=args.has_kitchen,
            cellar=args.cellar,
            flat_type=args.flat_type,
            year_constructed=args.year_constructed,
            floor=args.floor,
            interior_quality=args.interior_quality,
            condition=args.condition,
        )
    except ValueError as error:
        parser.error(str(error))

    print(f"Predicted monthly cold rent: €{prediction:,.2f}")


if __name__ == "__main__":
    main()
