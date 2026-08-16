# Model Card: Munich Rent Predictor

## Overview

| Item             | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| Model            | Enhanced Ridge regression                                             |
| Configuration    | `Ridge(alpha=10.0)`                                                   |
| Task             | Predict monthly cold rent (`baseRent`)                                |
| Scope            | Apartments in Munich, Germany                                         |
| Training data    | 4,380 cleaned rental listings                                         |
| Artifact         | [`enhanced_ridge.joblib`](../models/enhanced_ridge.joblib)            |
| Live application | [Munich Rent Predictor](https://munich-rent-predictor.streamlit.app/) |

The model estimates monthly cold rent using living space, number of rooms, neighbourhood, apartment characteristics, and amenities. It is deployed through Streamlit and is also available through a CLI and the reusable `predict_rent()` function.

## Intended Use

The model is intended for:

* educational demonstrations of regression and model deployment;
* exploring patterns in historical Munich rental listings;
* producing approximate estimates for typical Munich apartments.

Predictions should be treated as rough estimates, not authoritative valuations.

## Inappropriate Use

The model should not be used:

* as a professional or legally binding property valuation;
* as a replacement for the official Munich rent index;
* to automatically set or negotiate rent;
* for legal, lending, insurance, or other high-impact decisions;
* for apartments outside Munich.

## Model and Data

The deployed scikit-learn pipeline applies numerical imputation and scaling, categorical imputation and one-hot encoding, and Ridge regression with `alpha=10.0`.

The model was trained on 4,380 cleaned historical Munich rental listings. The target is monthly cold rent before service charges and other additional housing costs.

For details, see:

* [Data source and licence](data_source.md)
* [Data selection and preprocessing](data_selection_and_preprocessing.md)
* [Training implementation](../src/rental_price_predictor/train_final_model.py)

## Evaluation

The model was evaluated on a held-out test set of 876 listings.

| Metric |  Result |
| ------ | ------: |
| MAE    | €289.13 |
| RMSE   | €456.29 |
| R²     |   0.786 |

The MAE means that predictions differed from the listed monthly cold rent by approximately €289 on average. Some unusual or expensive apartments had considerably larger errors.

Detailed results are available in:

* [Model selection and evaluation](model_selection.md)
* [Error analysis and feature experiment](error_analysis.md)

## Key Limitations and Risks

* The model learns from historical advertisements and may not represent current market conditions or completed rental contracts.
* It was trained only on Munich listings.
* Predictions are less reliable for unusual, luxury, very large, or very expensive apartments.
* Errors or biases in the original listings may be reflected in the predictions.
* Model relationships are statistical associations, not proof of causation.
* Amenity inputs are represented as `True` or `False`. The training data did not contain a meaningful unknown amenity state, so an unchecked amenity means absent—not unknown.
* Missing optional information can be imputed, but predictions based on limited information may be less apartment-specific.

Users should compare predictions with recent local listings and seek professional or legal guidance when an authoritative valuation is required.

## Related Documentation

* [Project README](../README.md)
* [Data source and licence](data_source.md)
* [Data selection and preprocessing](data_selection_and_preprocessing.md)
* [Model selection and evaluation](model_selection.md)
* [Error analysis](error_analysis.md)
* [Public application](https://munich-rent-predictor.streamlit.app/)