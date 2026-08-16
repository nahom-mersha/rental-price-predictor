# Munich Rental Price Predictor

An end-to-end machine-learning project that predicts monthly cold rent for apartments in Munich.

The project includes data cleaning, exploratory analysis, a NumPy implementation of linear regression, comparison with scikit-learn models, error analysis, feature experiments, a prediction CLI, and a publicly deployed Streamlit application.

## Live Application

Try the deployed application:

[Munich Rent Predictor](https://munich-rent-predictor.streamlit.app/)

The application accepts apartment details and estimates the monthly cold rent in euros.

## Project Objective

The objective was to build a complete regression workflow while learning both machine-learning fundamentals and practical engineering skills.

The project covers:

* cleaning and filtering real rental data;
* implementing linear regression manually with NumPy;
* building leakage-safe scikit-learn pipelines;
* comparing baseline, linear, Ridge, and decision-tree models;
* evaluating models with MAE, RMSE, and R²;
* analysing prediction errors;
* testing additional features;
* creating a reusable prediction function;
* providing a command-line interface;
* deploying an interactive Streamlit application;
* adding tests, logging, documentation, and reproducibility instructions.

## Dataset

The project uses the German apartment rental dataset described in [`docs/data_source.md`](docs/data_source.md).

The complete dataset contains:

* 268,850 raw rental listings;
* 4,383 listings located in Munich;
* 4,380 Munich listings after project-specific cleaning.

The raw dataset is not committed because it is too large.

To reproduce the project, download the dataset and place it at:

```text
data/raw/immo_data.csv
```

## Data Preparation

The data-cleaning workflow:

1. Loads the raw rental dataset.
2. Keeps listings where `regio2` is `München`.
3. Selects the model features and `baseRent` target.
4. Removes listings with a base rent below €100.
5. Removes listings with more than 500 m² of living space.
6. Removes one manually validated incorrect target value.

Missing numerical values are replaced with the training median. Missing categorical values are represented with a constant category before one-hot encoding.

## Final Model

The deployed model is a Ridge regression model with:

```python
Ridge(alpha=10.0)
```

The complete scikit-learn pipeline contains:

* median imputation for numerical features;
* numerical feature standardisation;
* constant imputation for categorical features;
* one-hot encoding for categorical features;
* Ridge regression.

The final model is trained on all 4,380 cleaned Munich listings and saved as:

```text
models/enhanced_ridge.joblib
```

## Model Features

### Numerical features

* `livingSpace`
* `noRooms`
* `yearConstructed`
* `floor`

### Categorical features

* `typeOfFlat`
* `regio3`
* `interiorQual`
* `condition`
* `balcony`
* `garden`
* `lift`
* `hasKitchen`
* `cellar`

The Streamlit application requires living space, number of rooms, and neighbourhood. The remaining apartment details are presented as optional inputs, while amenity checkboxes represent known `True` or `False` values.

## Model Performance

The enhanced Ridge model achieved the following results on the held-out test set:

| Metric |  Result |
| ------ | ------: |
| MAE    | €289.13 |
| RMSE   | €456.29 |
| R²     |   0.786 |

An MAE of approximately €289 means that the prediction differed from the real monthly cold rent by about €289 on average.

The enhanced feature set improved on the earlier Ridge model, which achieved an MAE of €308.84.

Detailed findings are available in [`docs/error_analysis.md`](docs/error_analysis.md).

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/nahom-mersha/rental-price-predictor.git
cd rental-price-predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

## Run the Streamlit Application

Make sure the trained model exists at:

```text
models/enhanced_ridge.joblib
```

Then run:

```bash
streamlit run app.py
```

Open the local address shown by Streamlit, normally:

```text
http://localhost:8501
```

## Use the Prediction CLI

The CLI is a thin wrapper around the reusable `predict_rent()` function.

Example:

```bash
python -m rental_price_predictor.cli \
  --living-space 70 \
  --no-rooms 2 \
  --neighbourhood Schwabing \
  --lift \
  --has-kitchen \
  --cellar
```

The command prints an estimated monthly cold rent.

To view all available arguments:

```bash
python -m rental_price_predictor.cli --help
```

## Train the Final Model

The raw dataset is required for retraining.

Run:

```bash
python -m rental_price_predictor.train_final_model
```

This command:

1. loads and cleans the Munich rental data;
2. builds the enhanced Ridge pipeline;
3. trains the model on all cleaned listings;
4. saves the fitted pipeline to `models/enhanced_ridge.joblib`.

The command also logs the important data-loading, cleaning, training, and saving events.

## Run the Model Experiments

To reproduce the baseline and scikit-learn model comparisons:

```bash
python -m rental_price_predictor.sklearn_models
```

The experiment compares:

* a mean prediction baseline;
* linear regression;
* Ridge regression with different regularisation strengths;
* a decision-tree regressor with grid search.

## Run the Tests

```bash
pytest
```

The tests cover:

* manual statistics calculations;
* prediction output and input-column construction;
* invalid living-space validation;
* invalid amenity-type validation.

## Code Quality

Format the project:

```bash
ruff format .
```

Run linting:

```bash
ruff check .
```

Run all configured pre-commit checks:

```bash
pre-commit run --all-files
```

## Docker

Build the Docker image:

```bash
docker build -t rental-price-predictor .
```

Run it:

```bash
docker run --rm rental-price-predictor
```

## Project Structure

```text
rental-price-predictor/
├── app.py
├── configs/
├── data/
│   ├── processed/
│   ├── raw/
│   └── sample/
├── docs/
├── models/
│   └── enhanced_ridge.joblib
├── src/
│   └── rental_price_predictor/
│       ├── cli.py
│       ├── prediction.py
│       ├── sklearn_models.py
│       └── train_final_model.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limitations

* The model was trained only on historical Munich rental listings.
* It predicts monthly cold rent rather than total housing costs.
* Its estimates are not professional property valuations.
* Predictions may be less reliable for unusual, luxury, or very expensive apartments.
* The data may contain reporting errors and may not represent the current rental market.
* The model learns statistical associations and does not establish causal relationships.
* Amenity fields in the training data contain `True` or `False`; the model does not meaningfully represent an unknown amenity state.

## Learning Notes

My concise learning notes for this project are available in my
[AI Notes repository](https://github.com/nahom-mersha/ai-notes/tree/main/Project%202%20-%20Rental%20Price%20Predictor).

## Purpose

This repository is Project 2 of my AI Engineering roadmap.

Its purpose is to develop a practical understanding of regression while connecting machine-learning theory with testing, reusable pipelines, command-line tools, deployment, and responsible model documentation.
