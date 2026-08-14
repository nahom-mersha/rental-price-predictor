# Data Selection and Preprocessing

## Purpose

This document explains how the raw German apartment-listing dataset was narrowed to the final Munich modelling dataset, why the target and features were selected, which data-quality rules were applied, and how preprocessing was fitted without leaking information from validation or test data.

It complements [`data_source.md`](data_source.md), which records the dataset origin, [`model_selection.md`](model_selection.md), which documents cross-validation and model selection, and [`error_analysis.md`](error_analysis.md), which documents the later target-error investigation.

## Prediction problem

Each row represents one advertised rental property. The system receives property information and predicts:

```text
baseRent = monthly cold rent in euros
```

Cold rent was chosen instead of total rent because it represents the property's advertised base price before service charges and heating costs. It gives the model a clearer and more consistent target than a total that may combine different additional charges.

This is a supervised regression problem because the target is a continuous numerical value and the training data contains examples with known `baseRent` values.

## Raw data source

The project uses the public Kaggle dataset **Apartment Rental Offers in Germany**, collected from ImmoScout24 listings.

| Property | Value |
|---|---:|
| Raw rows | 268,850 |
| Raw columns | 49 |
| Geographic coverage | Germany |
| Local raw path | `data/raw/immo_data.csv` |

The CSV is not committed to Git because of its size. The source and download location are documented in [`data_source.md`](data_source.md).

## Geographic selection

The project focuses on Munich city. Records were selected using the exact `regio2` label:

```python
munich_df = raw_df.loc[raw_df["regio2"].eq("München")].copy()
```

This produced:

| Selection stage | Rows |
|---|---:|
| Full German dataset | 268,850 |
| Munich city | 4,383 |

The Munich subset contained 42 `regio3` neighbourhood labels.

Restricting the project to one city makes the prediction task more coherent because rental markets differ greatly across Germany. It also allows neighbourhood to represent meaningful local price variation. The trade-off is that the resulting model must not be presented as a Germany-wide predictor.

## Feature selection

### Initial NumPy learning model

The from-scratch linear-regression work used two numerical features:

| Feature | Meaning | Reason for inclusion |
|---|---|---|
| `livingSpace` | Apartment area in square metres | Size is expected to be one of the strongest rent predictors. |
| `noRooms` | Number of rooms | Captures layout and capacity beyond area alone. |

This deliberately small input set made it practical to learn the prediction equation, MSE, vectorized gradients, gradient descent, scaling, and closed-form solutions.

### Professional scikit-learn models

The professional pipelines retained the two numerical features and added two categorical features:

| Feature | Type | Reason for inclusion |
|---|---|---|
| `livingSpace` | Numerical | Measures property size. |
| `noRooms` | Numerical | Adds layout information conditional on size. |
| `typeOfFlat` | Categorical | Distinguishes apartment types such as loft, penthouse, or ground-floor flat. |
| `regio3` | Categorical | Represents Munich neighbourhood and broad location effects. |

The professional model table therefore used:

```python
NUMERICAL_FEATURES = [
    "livingSpace",
    "noRooms",
]

CATEGORICAL_FEATURES = [
    "typeOfFlat",
    "regio3",
]

TARGET = "baseRent"
```

These features are understandable, available as property inputs, and usable by both linear and tree-based models.

## Leakage exclusions

The following rent-related fields were not used as predictors:

- `totalRent`;
- `serviceCharge`;
- `heatingCosts`.

They are components of, or are directly combined with, the advertised rent. Using them would make the task unrealistic and could leak information too closely related to the target. For example:

```text
total rent ≈ base rent + service charge + heating costs
```

A deployed predictor should estimate cold rent from property characteristics rather than require another rent figure that nearly reveals the answer.

The target `baseRent` was separated from `X` before model fitting and used only as `y`.

## Initial data-quality inspection

Before cleaning, the Munich subset contained several suspicious extremes:

| Column | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| `baseRent` | €1,767.75 | €1.00 | €20,100.00 |
| `livingSpace` | 94.70 m² | — | 66,100 m² |
| `noRooms` | 2.531 | — | — |

The €1 minimum rent and 66,100 m² living-space maximum were not plausible Munich apartment listings and motivated narrow validation rules.

No missing values were found in the initial core columns `livingSpace`, `noRooms`, and `baseRent`. No exact duplicates remained in the selected modelling table.

Several candidate descriptive features had substantial missingness:

| Column | Missing values | Missing percentage |
|---|---:|---:|
| `interiorQual` | 1,584 | 36.1% |
| `condition` | 1,206 | 27.5% |
| `typeOfFlat` | 847 | 19.3% |
| `yearConstructed` | 751 | 17.1% |

Missingness was not treated as an automatic reason to discard every affected row. `typeOfFlat` was retained and handled by the categorical preprocessing pipeline. Other property-quality features were deferred from the first professional model so that the initial comparison remained focused and explainable.

## Cleaning decisions

### Stage 1: obvious validity rules

Two narrow rules were applied to the Munich modelling table:

```python
model_df = model_df.loc[
    (model_df["baseRent"] >= 100) & (model_df["livingSpace"] <= 500)
].copy()
```

These rules removed two rows:

| Stage | Rows remaining | Rows removed |
|---|---:|---:|
| Munich subset | 4,383 | — |
| After initial validity rules | 4,381 | 2 |

The lower target threshold removed an implausible base rent below €100. The area threshold removed an implausibly large apartment record. The project did not apply a general upper-rent threshold because unusually expensive apartments can be legitimate and contain important market information.

### Stage 2: validated target error discovered during error analysis

Later error analysis identified index `213625` as uniquely inconsistent:

| Field | Value |
|---|---:|
| `baseRent` | €20,100 |
| `serviceCharge` | €140 |
| `heatingCosts` | €150 |
| `totalRent` | €2,390 |

The other rent components implied:

```text
implied base rent = 2,390 - 140 - 150 = €2,100
```

The exact €18,000 discrepancy strongly supported an extra-zero target error. The row was excluded rather than silently corrected because the project did not have an authoritative replacement value.

```python
# Exclude one validated target-data error:
# baseRent is €20,100, but totalRent - serviceCharge - heatingCosts
# implies a likely base rent of €2,100.
model_df = model_df.drop(index=213625)
```

Other high-rent rows were kept because they were internally consistent. This preserved legitimate information about expensive apartments.

### Final row counts

| Processing stage | Rows |
|---|---:|
| Raw German dataset | 268,850 |
| Munich city subset | 4,383 |
| After initial validity rules | 4,381 |
| After validated target exclusion | 4,380 |

The row `213625` issue was discovered by inspecting an original test error. Results obtained after its exclusion are therefore described as a **data-quality-corrected re-evaluation**, not as an evaluation on a completely untouched test set.

## Feature and target separation

After selection and cleaning, inputs and target were separated explicitly:

```python
X = model_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

y = model_df[TARGET]
```

This produces a feature table `X` and a one-dimensional target series `y`. The target must never be included in preprocessing features because that would give the model direct access to the answer.

## Reproducible train/test split

The final modelling data was split with:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

| Partition | Rows |
|---|---:|
| Training | 3,504 |
| Held-out test | 876 |

`random_state=42` makes the shuffled split reproducible. Approximately 80% of the data is available for fitting and cross-validation, while 20% remains held out for final evaluation.

## Why splitting happens before learned preprocessing

Imputation values, scaling statistics, and category mappings are learned from data. If they were fitted using the entire dataset, information from test rows would influence the training representation.

The correct flow is:

```text
clean rows
→ split into training and test data
→ fit preprocessing on training data only
→ apply learned preprocessing to training and test data
→ fit model on transformed training data
→ evaluate on transformed test data
```

For cross-validation, the same principle applies within every fold. The pipeline fits preprocessing on the four training folds and applies it to the remaining validation fold. It does not fit the transformer once using all five folds.

## NumPy-model preprocessing

The learning-focused NumPy implementation standardized the numerical inputs manually.

For each feature:

```text
scaled value = (original value - training mean) / training standard deviation
```

The means and standard deviations were calculated from `X_train` only. The same training values were then applied to `X_test`. Therefore, the scaled test features were not expected to have an exact mean of zero or standard deviation of one.

An intercept column of ones was added after scaling:

```python
X_train_design = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
```

This allowed the model to learn a non-zero bias term. With centered numerical features, the learned intercept was close to the mean training rent.

The target `y` was not standardized. Keeping it in euros made predictions, residuals, MSE, MAE, and RMSE directly interpretable. Scaling the target is possible, but it was unnecessary for this project and would require inverse transformation before reporting predictions.

## Professional-pipeline preprocessing

The scikit-learn models use a `ColumnTransformer` containing separate numerical and categorical pipelines.

### Numerical pipeline

```text
livingSpace, noRooms
→ median imputation
→ standardization with StandardScaler
```

The median imputer makes the pipeline robust to future missing numerical inputs even though the current core numerical columns were complete. `StandardScaler` learns each training fold's mean and standard deviation.

Scaling is especially important for Ridge because L2 regularization penalizes coefficient magnitudes. Without comparable feature scales, the penalty would affect features unevenly.

### Categorical pipeline

```text
typeOfFlat, regio3
→ replace missing values with "missing"
→ one-hot encoding
→ drop the first category
```

The configuration uses:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
)
```

One-hot encoding converts each category into numerical indicator columns. Dropping the first category avoids a redundant dummy column for linear models. `handle_unknown="ignore"` prevents prediction from failing when a valid future input contains a category that was absent from a particular training fold; that unseen category is represented by zeros across the learned indicator columns.

### Combined pipeline

The preprocessor and estimator are joined into one model pipeline:

```text
raw feature DataFrame
→ ColumnTransformer
→ Linear Regression, Ridge, or Decision Tree
→ predicted base rent
```

Using the same preprocessing construction for every professional model makes their comparison fair and allows raw input rows to be passed directly to `.fit()` and `.predict()`.

## Data-quality and preprocessing decisions summary

| Decision | Reason |
|---|---|
| Predict `baseRent` | Provides a clear monthly cold-rent target. |
| Limit data to Munich | Creates a coherent local rental market and enables neighbourhood analysis. |
| Keep legitimate expensive listings | Unusual values are not automatically errors and contain useful information. |
| Remove `baseRent < €100` | Filters implausible target values. |
| Remove `livingSpace > 500 m²` | Filters an obvious area error while retaining large luxury apartments below the threshold. |
| Exclude index `213625` | Independent rent-field evidence validates a target-data error. |
| Exclude rent-component predictors | Prevents target leakage and unrealistic prediction requirements. |
| Split before learned preprocessing | Keeps validation and test information out of training transformations. |
| Fit preprocessing inside pipelines | Prevents leakage inside cross-validation folds and keeps inference reproducible. |
| Median-impute numerical inputs | Provides a robust numerical fallback. |
| Encode missing categories explicitly | Preserves rows while allowing missingness to be represented. |
| Standardize numerical features | Supports gradient descent and fair Ridge regularization. |
| Keep `y` in euros | Preserves direct metric and prediction interpretation. |

## Reproducibility

The professional data loading and preprocessing are implemented in:

```text
src/rental_price_predictor/sklearn_models.py
```

The expected finalized data-loading sequence is:

```python
raw_df = pd.read_csv("data/raw/immo_data.csv")

munich_df = raw_df.loc[raw_df["regio2"].eq("München")].copy()

model_df = munich_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

model_df = model_df.loc[
    (model_df["baseRent"] >= 100) & (model_df["livingSpace"] <= 500)
].copy()

model_df = model_df.drop(index=213625)
```

From the project root, the professional experiment can be reproduced with:

```bash
python -m rental_price_predictor.sklearn_models
```

## Limitations and follow-up work

1. The model is restricted to Munich and should not be used as a Germany-wide rent estimator.
2. The dataset contains historical advertised rents rather than confirmed transaction prices.
3. The index-based exclusion of `213625` depends on the original CSV retaining the same index. A future production data-validation rule should identify inconsistencies from field values rather than a fixed row index.
4. Several useful quality and amenity features have substantial missingness and were not included in the initial model.
5. One-hot encoding can produce many sparse columns, and rare categories may have unstable learned effects.
6. The target correction was discovered through original test-set inspection, so the corrected evaluation must retain its methodological caveat.
7. The dataset licence still needs to be stated explicitly in `data_source.md`; it should be verified from the authoritative Kaggle source rather than inferred.

The next bounded modelling experiment will test selected property-quality, age, refurbishment, amenity, and floor features using the same leakage-safe cross-validation protocol. Its outcome must be compared against the current Ridge baseline before changing the selected model.

## Conclusion

The final dataset-preparation process reduced 268,850 German listings to 4,380 cleaned Munich records using transparent geographic selection and three narrowly justified removals. Feature selection kept the initial task understandable while adding property type and neighbourhood for the professional models.

All learned preprocessing is fitted only from training data, including inside cross-validation folds. This separation is essential: a reproducible model is not only an estimator, but the complete chain of cleaning, feature selection, imputation, scaling, encoding, fitting, and prediction.
