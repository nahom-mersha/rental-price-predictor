# Model Selection, Cross-Validation, and Hyperparameter Tuning

## Purpose

This document explains how the professional scikit-learn regression models were compared and how Ridge Regression with `alpha=10.0` was selected. It records the experimental design, leakage-safe preprocessing, five-fold cross-validation, Ridge regularization search, Decision Tree grid search, final held-out results, and the reasoning behind the final model choice.

The separate NumPy linear-regression implementation was primarily a learning exercise and is not treated as a competing production pipeline in this report.

## Prediction task

- **Target:** `baseRent`, the monthly cold rent in euros
- **Numerical features:** `livingSpace`, `noRooms`
- **Categorical features:** `typeOfFlat`, `regio3`
- **Location scope:** Munich city listings
- **Random seed:** `42`
- **Train/test split:** 80% training, 20% held-out test
- **Cross-validation:** five folds on the training data
- **Primary selection metric:** mean absolute error (MAE)

After the finalized cleaning decisions, the dataset split contained:

| Partition | Listings |
|---|---:|
| Training | 3,504 |
| Held-out test | 876 |

The test set was kept outside cross-validation and hyperparameter selection so that it could provide a final estimate on data that did not guide the model choice.

## Leakage-safe preprocessing

Every professional model was placed after the same `ColumnTransformer` inside a scikit-learn `Pipeline`. This ensured that preprocessing was learned independently inside each cross-validation training fold rather than once from all training rows.

### Numerical preprocessing

1. Fill missing values using the median learned from the current training fold.
2. Standardize the numerical features using the fold's learned mean and standard deviation.

Standardization is especially important for Ridge because its L2 penalty acts directly on coefficient magnitudes. Features measured on very different scales would otherwise receive uneven regularization.

### Categorical preprocessing

1. Replace missing categorical values with the label `missing`.
2. One-hot encode the categories.
3. Drop the first encoded level to avoid redundant dummy columns.
4. Ignore previously unseen categories at prediction time instead of failing.

### Why preprocessing belongs inside the pipeline

If the imputer, scaler, or encoder were fitted before cross-validation, information from each validation fold could influence the transformation applied to its training fold. Keeping preprocessing inside the pipeline prevents this leakage and makes every model comparison follow the same reproducible workflow.

## Evaluation metrics

### Mean absolute error

MAE was the primary model-selection metric:

```text
MAE = mean(abs(actual rent - predicted rent))
```

It is easy to interpret in euros and answers the practical question: how far away is the prediction on average?

### Root mean squared error

RMSE squares errors before averaging and therefore penalizes large mistakes more heavily than MAE. It was reported on the held-out test set as a secondary metric, particularly because expensive outliers can affect rental predictions strongly.

### R²

R² measures how much target variation the model explains relative to predicting the test-set mean. A value closer to one is better. A value around zero indicates little improvement over a constant prediction, and a negative value indicates worse performance than that reference.

## Mean baseline

The baseline predicted the mean training-set rent for every test listing. It did not use any input features.

| Metric | Result |
|---|---:|
| Test MAE | €685.86 |
| Test RMSE | €988.72 |
| Test R² | -0.003 |

The slightly negative R² indicates that the constant mean prediction did not explain the held-out variation. The baseline establishes the minimum standard that a useful learned model should exceed.

## Why cross-validation was used

A single validation split can make a model appear unusually good or bad depending on which listings happen to enter it. Five-fold cross-validation provides a more stable training-set comparison:

1. Split the training data into five folds.
2. Train on four folds and validate on the remaining fold.
3. Repeat until each fold has served as validation data once.
4. Average the five validation MAE values.

The test set was not one of these folds. It remained held out while algorithms and hyperparameters were compared.

Scikit-learn's scoring name was `neg_mean_absolute_error`. Scikit-learn represents losses as negative scores so that its general rule of “larger score is better” still works. The implementation multiplied the returned values by `-1` to report ordinary positive MAE values.

## Ordinary Linear Regression

Linear Regression provided an unregularized professional reference using the complete preprocessing pipeline.

| Evaluation | Result |
|---|---:|
| Mean CV MAE | €312.99 |
| Test MAE | €310.23 |
| Test RMSE | €478.98 |
| Test R² | 0.765 |

Its cross-validation and test MAE values were close, which did not suggest a large train-to-test generalization gap.

## Ridge Regression tuning

Ridge uses the same linear prediction structure as ordinary Linear Regression but adds an L2 penalty to the training objective:

```text
ordinary squared-error loss + alpha * sum(coefficient²)
```

The penalty discourages unnecessarily large coefficients. `alpha` controls its strength:

- a very small `alpha` behaves similarly to ordinary Linear Regression;
- a moderate `alpha` can reduce variance and stabilize correlated coefficients;
- an excessively large `alpha` can shrink useful effects too much and cause underfitting.

### Candidate alphas

Five values were defined before selecting the final Ridge model:

| Alpha | Mean five-fold CV MAE |
|---:|---:|
| 0.01 | €312.99 |
| 0.1 | €312.97 |
| 1.0 | €312.71 |
| **10.0** | **€311.09** |
| 100.0 | €318.16 |

`alpha=10.0` achieved the lowest mean cross-validation MAE in the tested grid. The gradual improvement from `0.01` to `10.0` suggests that modest coefficient shrinkage helped. Performance worsened at `100.0`, indicating that this stronger penalty removed too much useful signal.

This result does not prove that `10.0` is the globally optimal continuous value. It establishes only that it was the best of the predefined candidates.

### Final Ridge result

After selecting `alpha=10.0` using training-set cross-validation, the Ridge pipeline was fitted to the complete training set and evaluated on the held-out test set.

| Evaluation | Result |
|---|---:|
| Mean CV MAE | €311.09 |
| Test MAE | €308.84 |
| Test RMSE | €479.18 |
| Test R² | 0.764 |

## Decision Tree tuning

The Decision Tree provided a nonlinear comparison. Unlike the linear models, it can divide the feature space into regions and represent interactions without explicitly defining them. An unrestricted tree can also overfit, so structural hyperparameters were tuned using `GridSearchCV`.

### Parameter grid

| Hyperparameter | Candidate values | Purpose |
|---|---|---|
| `max_depth` | `5`, `10`, `15`, `None` | Limits how many levels the tree can grow. |
| `min_samples_leaf` | `5`, `10`, `20` | Requires a minimum number of training samples in every leaf. |
| `min_samples_split` | `2`, `20`, `50` | Requires enough samples before a node may split. |

The grid contained:

```text
4 × 3 × 3 = 36 hyperparameter combinations
```

With five-fold cross-validation, this required 180 fold-level model fits. `GridSearchCV` selected the candidate with the best mean validation score and, by default, refitted that candidate on the complete training set.

### Selected tree

```python
{
    "model__max_depth": None,
    "model__min_samples_leaf": 5,
    "model__min_samples_split": 50,
}
```

| Evaluation | Result |
|---|---:|
| Best mean CV MAE | €330.79 |
| Test MAE | €318.90 |
| Test RMSE | €495.94 |
| Test R² | 0.748 |

Even though `max_depth=None` allowed unrestricted depth, the minimum split and leaf requirements constrained the tree. The tuned tree still performed worse than both linear models on cross-validation and held-out evaluation.

## Final comparison

| Model | Mean CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Mean baseline | — | €685.86 | €988.72 | -0.003 |
| Linear Regression | €312.99 | €310.23 | **€478.98** | **0.765** |
| **Ridge (`alpha=10.0`)** | **€311.09** | **€308.84** | €479.18 | 0.764 |
| Decision Tree | €330.79 | €318.90 | €495.94 | 0.748 |

All three learned models substantially outperformed the mean baseline. The two linear models were almost tied and both outperformed the tuned Decision Tree.

## Why Ridge with alpha 10 was selected

Ridge with `alpha=10.0` was selected as the main model for four reasons:

1. **It had the best mean cross-validation MAE.** Its €311.09 result was €1.90 better than ordinary Linear Regression and €19.70 better than the tuned Decision Tree.
2. **It had the best held-out MAE.** Its €308.84 test MAE was €1.39 better than Linear Regression and €10.06 better than the Decision Tree.
3. **MAE was the predefined primary metric.** Model selection therefore followed the metric chosen before the final comparison rather than switching to whichever metric favoured a different model afterward.
4. **L2 regularization provides a modest stability benefit.** The numerical and one-hot encoded predictors can overlap or correlate. Ridge gently shrinks their coefficients and offers some protection against unstable, overly large weights.

Ordinary Linear Regression had a marginally better RMSE by €0.20 and a marginally better R² by 0.001. These differences are negligible and do not justify overriding the primary-metric decision. The honest conclusion is that Linear Regression and Ridge performed nearly identically; Ridge was chosen by a small, consistent MAE advantage and its regularization benefit, not because it dominated every metric.

## Why a final fit was needed

During each cross-validation round, a model trained on only four of the five training folds. After the algorithm and hyperparameter were chosen, the selected pipeline was fitted once using **all** available training rows. This allowed the final model to learn from the complete training partition.

The held-out test set was then used for final evaluation. It was not used to select `alpha`, choose the tree parameters, or decide which algorithm looked best during cross-validation.

The Decision Tree was handled slightly differently in code because `GridSearchCV(refit=True)` refitted its best pipeline on the full training set automatically. It therefore did not need another explicit `.fit()` before test prediction.

## Data-quality correction caveat

Later error analysis found a validated target-data error at index `213625`: the recorded `baseRent` was €20,100, while `totalRent - serviceCharge - heatingCosts` implied a likely base rent of €2,100. The row was excluded before splitting, and the reported 3,504/876 split and final comparison metrics in this document are the resulting data-quality-corrected re-evaluation.

Because the problem was discovered by inspecting errors from the original held-out test set, this re-evaluation is not equivalent to evaluation on a completely fresh test set. The original finding and its evidence are documented separately in [`error_analysis.md`](error_analysis.md).

## Reproducibility

The experiment is implemented in:

```text
src/rental_price_predictor/sklearn_models.py
```

From the project root, run:

```bash
python -m rental_price_predictor.sklearn_models
```

The raw CSV is intentionally not committed. It must be available at:

```text
data/raw/immo_data.csv
```

For results to match this document, the data-loading function must include the finalized exclusion of index `213625` before `train_test_split()`.

## Conclusion

The comparison followed a reproducible selection protocol: shared leakage-safe preprocessing, five-fold training-set cross-validation, predefined hyperparameter candidates, a primary metric in euros, and final held-out evaluation. Ridge Regression with `alpha=10.0` was selected because it achieved the lowest cross-validation and test MAE while retaining the interpretability of a linear model and adding L2 regularization.

The performance difference from ordinary Linear Regression was small, so the decision should be presented as a reasoned tie-break rather than a dramatic superiority claim. The tuned Decision Tree did not improve performance for the current dataset and feature representation.
