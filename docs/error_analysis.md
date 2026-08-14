# Ridge Regression Error Analysis

## Purpose

This document records the error analysis performed for the selected rental-price model: Ridge Regression with `alpha=10.0`. It documents the model's finalized results before the planned feature-expansion experiment, the investigation and exclusion of one validated target-data error, performance by neighbourhood, learned feature effects, the residual-plot findings, and the model's current limitations.

All numerical results below come from locally executed project outputs collected.

## Model and evaluation setup

- **Prediction target:** `baseRent` (monthly cold rent in euros)
- **Numerical features:** `livingSpace`, `noRooms`
- **Categorical features:** `typeOfFlat`, `regio3`
- **Selected model:** Ridge Regression with `alpha=10.0`
- **Split:** 80% training and 20% held-out test data
- **Random seed:** `42`
- **Model selection:** five-fold cross-validation on the training set
- **Residual definition:**

  ```text
  residual = actual rent - predicted rent
  ```

  A positive residual means that the model underpredicted the rent. A negative residual means that it overpredicted the rent.

After the finalized cleaning rules were applied, the split contained 3,504 training listings and 876 test listings.

## Finalized Ridge results

These results describe the selected Ridge model after excluding the validated bad target at index `213625` and rerunning the split and evaluation.

| Evaluation | Result |
|---|---:|
| Mean cross-validation MAE | €311.09 |
| Held-out test MAE | €308.84 |
| Held-out test RMSE | €479.18 |
| Held-out test R² | 0.764 |

The test MAE means that the model's prediction was approximately €309 away from the recorded monthly base rent on average. The RMSE is higher because it gives greater weight to the remaining large errors. An R² of 0.764 means that the model explained approximately 76.4% of the variation in the held-out target values.

These are **data-quality-corrected re-evaluation results**. The target error at index `213625` was discovered while inspecting the original held-out errors, so these numbers must not be described as results from a completely untouched test set.

## Validated target-data error at index 213625

### Initial observation

Before the additional cleaning decision, index `213625` was the model's largest error:

| Field | Value |
|---|---:|
| Neighbourhood | Lehel |
| Living space | 110 m² |
| Rooms | 3 |
| Recorded `baseRent` | €20,100.00 |
| Ridge prediction | €2,827.37 |
| Residual | +€17,272.63 |

An unusual target value is not automatically incorrect. The row was therefore investigated using other rent fields in the original record.

### Internal consistency check

| Rent field | Recorded value |
|---|---:|
| `baseRent` | €20,100.00 |
| `serviceCharge` | €140.00 |
| `heatingCosts` | €150.00 |
| `totalRent` | €2,390.00 |

Using the recorded components produced:

```text
calculated total = baseRent + serviceCharge + heatingCosts
                 = 20,100 + 140 + 150
                 = €20,390

recorded total difference = totalRent - calculated total
                          = 2,390 - 20,390
                          = -€18,000
```

The base rent implied by the other three fields was:

```text
implied base rent = totalRent - serviceCharge - heatingCosts
                  = 2,390 - 140 - 150
                  = €2,100
```

The exact €18,000 difference strongly supports an extra-zero error in the recorded target: `€20,100` instead of a likely `€2,100`.

### Why the decision was narrow

The project did not remove all expensive listings. Among the complete Munich records with `baseRent >= €5,000`, the other inspected listings were internally consistent:

| Index | Recorded base rent | Implied base rent | Difference |
|---:|---:|---:|---:|
| 104969 | €5,980 | €5,980 | €0 |
| 164826 | €5,402 | €5,402 | €0 |
| 187090 | €6,000 | €6,000 | €0 |
| 215352 | €5,430 | €5,430 | €0 |

Another suspicious record, index `199459`, contained `baseRent=€995`, `serviceCharge=€102`, `heatingCosts=€65`, and `totalRent=€1`. In that case, the target looked plausible and the supporting `totalRent` field was evidently unreliable. This demonstrated that a disagreement between rent fields does not automatically prove that `baseRent` is wrong.

### Cleaning decision

Index `213625` was excluded before `train_test_split()` rather than silently corrected to €2,100. Exclusion was the more conservative and transparent decision because the original target was demonstrably inconsistent, but the project did not have authoritative access to the intended replacement value.

Cleaning before the split also ensured that the invalid target could enter neither training nor test data. Removing it only from the test set would not protect future splits from training on the bad target.

## Largest errors after the exclusion

After excluding index `213625`, the largest absolute test error fell from approximately €17,273 to approximately €5,645.

| Living space | Rooms | Neighbourhood | Actual rent | Predicted rent | Residual | Absolute error |
|---:|---:|---|---:|---:|---:|---:|
| 236.60 m² | 5.5 | Bogenhausen | €10,700.00 | €5,054.65 | +€5,645.35 | €5,645.35 |
| 143.00 m² | 4.0 | Lehel | €6,800.00 | €3,453.41 | +€3,346.59 | €3,346.59 |
| 245.00 m² | 5.0 | Haidhausen | €8,500.00 | €5,805.19 | +€2,694.81 | €2,694.81 |
| 148.00 m² | 4.0 | Altstadt | €5,950.00 | €3,546.87 | +€2,403.13 | €2,403.13 |
| 115.13 m² | 3.0 | Altstadt | €4,974.00 | €2,900.71 | +€2,073.29 | €2,073.29 |
| 213.90 m² | 6.0 | Bogenhausen | €6,417.00 | €4,442.41 | +€1,974.59 | €1,974.59 |
| 100.00 m² | 3.0 | Maxvorstadt | €4,690.00 | €2,838.78 | +€1,851.22 | €1,851.22 |
| 237.00 m² | 3.0 | Altstadt | €4,000.00 | €5,749.99 | -€1,749.99 | €1,749.99 |
| 25.00 m² | 1.0 | Pasing | €1,999.00 | €349.84 | +€1,649.16 | €1,649.16 |
| 110.00 m² | 4.0 | Schwabing | €4,500.00 | €2,894.14 | +€1,605.86 | €1,605.86 |

Most of these extreme cases were underpredictions of expensive or unusual apartments. However, the Altstadt listing with 237 m² was overpredicted by approximately €1,750, showing that the model was not simply biased downward in every expensive case.

## Error by neighbourhood

Neighbourhood MAE was calculated only for neighbourhoods containing at least ten held-out listings. This threshold reduces, but does not eliminate, instability from very small groups.

| Neighbourhood | Test listings | MAE |
|---|---:|---:|
| Altstadt | 21 | €613.44 |
| Lehel | 11 | €604.98 |
| Nymphenburg | 14 | €415.40 |
| Maxvorstadt | 54 | €390.53 |
| Bogenhausen | 84 | €388.30 |
| Haidhausen | 40 | €355.62 |
| Ramersdorf | 15 | €341.46 |
| Sendling | 12 | €330.30 |
| Harlaching | 11 | €324.76 |
| Neuhausen | 49 | €317.02 |
| Ludwigsvorstadt_Isarvorstadt | 44 | €312.08 |
| Solln | 28 | €296.63 |
| Schwabing_West | 48 | €296.48 |
| Perlach | 53 | €284.19 |
| Schwabing | 50 | €279.98 |
| Aubing | 20 | €279.85 |
| Obermenzing | 28 | €279.08 |
| Pasing | 23 | €267.78 |
| Hadern | 13 | €262.21 |
| Trudering | 33 | €256.48 |
| Berg_am_Laim | 15 | €256.40 |
| Sendling_Westpark | 15 | €248.94 |
| Milbertshofen | 20 | €246.31 |
| Obersendling | 27 | €223.12 |
| Laim | 19 | €212.97 |
| Obergiesing | 26 | €209.45 |
| Untermenzing | 11 | €205.31 |
| Schwanthalerhöhe | 12 | €198.41 |
| Moosach | 19 | €196.59 |

The overall Ridge test MAE was €308.84, but error varied considerably across neighbourhoods. Altstadt and Lehel both had MAE values around €605–€613, approximately twice the overall test MAE. Premium central neighbourhoods may contain more heterogeneous or luxury properties whose price depends on information absent from the current features.

Sample size must remain part of the interpretation. Lehel's result came from only 11 test listings and is therefore less stable than Bogenhausen's result based on 84 listings. The neighbourhood results describe this specific held-out split and must not be treated as permanent rankings of model quality.

## Ridge feature effects

Ridge coefficients are learned weights. Positive coefficients push the prediction upward; negative coefficients push it downward, holding the other transformed features fixed. Ridge applies L2 regularization, which shrinks the weights toward zero.

### Numerical features

| Feature | Coefficient | Interpretation |
|---|---:|---|
| `livingSpace` | +€975.13 | An increase of one training-set standard deviation in living space raises the prediction by approximately €975, holding the other features fixed. It does **not** mean one additional square metre adds €975 because the feature was standardized. |
| `noRooms` | -€133.44 | For listings with the same living space and other inputs, an increase of one standardized room-count unit lowers the prediction by approximately €133. This conditional effect may reflect the overlap between room count and living space; it is not a general claim that more rooms make apartments cheaper. |

### Largest positive categorical coefficients

| Feature | Coefficient |
|---|---:|
| `regio3_Lehel` | +€534.23 |
| `regio3_Altstadt` | +€507.60 |
| `regio3_Maxvorstadt` | +€402.06 |
| Missing `typeOfFlat` | +€394.02 |
| `regio3_Schwabing` | +€345.20 |
| `regio3_Ludwigsvorstadt_Isarvorstadt` | +€302.95 |
| `regio3_Haidhausen` | +€249.10 |
| `regio3_Schwabing_West` | +€219.38 |

### Largest negative categorical coefficients

| Feature | Coefficient |
|---|---:|
| `regio3_Solln` | -€282.42 |
| `regio3_Fürstenried` | -€211.03 |
| `regio3_Trudering` | -€195.24 |
| `regio3_Pasing` | -€194.54 |
| `regio3_Hadern` | -€194.44 |
| `regio3_Aubing` | -€190.08 |
| `regio3_Untermenzing` | -€175.55 |
| `regio3_Perlach` | -€164.85 |

The positive coefficients for Lehel, Altstadt, Maxvorstadt, and Schwabing are consistent with higher predicted rents in premium central areas after controlling for the model's other inputs. Negative coefficients act in the opposite direction within the fitted categorical encoding.

The positive coefficient for a missing `typeOfFlat` value does not mean missing data causes higher rent. It only records an association present in this training dataset. More generally, coefficients are conditional associations, not causal effects, and correlated features can redistribute weight among one another.

## Residual plot

The residual plot used predicted monthly rent on the horizontal axis and `actual - predicted` residual on the vertical axis. A horizontal red line at zero represented a perfect prediction.

For the dense range of ordinary predictions, approximately €500–€2,500, residuals were relatively concentrated around zero. As predicted rent increased, the residuals became much more widely dispersed. The plot contained both large positive residuals and negative residuals among expensive predictions.

This widening spread is evidence of **heteroscedasticity**: the error variance is not constant across the prediction range. The model is less reliable for high-rent listings. The largest remaining positive point corresponded to the Bogenhausen apartment predicted at approximately €5,055 but recorded at €10,700.

The plot supports the conclusion that premium and unusual listings depend on characteristics the current model does not fully observe. It does not show a simple pattern of always underpredicting expensive apartments, because some expensive predictions were overestimates.

## Current limitations

1. **Limited property detail.** The model uses only living space, room count, flat type, and neighbourhood. It does not yet model condition, interior quality, refurbishment, building age, floor, amenities, or energy characteristics.
2. **Reduced reliability for premium listings.** The residual variance increases at higher predicted rents, and several of the largest errors occur in premium neighbourhoods or unusual properties.
3. **Linear and additive structure.** Ridge models additive linear effects after preprocessing. It may miss interactions and nonlinear relationships, such as neighbourhood changing the value of additional living space.
4. **Categorical and missing-data interpretation.** One-hot coefficients describe associations in the fitted encoding. They do not establish causal effects, and missingness can act as a proxy for unobserved listing characteristics.
5. **Unequal category sample sizes.** Neighbourhood MAE estimates based on small test groups are unstable. The minimum of ten listings is only a pragmatic reporting threshold.
6. **Outlier sensitivity remains.** The validated bad target was removed, but legitimate luxury listings still produce large residuals and strongly affect RMSE.
7. **Test-set reuse after data investigation.** Row `213625` was discovered through inspection of the original test errors. The corrected evaluation is therefore transparent and useful for data-quality analysis, but it is not equivalent to evaluation on a completely fresh, untouched test set.
8. **Scope and generalization.** Results apply to the Munich listings represented in this dataset. They should not be assumed to transfer directly to another city, time period, or changing rental market.
9. **Listing prices are not final transaction prices.** The target represents recorded listing data and may include further inconsistencies not detected by the current rules.

## Planned model-improvement experiment

The error analysis suggests a bounded follow-up experiment using additional features that are available at prediction time and may explain premium-property variation:

- property quality: `interiorQual`, `condition`;
- age and refurbishment: `yearConstructed`, `lastRefurbish`;
- amenities: `balcony`, `garden`, `lift`, `hasKitchen`, `cellar`;
- building position: `floor`.

The experiment must be defined before inspecting its results, use the same training/test protocol, and compare candidate settings using cross-validation on the training data. Only the selected candidate should receive a final held-out test evaluation. Improvement should be judged using overall MAE and RMSE together with the high-rent residual spread and category-level errors.

`totalRent`, `serviceCharge`, and `heatingCosts` must remain excluded from predictors because they are directly tied to `baseRent` and would introduce target leakage or an unrealistic prediction dependency.

This experiment has **not yet been run**. Its result must be documented whether it improves the model or not.

## Conclusion

The finalized Ridge model performs substantially better than the mean baseline and achieves a held-out MAE of €308.84. Error analysis revealed one uniquely validated target-data error, which was conservatively excluded before retraining. After exclusion, the model's remaining weaknesses are concentrated in expensive, unusual, and some premium-neighbourhood listings.

The neighbourhood analysis, coefficients, and residual plot all point to the same limitation: the current feature set captures size and broad location effects but does not fully represent property quality and luxury-specific characteristics. The planned feature-expansion experiment will test that hypothesis in a controlled and reproducible way.
