# Data Source Summary

## Dataset

This project uses the Kaggle dataset [**Apartment Rental Offers in Germany**](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany), published by **CorrieBar** from rental listings collected from ImmoScout24.

| Property | Value |
|---|---|
| Kaggle identifier | `corrieaar/apartment-rental-offers-in-germany` |
| Current Kaggle version | Version 6 |
| Last updated on Kaggle | 20 April 2020 |
| Raw file used | `immo_data.csv` |
| Original shape | 268,850 listings and 49 columns |
| Local project path | `data/raw/immo_data.csv` |
| Original listing platform | ImmoScout24 |

The dataset includes rental prices, living space, room counts, regional labels, property characteristics, amenities, energy information, and some free-text listing fields. It contains advertised rental offers rather than confirmed rental transactions.

## Project scope

This project is limited to **Munich city** using the exact regional label:

```python
munich_df = raw_df.loc[
    raw_df["regio2"].eq("München")
].copy()
```

This selection produces **4,383 Munich listings across 42 neighbourhoods**.

Restricting the project to Munich creates a more coherent local rental market and allows `regio3` to represent neighbourhood-level location effects. The resulting model must therefore not be presented as a Germany-wide rental predictor.

## Prediction target and features

The prediction target is:

```text
baseRent = monthly cold rent in euros
```

Cold rent was chosen because it represents the property's advertised base price before additional service and heating charges.

The initial learning-focused NumPy model uses:

- `livingSpace` — living area in square metres;
- `noRooms` — number of rooms.

The professional scikit-learn models also use:

- `typeOfFlat` — apartment type;
- `regio3` — Munich neighbourhood.

Property-quality, age, refurbishment, amenity, and floor features are candidates for a later, controlled model-improvement experiment.

## Target-leakage exclusions

The following rent-related fields are excluded from model inputs:

- `totalRent`;
- `serviceCharge`;
- `heatingCosts`.

They are components of, or are directly combined with, the target. In simplified form:

```text
total rent ≈ base rent + service charge + heating costs
```

Using these columns would make the prediction task unrealistic and could reveal information too closely related to the answer. The project predicts cold rent from property characteristics rather than from another rent figure.

## Licence and permitted use

The official Kaggle metadata labels the licence as:

> **Data files © Original Authors**

The dataset description also states that the data belongs to ImmoScout24 and is for **research purposes only**.

This is not a permissive open-data licence such as CC0 or CC BY. For this project, the dataset is therefore treated as research and educational material:

- the raw CSV is not committed to this repository;
- the repository links users to the official Kaggle page instead of redistributing the data;
- ownership of the source listings is not claimed;
- the project should not present the data as approved for commercial use;
- anyone reusing the data should review the current Kaggle metadata and source terms themselves.

The licence and use statement can change, so the official [Kaggle dataset page](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany) remains authoritative.

## Attribution

A suitable project citation is:

> CorrieBar. *Apartment Rental Offers in Germany*, Version 6. Kaggle. Data collected from ImmoScout24. https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany

## Reproducing the dataset setup

Download the dataset from Kaggle and place the CSV locally at:

```text
data/raw/immo_data.csv
```

The raw dataset is intentionally excluded from Git. From the project root, the basic dataset shape and Munich selection can be verified with:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/raw/immo_data.csv'); munich = df[df['regio2'].eq('München')]; print('Full dataset rows:', len(df)); print('Munich city rows:', len(munich))"
```

Expected output:

```text
Full dataset rows: 268850
Munich city rows: 4383
```

## Known limitations

1. **Historical advertised prices:** the records represent listing offers from an earlier market period, not current Munich rents or confirmed transaction prices.
2. **Source inconsistencies:** the project found implausible and internally inconsistent values, including one validated target error. Cleaning decisions are documented separately.
3. **Missing data:** several property-quality and building columns contain substantial missingness.
4. **Duplicate or repeated offers:** listings were scraped at multiple times, so repeated or closely related offers may exist even when there are no exact duplicates in the selected modelling table.
5. **Limited generalization:** a model trained on this Munich subset should not be assumed to generalize to other cities, countries, or time periods.
6. **Listing-specific information:** important price drivers may be present only in unstructured descriptions, precise micro-location, photos, or information not recorded in the selected features.
7. **Privacy and provenance:** the source dataset contains some address and free-text fields. This project does not use those fields as model inputs and does not redistribute the raw records.
8. **Use restrictions:** the Kaggle licence label and research-only statement do not provide a broad commercial or redistribution permission.

## Related documentation

- [`data_selection_and_preprocessing.md`](data_selection_and_preprocessing.md) — complete filtering, cleaning, feature-selection, splitting, and preprocessing decisions;
- [`model_selection.md`](model_selection.md) — cross-validation, hyperparameter tuning, and selection of Ridge Regression;
- [`error_analysis.md`](error_analysis.md) — outlier investigation, category-level errors, feature effects, residual analysis, and model limitations.
