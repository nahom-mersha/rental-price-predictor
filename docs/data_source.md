# Data Source Summary

This project uses the Kaggle **Apartment Rental Offers in Germany** dataset, collected from ImmoScout24 rental listings. The original CSV contains 268,850 listings and 49 columns. link: https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany?resource=download

For this project, the data is limited to **Munich city** using:

```python
df["regio2"] == "München"
````

This produces 4,383 listings across 42 Munich neighbourhoods.

The prediction target is `baseRent`, the monthly cold rent in euros. Initial model features are `livingSpace` and `noRooms`; later models will also use property, amenity, and neighbourhood features.

`totalRent` is excluded because it is too closely related to the target and would cause target leakage. Some features have missing values, and a few rent values appear unusual, so these will be investigated and handled in later steps.
