import pandas as pd

DATA_PATH = "data/raw/immo_data.csv"

CANDIDATE_FEATURES = [
    "interiorQual",
    "condition",
    "yearConstructed",
    "lastRefurbish",
    "balcony",
    "garden",
    "lift",
    "hasKitchen",
    "cellar",
    "floor",
]


def main():
    raw_df = pd.read_csv(DATA_PATH)

    df = raw_df.loc[raw_df["regio2"].eq("München")].copy()

    # Apply the exact existing cleaning rules.
    df = df.loc[df["baseRent"].ge(100) & df["livingSpace"].le(500)].copy()

    # Remove the validated incorrect target row.
    df = df.drop(index=213625)

    audit_df = pd.DataFrame(
        {
            "dtype": df[CANDIDATE_FEATURES].dtypes.astype(str),
            "missing_count": df[CANDIDATE_FEATURES].isna().sum(),
            "missing_percent": (df[CANDIDATE_FEATURES].isna().mean().mul(100).round(1)),
            "unique_values": df[CANDIDATE_FEATURES].nunique(),
        }
    )

    print("Candidate feature audit:")
    print(audit_df.to_string())

    for feature in CANDIDATE_FEATURES:
        print(f"\n{feature} value counts:")
        print(df[feature].value_counts(dropna=False).head(15).to_string())


if __name__ == "__main__":
    main()
