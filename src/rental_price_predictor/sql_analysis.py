import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "rentals.db"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "immo_data.csv"

MUNICH_SUMMARY_QUERY = """
SELECT
    COUNT(*) AS listing_count,
    AVG(baseRent) AS average_rent,
    MIN(baseRent) AS minimum_rent,
    MAX(baseRent) AS maximum_rent
FROM rentals
WHERE regio2 = 'München';
"""

MISSINGNESS_QUERY = """
SELECT
    SUM(CASE WHEN yearConstructed IS NULL THEN 1 ELSE 0 END)
        AS missing_yearConstructed,
    SUM(CASE WHEN floor IS NULL THEN 1 ELSE 0 END)
        AS missing_floor,
    SUM(CASE WHEN interiorQual IS NULL THEN 1 ELSE 0 END)
        AS missing_interiorQual,
    SUM(CASE WHEN condition IS NULL THEN 1 ELSE 0 END)
        AS missing_condition
FROM rentals
WHERE regio2 = 'München';
"""

FLAT_TYPE_COUNTS_QUERY = """
SELECT
    typeOfFlat,
    COUNT(*) AS listing_count
FROM rentals
WHERE regio2 = 'München'
GROUP BY typeOfFlat
ORDER BY listing_count DESC;
"""

RENT_DISTRIBUTION_QUERY = """
SELECT
    CASE
        WHEN baseRent < 1000 THEN 'under_1000'
        WHEN baseRent < 2000 THEN '1000_to_1999'
        WHEN baseRent < 3000 THEN '2000_to_2999'
        ELSE '3000_plus'
    END AS rent_band,
    COUNT(*) AS listing_count
FROM rentals
WHERE regio2 = 'München'
GROUP BY rent_band
ORDER BY listing_count DESC;
"""


def create_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(CSV_PATH)

    connection = sqlite3.connect(DATABASE_PATH)

    raw_df.to_sql(
        name="rentals",
        con=connection,
        if_exists="replace",
        index=False,
    )

    connection.close()


def run_query(query: str) -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        create_database()
    connection = sqlite3.connect(DATABASE_PATH)
    result_df = pd.read_sql_query(query, connection)
    connection.close()

    return result_df


if __name__ == "__main__":
    print(run_query(MUNICH_SUMMARY_QUERY))
    print(run_query(MISSINGNESS_QUERY))
    print(run_query(FLAT_TYPE_COUNTS_QUERY))
    print(run_query(RENT_DISTRIBUTION_QUERY))
