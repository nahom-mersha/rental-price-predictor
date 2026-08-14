import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "rentals.db"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "immo_data.csv"


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


def run_query(query: str):
    connection = sqlite3.connect(DATABASE_PATH)
    result_df = pd.read_sql_query(query, connection)
    connection.close()

    return result_df


if __name__ == "__main__":
    q = """
SELECT COUNT(*)
FROM rentals
WHERE regio2 = 'München'
AND yearConstructed IS NULL
"""

    print(run_query(q))
