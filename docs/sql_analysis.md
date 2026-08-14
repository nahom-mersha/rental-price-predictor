# SQL Analysis

This project includes a small SQLite-based analysis workflow to demonstrate how SQL can be used alongside Pandas in a machine-learning project.

The original rental dataset is provided as a CSV file. For SQL practice, the dataset is loaded into a local SQLite database. SQL is then used to filter and aggregate the rental data before the results are returned to Pandas.

## Why SQL?

In many production data and AI systems, data is stored in relational databases or data warehouses rather than directly in CSV files.

A common workflow is:

```text
Database
    ↓
SQL query
    ↓
Selected / aggregated data
    ↓
Python / Pandas
    ↓
Preprocessing and modeling
```

Because this project starts with a CSV dataset, SQLite is used as a lightweight local database:

```text
Raw CSV
    ↓
SQLite database
    ↓
SQL analysis
    ↓
Pandas
```

The generated SQLite database is stored locally and is not committed to the repository because it can be recreated from the raw dataset.

## Database Setup

The SQL analysis code is located in:

```text
src/rental_price_predictor/sql_analysis.py
```

The raw CSV is loaded into a SQLite table named `rentals`.

If the database does not yet exist, the analysis code creates it automatically before executing a query.

## Filtering Munich Listings

SQL can select only listings from Munich:

```sql
SELECT COUNT(*)
FROM rentals
WHERE regio2 = 'München';
```

This query returned **4,383 raw Munich listings**.

The SQL analysis operates on the raw imported data, so these counts are before the cleaning and exclusions used by the machine-learning pipeline.

## Aggregate Rental Statistics

SQL aggregate functions can summarize the rental market:

```sql
SELECT
    COUNT(*) AS listing_count,
    AVG(baseRent) AS average_rent,
    MIN(baseRent) AS minimum_rent,
    MAX(baseRent) AS maximum_rent
FROM rentals
WHERE regio2 = 'München';
```

The raw Munich data produced:

- Listing count: 4,383
- Average base rent: approximately €1,767.75
- Minimum base rent: €1
- Maximum base rent: €20,100

The extreme minimum and maximum values illustrate why data-quality checks and cleaning are necessary before model training.

## Missing-Value Analysis

Missing values can be counted using conditional aggregation:

```sql
SELECT
    SUM(
        CASE
            WHEN yearConstructed IS NULL THEN 1
            ELSE 0
        END
    ) AS missing_yearConstructed,

    SUM(
        CASE
            WHEN floor IS NULL THEN 1
            ELSE 0
        END
    ) AS missing_floor,

    SUM(
        CASE
            WHEN interiorQual IS NULL THEN 1
            ELSE 0
        END
    ) AS missing_interiorQual,

    SUM(
        CASE
            WHEN condition IS NULL THEN 1
            ELSE 0
        END
    ) AS missing_condition
FROM rentals
WHERE regio2 = 'München';
```

Observed missing counts:

| Feature | Missing values | Missing percentage |
| --- | ---: | ---: |
| `yearConstructed` | 751 | 17.13% |
| `floor` | 631 | 14.40% |
| `interiorQual` | 1,584 | 36.14% |
| `condition` | 1,206 | 27.52% |

This demonstrates how SQL can be used to investigate feature availability before preprocessing and modeling.

## Category Counts

SQL can also inspect categorical variables:

```sql
SELECT
    typeOfFlat,
    COUNT(*) AS listing_count
FROM rentals
WHERE regio2 = 'München'
GROUP BY typeOfFlat
ORDER BY listing_count DESC;
```

The most common category was `apartment`, with 2,045 listings.

There were also 847 listings with a missing `typeOfFlat` value.

This type of query is useful for understanding categorical feature distributions before encoding them for a machine-learning model.

## Rental Price Distribution

A `CASE` expression can create calculated categories directly inside a query:

```sql
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
```

The resulting distribution was:

| Base-rent range | Listings |
| --- | ---: |
| €1,000–€1,999 | 2,470 |
| €2,000–€2,999 | 795 |
| Under €1,000 | 737 |
| €3,000+ | 381 |

The four groups contain all 4,383 raw Munich listings.

This demonstrates an important SQL pattern:

```text
WHERE
    ↓
Select relevant rows

CASE
    ↓
Create a calculated category

GROUP BY
    ↓
Group rows by that category

COUNT(*)
    ↓
Count rows within each group
```

## SQL Concepts Practiced

The analysis demonstrates:

- `SELECT` and `FROM`
- `WHERE`
- `COUNT`, `AVG`, `MIN`, and `MAX`
- aliases with `AS`
- `GROUP BY`
- `ORDER BY`
- `NULL` and `IS NULL`
- `CASE WHEN`
- conditional aggregation with `SUM`
- calculated columns
- using SQL results with Pandas

## SQL and Pandas

SQL and Pandas are complementary rather than competing tools.

SQL is commonly used to retrieve, filter, join, and aggregate data where it is stored. Pandas is then useful for Python-side exploration, preprocessing, feature engineering, and integration with machine-learning libraries.

In this project, `pandas.read_sql_query()` executes the SQL query and returns the result as a Pandas DataFrame, allowing the SQL database workflow to integrate naturally with the rest of the Python project.