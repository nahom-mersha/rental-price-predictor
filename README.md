
# Rental Price Predictor

A rental price prediction project that implements linear regression from scratch with NumPy and compares it with scikit-learn models.

This repository starts from a reusable AI project template. It includes testing, linting, logging, configuration, GitHub Actions, and Docker as a foundation for the project.

## Planned Features

- Load and explore rental-price data
- Prepare features for modelling
- Implement linear regression from scratch with NumPy
- Compare the NumPy implementation with scikit-learn models
- Evaluate models using appropriate regression metrics
- Document findings and limitations

## Development Tools

- 📁 `src` layout
- 🧪 `pytest`
- ✨ Ruff (formatting and linting)
- 📝 Logging
- ⚙️ YAML configuration
- 🤖 GitHub Actions (CI)
- 🐳 Docker

## Quick Start

Install:

```bash
pip install -e ".[dev]"
````

Run tests:

```bash
pytest
```

Build the Docker image:

```bash
docker build -t rental-price-predictor .
```

Run the Docker image:

```bash
docker run --rm rental-price-predictor
```

## Learning Notes

Project learning notes will be added to my [AI Notes repository](https://github.com/nahom-mersha/ai-notes) as the project progresses.

## Purpose

This repository is **Project 2** of my AI Engineering roadmap. Its purpose is to develop a practical understanding of regression, NumPy-based model implementation, and model evaluation.

## Reproduce the Dataset Setup

This project expects the raw Kaggle dataset to be placed locally at:

data/raw/immo_data.csv

The raw dataset is not committed to Git because it is too large. Download it from the Kaggle dataset page linked in docs/data_source.md.

From the project root, install the project dependencies:

pip install -e .

Then verify that the dataset can be loaded and filtered to Munich city:

python -c "import pandas as pd; df = pd.read_csv('data/raw/immo_data.csv'); munich = df[df['regio2'].eq('München')]; print('Full dataset rows:', len(df)); print('Munich city rows:', len(munich))"

Expected output:

Full dataset rows: 268850
Munich city rows: 4383
