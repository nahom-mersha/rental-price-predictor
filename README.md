
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
