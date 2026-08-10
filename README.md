# Reusable AI Template

A reusable Python project template that I built as the foundation for future AI engineering and machine learning projects.

It provides a clean starting point with testing, linting, logging, configuration, GitHub Actions, and Docker already configured.

## Features

- 📁 `src` layout
- 🧪 `pytest`
- ✨ Ruff (formatting & linting)
- 📝 Logging
- ⚙️ YAML configuration
- 🤖 GitHub Actions (CI)
- 🐳 Docker

## Quick Start

Install:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Build Docker image:

```bash
docker build -t reusable-ai-template .
```

Run Docker:

```bash
docker run --rm reusable-ai-template
```

## Learning Notes

I wrote short summary notes covering the main concepts I learned while building this project.

➡️ **AI Notes:** https://github.com/nahom-mersha/ai-notes

## Purpose

This repository is **Project 0** of my AI Engineering roadmap and serves as the reusable foundation for future AI projects.