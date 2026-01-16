
# Sentiment — Multilingual Toxicity Detection

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

Professional, research-focused implementation for detecting toxicity across multiple languages. This repository contains scripts for model training and a small inference API for serving predictions.

**Authors:** Rigon Pira, Gentrit Halimi, Argjend Azizi, Euron Ramadani, Ardit Shabani — UBT College

---

## What is this project?

This repository provides a compact, reproducible pipeline for training and serving a multilingual toxicity classifier. It is designed for academic experiments, reproducibility, and easy deployment for evaluation purposes.

- Audience: NLP researchers, undergraduate/graduate students, and practitioners.
- Focus: clarity, reproducibility, and lightweight deployment.

## Table of Contents
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Data & training](#data--training)
- [Running the API](#running-the-api)
- [Evaluation & reporting](#evaluation--reporting)
- [Design diagram](#design-diagram)
- [Authors & contact](#authors--contact)

## Project structure

High-level overview:

| Path | Purpose |
|---|---|
| `api/` | Inference server and saved model (`toxicity_api.py`, `toxicity_model.h5`) |
| `data/` | Raw and processed datasets (`train.csv`) |
| `model_training/` | Training scripts (`train_model.py`) |

Snapshot:

```text
Sentiment/
├─ api/
│  ├─ toxicity_api.py
│  └─ toxicity_model.h5
├─ data/
│  └─ train.csv
└─ model_training/
   └─ train_model.py
```

## Quick start

Prerequisites: `Python 3.8+` and a virtual environment.

Example `requirements.txt`:

```text
tensorflow>=2.8
numpy
pandas
scikit-learn
fastapi
uvicorn
flask

# Optional: text preprocessing
regex
nltk
```

Installation (Windows PowerShell example):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install "tensorflow>=2.8" numpy pandas scikit-learn fastapi uvicorn flask
```

Typical workflow:

1. Prepare or verify your dataset:
1. Verify your dataset is present in `data/` (e.g., `data/train.csv`).
2. Train a model (from the project root):

```powershell
cd model_training
python train_model.py --data ../data/train.csv --epochs 10 --output ../api/toxicity_model.h5
```

*(If you keep processed files in `data/`, point `--data` to that file instead.)*

3. Run the API server (two common patterns):

- If `api/toxicity_api.py` exposes a `FastAPI` app named `app`:

```powershell
uvicorn api.toxicity_api:app --host 0.0.0.0 --port 8000
```

- If `api/toxicity_api.py` is a Flask app:

```powershell
python api/toxicity_api.py
```

If you want, I can inspect `api/toxicity_api.py` and add the exact command below.

## Data & training

`train_model.py`: model definition, training, and model export to `api/toxicity_model.h5`.

Reproducibility recommendations:

- Fix random seeds for Python, NumPy and TensorFlow.
- Log hyperparameters and metrics to a JSON/CSV file.
- Keep raw data immutable and versioned.


## Authors & contact

- Rigon Pira
- Gentrit Halimi
- Argjend Azizi
- Euron Ramadani
- Ardit Shabani

Affiliation: UBT College

For questions or collaboration: open an issue in the repository or contact the authors.
