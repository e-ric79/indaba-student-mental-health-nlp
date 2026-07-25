# Mental Health Text Classifier

A text classification app that flags text into one of four categories — **Depression, Alcohol, Suicide, Drugs** — using BERT embeddings and logistic regression, served via FastAPI with a Streamlit frontend.

## Overview

- **Embeddings**: frozen `bert-base-uncased`, mean-pooled over tokens (max_length=30)
- **Classifier**: Logistic Regression (`class_weight='balanced'`)
- **Backend**: FastAPI serving the trained model
- **Frontend**: Streamlit UI for interactive predictions
- **Deployment**: Docker Compose, CI/CD via GitHub Actions

## Architecture

```
Text input → BERT (frozen, mean-pooled) → Logistic Regression → Predicted label
                                                    ↓
                                          FastAPI backend (/predict)
                                                    ↓
                                          Streamlit frontend
```

## Project Structure

```
project/
├── .github/workflows/     # CI/CD pipeline
├── backend/                # FastAPI app + model inference
├── frontend/                # Streamlit UI
├── docker-compose.yml
└── requirements.txt
```

## Running Locally

```bash
docker compose up --build
```

- Backend: http://localhost:8000/docs (FastAPI Swagger UI)
- Frontend: http://localhost:8501

## API

**POST** `/predict`

Request:
```json
{ "text": "sample input text" }
```

Response:
```json
{ "label": "Depression" }
```

## Model Training

See `notebooks/` for the training pipeline: data cleaning → BERT embedding extraction → logistic regression training → evaluation.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, scikit-learn, FastAPI, Streamlit, Docker, GitHub Actions

---

*Portfolio project — not intended for clinical or diagnostic use.*