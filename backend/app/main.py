from fastapi import FastAPI
from pydantic import BaseModel
from .model import predict_label

app = FastAPI()


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str


@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    label = predict_label(input.text)
    return {"label": label}


@app.get("/health")
def health():
    return {"status": "ok"}
