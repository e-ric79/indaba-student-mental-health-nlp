import joblib
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 30

# load tokenizer + BERT once at import time (not per-request)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()  # inference mode - disables dropout, no gradient tracking

# load the trained logistic regression classifier
# clf = joblib.load("saved_models/Logistic_model.pkl")
clf = joblib.load("saved_models/Logistic_model.pkl")


def get_embedding(text: str):
    """Convert a single text string into a mean-pooled BERT embedding."""
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():  # no backprop needed - frozen BERT used only as feature extractor
        outputs = bert_model(**encoded)

    # mean pooling over token embeddings, masking out padding tokens
    token_embeddings = outputs.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    return mean_pooled.numpy()


def predict_label(text: str):
    """Return the predicted label (string) for a given input text."""
    embedding = get_embedding(text)
    label = clf.predict(embedding)[0]
    return label
