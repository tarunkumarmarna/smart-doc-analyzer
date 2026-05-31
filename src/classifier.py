"""
classifier.py — Document classification using fine-tuned distilBERT
Inference only. Training happens in notebooks/train.ipynb.

WHY inference separate from training:
Training code (Trainer, TrainingArguments, datasets) is not needed in production.
Keeping inference wrapper clean and small = faster container startup.

Model loaded from HuggingFace Hub — no large files in repo.
WHY Hub: pytorch_model.bin is ~250MB. Too large for GitHub (100MB limit).
Push once to Hub, load anywhere with one line.
"""

from transformers import (
    AutoTokenizer,                       # loads tokenizer matching the model
    AutoModelForSequenceClassification   # loads classification head on top of distilBERT
)
import torch   # needed for inference forward pass
import torch.nn.functional as F  # for softmax to convert logits to probabilities


# AG News label mapping — 4 document categories
# WHY AG News: 4 clean categories, 120k samples, standard benchmark
# In README: noted as proxy — domain adaptation is Month 2 extension
LABEL_MAP = {
    0: "World / News",
    1: "Sports",
    2: "Business / Finance",
    3: "Technology / Science"
}

# HuggingFace Hub model path — replace with your actual username after pushing
# trainer.push_to_hub("your-username/smart-doc-classifier") in train.ipynb
MODEL_NAME = "your-username/smart-doc-classifier"

# load tokenizer and model once at module level
# WHY: transformer model load takes 2-3 seconds — load once, reuse many times
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# AutoModelForSequenceClassification = distilBERT encoder + linear classification head
# The head maps [CLS] token embedding → num_labels logits
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# set model to evaluation mode
# WHY: disables dropout layers (used in training for regularization)
# dropout in inference would give different results each forward pass — wrong
model.eval()


def classify_document(text: str) -> dict:
    """
    Classify a document or chunk of text into one of 4 categories.
    Returns label name and confidence score.

    Process:
    1. Tokenize text (BPE subword tokenization)
    2. Forward pass through distilBERT + classification head → logits
    3. Softmax logits → probabilities
    4. Argmax → predicted class index
    5. Map index → label name
    """
    # tokenize input text
    # truncation=True: cut to max 512 tokens if longer (distilBERT limit)
    # return_tensors="pt": return PyTorch tensors not lists
    inputs = tokenizer(
        text,
        return_tensors="pt",   # pt = PyTorch
        truncation=True,       # truncate if > 512 tokens
        max_length=512,        # distilBERT maximum sequence length
        padding=True           # pad shorter sequences to same length
    )

    # torch.no_grad(): disable gradient computation during inference
    # WHY: we don't need gradients — saves memory and speeds up forward pass
    with torch.no_grad():
        # forward pass — returns logits (raw unnormalized scores per class)
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_labels) = (1, 4)

    # softmax converts logits to probabilities that sum to 1
    # dim=1: apply softmax across the class dimension
    probabilities = F.softmax(logits, dim=1)  # shape: (1, 4)

    # argmax: index of highest probability = predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()  # scalar int

    # get confidence score for predicted class
    confidence = probabilities[0][predicted_class].item()  # scalar float

    return {
        "label": LABEL_MAP[predicted_class],        # human-readable label
        "label_id": predicted_class,                # integer class index
        "confidence": round(confidence, 4),         # confidence 0-1
        "all_scores": {                             # scores for all classes
            LABEL_MAP[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(LABEL_MAP))
        }
    }
