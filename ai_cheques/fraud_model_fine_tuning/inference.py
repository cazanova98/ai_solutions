import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

def model_fn(model_dir):
    """Load model and tokenizer from SageMaker model_dir"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_version = os.path.basename(model_dir)  # or timestamp/version tracking
    return {"model": model, "tokenizer": tokenizer, "model_version": model_version}


def input_fn(request_body, request_content_type):
    """Parse input request"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data["text"]  # expecting {"text": "..."}
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_bundle):
    """Run prediction"""
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    model_version = model_bundle["model_version"]

    # Tokenize input
    inputs = tokenizer(
        input_data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Run through model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    confidence_score, predicted_class = torch.max(probs, dim=1)

    label_map = {0: "Not Fraud", 1: "Fraud"}
    fraud_prediction = label_map[predicted_class.item()]

    return {
        "fraud": fraud_prediction,        # "Fraud" or "Not Fraud"
        "confidence": confidence_score.item(),  # Probability of prediction
        "model_version": model_version
    }

def output_fn(prediction, response_content_type):
    """Format prediction output"""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
