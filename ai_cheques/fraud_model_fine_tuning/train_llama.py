from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model
model_name = "meta-llama/Llama-3-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    num_labels=2 # fraud / not fraud
)

# Dataset
dataset = load_dataset("csv", data_files="s3://my-bucket/fraud_checks.csv") #Uploads from S3 bucket, its a placeholder name for now

train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
temp_dataset = train_test["test"]

valid_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
valid_dataset = valid_test["train"]
test_dataset = valid_test["test"]

def preprocess(proprietary_dataset):
    proprietary_dataset_concatenated = (
        f"Date: {proprietary_dataset['date']}; "
        f"MICR: {proprietary_dataset['MICR']}; "
        f"Amount: {proprietary_dataset['amount']}; "
        f"Signature Match Score: {proprietary_dataset['signature_match_score']}; "
        f"Bank: {proprietary_dataset['bank_name']}"
    )

    tokenized = tokenizer(
        proprietary_dataset_concatenated,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = proprietary_dataset["fraud_label"]
    return tokenized


train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Metrics 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Training
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(test_dataset)
logger.info(f"Test Results: {test_results}")

trainer.save_model("./results")
