# Fine-Tuning & Serverless API for Cheque Fraud Detection
## Overview

This repository contains code for fine-tuning a Llama 3 3B 8-bit model on cheque data and deploying it via a serverless API. The solution enables fraud detection with minimal manual review, integrating human feedback for continuous improvement.

### Features

Custom Model Fine-Tuning:

- Llama 3 3B 8-bit fine-tuned with proprietary cheque data, low size (3.14GB) LLM quantized so it fits in serverless.

- Produces metrics: F1-Score, Recall, Precision, Confusion Matrix

Serverless API Deployment:

- AWS Lambda-based orchestration

- SageMaker endpoint integrated with Lambda for real-time inference

- Returns fraud probability and confidence metrics

Human-in-the-loop Feedback:

- Low-confidence predictions sent to manual review

- Reviewed data is used in monthly re-training
