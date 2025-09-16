Cheque Fraud Detection AI Pipeline
Overview

This repository contains an end-to-end AI/ML pipeline for detecting cheque fraud. The system leverages AWS services, deep learning models, and serverless architecture to automate cheque verification, minimize manual review, and improve fraud detection accuracy.

Features

OCR Text Extraction: Extract raw text from cheques using AWS Textract.

Field Extraction: Structure extracted data with AWS Bedrock.

Image Enhancement: Improve cheque image quality using AWS SageMaker JumpStart.

Signature Detection: Detect signatures with YOLOv5 regardless of coordinates.

Signature & Bank Verification: Compare signatures and bank IDs against a RAG database for validation.

Fraud Prediction: Custom SageMaker model classifies cheques as fraud or non-fraud.

Serverless Orchestration: Fully automated workflow using AWS Lambda.

Monitoring & Logging: CloudWatch logs all processing steps and model outputs.

Pipeline Architecture

Cheque Upload → triggers Lambda orchestration

OCR with Textract → extract raw text

Field Extraction with Bedrock → structured fields

Image Upscaling → improves image clarity

Signature Extraction (YOLO) → bounding boxes for signature

Signature/Bank Validation → comparison with RAG database

Fraud Prediction → SageMaker endpoint returns probability

Logs & Monitoring → CloudWatch stores all logs and metrics

AI/ML Model

Models used: Llama 3 3B 8-bit (custom fine-tuned), Claude V2

Fine-tuning strategy: Re-trained monthly with new human-validated data

Evaluation Metrics: F1-Score, Recall, Precision, Confusion Matrix

Confidence Thresholds: Only predictions above the threshold are automatically accepted; others go to manual review

Data Pipeline

Preprocessing: OCR and image enhancement

Feature extraction: amount, signature, account number, date

Data storage: S3 buckets with versioning for checkpoints and datasets

Continuous training: Monthly updates using feedback from human reviewers

Deployment

Fully serverless architecture using AWS Lambda

SageMaker endpoints for real-time fraud prediction

Scalable to handle thousands of cheques per day

Monitoring

Key metrics tracked:

Confusion Matrix, F1-Score, Recall, Precision

% of predictions below confidence threshold

Lambda execution times and errors

Endpoint latency and throughput

Cost Estimation (Example)

10,000 cheques/day, 500 KB each, 2s processing per cheque

Approximate monthly cost: ~$950 (Textract, Bedrock, SageMaker, Lambda, CloudWatch)

How to Use

Upload cheque images to the configured S3 bucket

Lambda orchestrates the full pipeline

Check CloudWatch for logs and SageMaker endpoint for predictions

Human reviewers validate low-confidence predictions
