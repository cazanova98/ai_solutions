# Cheque Fraud Detection AI Pipeline
## Overview

This repository contains an end-to-end AI/ML pipeline for detecting cheque fraud. The system leverages AWS services, deep learning models, and serverless architecture to automate cheque verification, minimize manual review, and improve fraud detection accuracy.

### Features

- OCR Text Extraction: Extract raw text from cheques using AWS Textract.

- Field Extraction: Structure extracted data with AWS Bedrock.

- Image Enhancement: Improve cheque image quality using AWS SageMaker JumpStart.

- Signature Detection: Detect signatures with YOLOv5 regardless of coordinates.

- Signature & Bank Verification: Compare signatures and bank IDs against a RAG database for validation.

- Fraud Prediction: Custom SageMaker model classifies cheques as fraud or non-fraud.

- Serverless Orchestration: Fully automated workflow using AWS Lambda.

- Monitoring & Logging: CloudWatch logs all processing steps and model outputs.

### Pipeline Architecture

- Cheque Upload → triggers Lambda orchestration

- Image Upscaling → improves image clarity

- OCR with Textract → extract raw text

- Field Extraction with Bedrock → structured fields

- Signature Extraction (YOLO) → bounding boxes for signature

- Signature/Bank Validation → comparison with RAG database

- Fraud Prediction → SageMaker endpoint returns probability

- Logs & Monitoring → CloudWatch stores all logs and metrics

### AI/ML Model

- Models used: Llama 3 3B 8-bit (custom fine-tuned), Claude V2

- Fine-tuning strategy: Re-trained monthly with new human-validated data (not implemented here, but you can do it with EventBridge)

- Evaluation Metrics: F1-Score, Recall, Precision, Confusion Matrix

### RAG

- Signature databases used to compare against signatures in checks to obtain signature compatibility %
- Bank ID databases used to compare MICR bank id against the bank name to check for possible missmatch.
  
### Data Pipeline

- Preprocessing: OCR and image enhancement

- Feature extraction: amount, signature, account number, date

- Data storage: S3 buckets with versioning for checkpoints and datasets

- Continuous training: Monthly updates using feedback from human reviewers

### Deployment

- Fully serverless architecture using AWS Lambda

- SageMaker endpoints for real-time fraud prediction

- Scalable to handle thousands of cheques per day

- Check CloudWatch for logs and SageMaker endpoint for predictions

- Human reviewers validate low-confidence predictions
- 
### Future Improvements

- Collaborate with business stakeholders to refine and optimize feature engineering for improved model performance.
  

