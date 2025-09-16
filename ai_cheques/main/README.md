# Lambda Orchestration Function for Cheque Fraud Detection
## Overview

This AWS Lambda function orchestrates the end-to-end cheque fraud detection pipeline. It coordinates the processing of cheque images, triggers AI/ML models, and logs results, ensuring serverless, scalable, and automated fraud detection.

### Features

#### Automated orchestration of the full pipeline:

- Signature detection with YOLOv5
  
- OCR with AWS Textract

- Field extraction with AWS Bedrock

- Image enhancement with SageMaker JumpStart


- Fraud prediction with SageMaker custom model

- Easy integration with existing systems (e.g., Early Warning System) through S3.

- Error handling and logging for reliability

- Serverless execution with automatic scaling

- CloudWatch logging for monitoring and audit

### Function Workflow

- Trigger: Cheque image uploaded to S3 bucket

- Step 1: Enhance image with SageMaker JumpStart Run OCR on the cheque using Textract

- Step 2: Run OCR on the cheque using Textract 

- Step 3: Extract structured fields with Bedrock

- Step 4: Detect signature with YOLOv5

- Step 5: Validate signature and bank ID against RAG database

- Step 6: Send structured data to SageMaker endpoint for fraud prediction

- Step 7: Log results and metrics in CloudWatch

### Deployment

- Define the function using AWS SAM

- Attach IAM role with permissions for:

- S3 (read/write)

- SageMaker endpoints

- Textract & Bedrock

- CloudWatch logs

- Set up S3 event triggers for new cheque uploads
