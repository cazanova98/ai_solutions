# Signature Detection for Cheque Fraud Pipeline
## Overview

This repository contains code for detecting signatures in cheque images using YOLOv5 and deploying the model via AWS SageMaker. Signature detection is a key step in validating cheque authenticity and comparing against a RAG database for fraud detection.

### Features

Signature Extraction:

- Detects signatures regardless of their location on the cheque

- Produces bounding boxes and confidence scores

SageMaker Deployment:

- Model deployed as an endpoint for real-time inference

- Fully integrated with the cheque fraud detection pipeline
