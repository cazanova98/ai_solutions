import boto3
import base64
import cv2
import json
import logging
import numpy as np
import os
import re
import unicodedata
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MICRBankMatcher:
    def __init__(self, s3_bucket, s3_key):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.rag_db = self.load_rag_db()

    def load_rag_db(self):
        s3_client = boto3.client("s3")
        try:
            response = s3_client.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
            rag_db = json.loads(response["Body"].read())
            logger.info("RAG database loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RAG database: {e}")
            self.rag_db = {}

    @staticmethod
    def normalize_bank_name(name: str) -> str:
        """
        Normalize bank name:
        - lowercase
        - remove accents
        - remove spaces and special characters
        """
        if not name:
            return ""
        name = name.lower().strip()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
        name = re.sub(r"[^a-z0-9]", "", name)  # remove non-alphanumeric
        return name

    def match(self, micr: str, bank_name: str):
        """
        Returns:
            rag_bank_name (str or None): normalized bank name from RAG
            match (bool): whether input bank_name matches RAG
        """
        try:
            digits = ''.join(re.findall(r'\d', micr))[:9]
            if len(digits) < 9:
                logger.warning(f"MICR does not contain 9 digits: {micr}")
                return None, False

            rag_bank_name = self.rag_db.get(digits)
            if not rag_bank_name:
                logger.warning(f"MICR {digits} not found in RAG database")
                return None, False

            normalized_input = self.normalize_bank_name(bank_name)
            normalized_rag = self.normalize_bank_name(rag_bank_name)

            match = normalized_input == normalized_rag
            logger.info(f"MICR: {micr}, Input: {bank_name}, RAG: {rag_bank_name}, Match: {match}")
            return match

        except Exception as e:
            logger.error(f"MICR lookup failed: {e}")
            return False
        
def enhance_image(img_b64, endpoint_name):
    logger.info("Enhancing image using endpoint: %s", endpoint_name)
    sm_client = boto3.client("sagemaker-runtime")
    payload = {"inputs": img_b64}
    try:
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read())
        enhanced_img_b64 = result["generated_image"]
        return base64.b64decode(enhanced_img_b64)
    except Exception as e:
        logger.error("Image enhancement failed: %s", e)
        raise

def is_low_quality(img, sharpness_threshold=100):
    gray = np.array(img.convert("L"))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    logger.info("Image sharpness (Laplacian variance): %.2f", laplacian_var)
    return laplacian_var < sharpness_threshold

def extract_text_from_image(image_bytes):
    logger.info("Extracting text from image using Textract")
    textract_client = boto3.client("textract")
    try:
        textract_response = textract_client.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['FORMS']
        )
        raw_text = ""
        for block in textract_response['Blocks']:
            if block['BlockType'] == 'LINE':
                raw_text += block['Text'] + "\n"
        logger.info("Extracted text length: %d", len(raw_text))
        return raw_text
    except Exception as e:
        logger.error("Text extraction failed: %s", e)
        raise

    
def extract_fields_with_bedrock(raw_text, model_id="anthropic.claude-v2"):
    logger.info("Extracting fields with Bedrock model: %s", model_id)
    bedrock_client = boto3.client("bedrock-runtime")
    system_prompt = (
        """You are an expert at extracting structured data from financial documents. 
        Extract ONLY the following fields from a check and return a valid JSON object: 
        amount, bank_name, MICR, date. Do not include any explanation or extra text.
        If a field is not present, return an empty string for that field.
        Schema: {
        "amount": "", 
        "bank_name": "", 
        "MICR": "", 
        "date": ""
        }"""
    )
    user_message = raw_text
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 512
    }
    try:
        bedrock_response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        fields_json = json.loads(bedrock_response["body"])["completion"]
        logger.info("Extracted fields: %s", fields_json)
        return fields_json
    except Exception as e:
        logger.error("Field extraction with Bedrock failed: %s", e)
        raise

def extract_signature_vector(image_bytes, endpoint_name): # Here we put the SageMaker signature vector endpoint, this is just a placeholder
    logger.info("Extracting signature vector using endpoint: %s", endpoint_name)
    sm_client = boto3.client("sagemaker-runtime")
    payload = {"image_bytes": base64.b64encode(image_bytes).decode('utf-8')}
    try:
        response = sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read())
        vectors = result.get("vectors", None)
        logger.info("Extracted signature vectors: %s", vectors)
        return vectors
    except Exception as e:
        logger.error("Signature vector extraction failed: %s", e)
        raise


def calculate_signature_similarity(query_vector, rag_vectors):
    """
    Calculates cosine similarity (%) between a query signature vector and each vector in a RAG dataset.
    Maps cosine similarity from [0, 1].
    Basically this measures how similar the signatures are.
    """
    # Normalize query vector
    query_vec = np.array(query_vector, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Normalize RAG vectors
    rag_matrix = np.array(rag_vectors, dtype=np.float32)
    rag_matrix = rag_matrix / np.linalg.norm(rag_matrix, axis=1, keepdims=True)

    # Compute cosine similarities (dot product)
    cosine_sims = np.dot(rag_matrix, query_vec)

    # Scale to percentage [0–1]
    similarities = (((cosine_sims + 1) / 2)).tolist()

    return similarities    

def detect_fraud(payload_json, endpoint_name):
    sm_client = boto3.client("sagemaker-runtime")
    response = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload_json)
    )
    result = json.loads(response["Body"].read())
    # Assume the model returns a dict with a 'fraud' key (True/False or string)
    return result.get("fraud", None), result.get("confidence", 0), result.get("model_version", "")

    
def lambda_handler(event, context):
    try:
        img_b64 = event["image_base64"]

        # 1. Enhance image ONLY if low quality
        if is_low_quality(Image.open(base64.b64decode(img_b64))):
            enhanced_img_bytes = enhance_image(img_b64, os.getenv("IMAGE_ENHANCEMENT_ENDPOINT"))
        else:
            enhanced_img_bytes = base64.b64decode(img_b64)

        # 2. Extract text
        raw_text = extract_text_from_image(enhanced_img_bytes)

        # 3. Extract fields
        fields_json = extract_fields_with_bedrock(raw_text)

        # 4. Extract signature vectors
        signature_vectors = extract_signature_vector(enhanced_img_bytes, os.getenv("SIGNATURE_EXTRACTION_ENDPOINT"))

        # 5. Calculate similarity of signatures (if any)
        signature_similarity = calculate_signature_similarity(signature_vectors[0], np.load("rag_signature_vectors.npy")) if signature_vectors else []

        # 6. MICR lookup and bank name comparison
        bank_match = MICRBankMatcher(s3_bucket=os.getenv("RAG_DB_BUCKET"), s3_key=os.getenv("RAG_DB_KEY")).match(fields_json.get("MICR", ""), fields_json.get("bank_name", ""))

        if bank_match:
        # 7. Prepare payload for fraud detection
            fraud_payload = {
                "Date": fields_json.get("date", ""),
                "MICR": fields_json.get("MICR", ""),
                "Amount": fields_json.get("amount", ""),
                "signature_match_score": signature_similarity[0] if signature_similarity else None,
                "bank_name": fields_json.get("bank_name", "")
            }
            fraud_result, confidence, model_version = detect_fraud(fraud_payload, os.getenv("FRAUD_DETECTION_ENDPOINT"))
        else:
            fraud_result = 1
            confidence = 1  # If bank doesn't match, flag as fraud (1)
            model_version = "N/A"
        # 8. Prepare output for S3
        client_id = event.get("client_id", "")
        check_id = event.get("check_id", "")
        output_data = {
            "client_id": client_id,
            "check_id": check_id,
            "date": fields_json.get("date", ""),
            "micr": fields_json.get("MICR", ""),
            "amount": fields_json.get("amount", ""),
            "signature_match_score": signature_similarity[0] if signature_similarity else None,
            "bank_name": fields_json.get("bank_name", ""),
            "bank_name_match": bank_match,
            "fraud_result": fraud_result,
            "confidence": confidence,
            "model_version": model_version
        }
        s3 = boto3.client("s3")
        s3_key = f"{client_id}/{check_id}.json" if client_id and check_id else f"output_{context.aws_request_id}.json"
        s3.put_object(Bucket=os.getenv("S3_OUTPUT_BUCKET"), Key=s3_key, Body=json.dumps(output_data))

        return {
            "status": "success",
            "s3_key": s3_key
        }
    except Exception as e:
        logger.error(f"Pipeline error for check_id={event.get('check_id', '')}: {e}", exc_info=True)
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }