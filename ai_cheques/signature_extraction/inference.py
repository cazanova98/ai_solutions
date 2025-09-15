# inference.py
import io
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import clip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_signature_vector(image_path, yolo_model_path="YOLOs.pt", clip_model_name="ViT-B/32"):
    """
    Automatically detects signature(s) in an image using YOLO and return their CLIP vector embeddings.
    Returns a list of vectors (one per signature).
    YOLO is an open source pre-trained model that dynamically detects objects in an image and returns its coordinates.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model loading
    try:
        yolo = YOLO(yolo_model_path)
        model, preprocess = clip.load(clip_model_name, device=device)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None

    # Image loading
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Could not open image: {e}")
        return None

    # Detection and vector extraction
    try:
        results = yolo(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
    except Exception as e:
        logger.error(f"YOLO detection or output parsing failed: {e}")
        return None

    vectors = []
    for i, label in enumerate(labels):
        if class_names[int(label)] == "signature":
            x1, y1, x2, y2 = boxes[i].astype(int)
            sig_img = image.crop((x1, y1, x2, y2))
            try:
                sig_tensor = preprocess(sig_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vector = model.encode_image(sig_tensor).cpu().numpy()[0]
                    vector = vector / np.linalg.norm(vector)  # normalize
                vectors.append(vector)
            except Exception as e:
                logger.error(f"CLIP embedding failed: {e}")
    if not vectors:
        logger.warning("No signature detected in image.")
    return vectors if vectors else None

# SageMaker handler
def model_fn(model_dir):
    # Load models once at startup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO("YOLOs.pt")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return {"yolo": yolo, "clip_model": clip_model, "preprocess": preprocess, "device": device}

def input_fn(request_body, request_content_type):
    # Accept image bytes
    if request_content_type == "application/octet-stream":
        return Image.open(io.BytesIO(request_body)).convert("RGB")
    raise Exception("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    # Run detection and vector extraction
    yolo = model["yolo"]
    clip_model = model["clip_model"]
    preprocess = model["preprocess"]
    device = model["device"]

    results = yolo(input_data)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names

    vectors = []
    for i, label in enumerate(labels):
        if class_names[int(label)] == "signature":
            x1, y1, x2, y2 = boxes[i].astype(int)
            sig_img = input_data.crop((x1, y1, x2, y2))
            sig_tensor = preprocess(sig_img).unsqueeze(0).to(device)
            with torch.no_grad():
                vector = clip_model.encode_image(sig_tensor).cpu().numpy()[0]
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector.tolist())
    return {"vectors": vectors}