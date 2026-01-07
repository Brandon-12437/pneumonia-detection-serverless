import os
import onnxruntime as ort
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json

# Optimize for Lambda's CPU
os.environ["OMP_NUM_THREADS"] = "1"

# Global variables for model caching (Warm Start)
session = None
input_name = None
output_name = None
classes = ["normal", "pneumonia"]

def load_model():
    """Loads the model into memory only once."""
    global session, input_name, output_name
    if session is None:
        print("--> Loading ONNX model...")
        session = ort.InferenceSession(
            "xception_pneumonia.onnx",
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"--> Model loaded. Input: {input_name}, Output: {output_name}")

def preprocess_image(url):
    """Downloads and prepares image for Xception."""
    print(f"--> Fetching image from: {url}")
    response = requests.get(url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image = image.resize((150, 150))
    # Normalize to [0, 1] as expected by most Xception models
    x = np.array(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def lambda_handler(event, context=None):
    load_model()
    
    try:
        # 1. Parse Input (Handles direct calls and API Gateway)
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event

        url = body.get("url")
        if not url:
            return {"statusCode": 400, "body": json.dumps({"error": "No URL provided"})}

        # 2. Inference
        x = preprocess_image(url)
        raw_output = session.run([output_name], {input_name: x})[0][0]
        print(f"--> RAW MODEL OUTPUT: {raw_output}")

        # 3. Handle different output shapes
        # If model returns 1 value (Sigmoid), we calculate the other class
        if len(raw_output) == 1:
            p_pneumonia = float(raw_output[0])
            p_normal = 1.0 - p_pneumonia
            preds = [p_normal, p_pneumonia]
        else:
            # If model returns 2 values (Softmax)
            preds = [float(p) for p in raw_output]

        result = dict(zip(classes, preds))
        print(f"--> PROBABILITIES: {result}")

        # 4. Return successful response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        print(f"!!! ERROR: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "normal": 0.0, "pneumonia": 0.0})
        }