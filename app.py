from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔧 CHANGE THIS TO YOUR MODEL FILENAME:
MODEL_PATH = "best.pt"  # Replace with your actual model filename

app = Flask(__name__)

# Load model at startup
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully: {MODEL_PATH}")
            return True
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

@app.route('/detect', methods=['POST'])
def detect_food():
    try:
        if model is None:
            return jsonify({
                "detections": [],
                "status": "error",
                "message": "Model not loaded"
            }), 500
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                "detections": [],
                "status": "error",
                "message": "No image provided"
            }), 400
        
        file = request.files['image']
        
        # Convert to PIL Image
        image = Image.open(file.stream)
        img_array = np.array(image)
        
        # Run YOLO detection
        results = model(img_array, verbose=False)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    
                    if confidence > 0.3:
                        detections.append({
                            "class": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                        })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            "detections": detections,
            "status": "success",
            "message": f"Found {len(detections)} food items",
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]}
        })
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({
            "detections": [],
            "status": "error",
            "message": f"Detection failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask application...")
        port = int(os.environ.get('PORT', 80))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
