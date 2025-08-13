import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import json
import os

# Page config
st.set_page_config(
    page_title="NutriVision - Food Detection", 
    page_icon="🍎",
    layout="wide"
)

#  CHANGE THIS TO YOUR MODEL FILENAME:
MODEL_PATH = "best.pt"  # actual model filename

@st.cache_resource
def load_model():
    """Load the YOLO model (cached for performance)"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file {MODEL_PATH} not found. Please upload your YOLOv8 model file.")
            return None
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def detect_food(image):
    """Process uploaded image and return detection results"""
    model = load_model()
    if model is None:
        return None
        
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Run YOLO detection
        results = model(img_array)
        
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
        
        return {
            "detections": detections,
            "status": "success",
            "message": f"Found {len(detections)} food items",
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]}
        }
        
    except Exception as e:
        return {
            "detections": [],
            "status": "error", 
            "message": f"Detection failed: {str(e)}"
        }

# Streamlit UI
st.title("🍎 NutriVision - AI Food Detection")
st.write("Upload an image to detect food items using YOLOv8")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of food items"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.header("Detection Results")
    
    if uploaded_file is not None:
        if st.button(' Detect Food Items', type="primary"):
            with st.spinner('Analyzing your food image...'):
                result = detect_food(image)
                
            if result and result["status"] == "success":
                st.success(f" {result['message']}")
                
                if result["detections"]:
                    st.subheader("Detected Food Items:")
                    
                    # Display results in a nice format
                    for i, detection in enumerate(result["detections"]):
                        with st.container():
                            st.write(f"**{detection['class'].title()}**")
                            st.write(f"Confidence: {detection['confidence']:.1%}")
                            st.write(f"Location: {detection['bbox']}")
                            st.write("---")
                
                # Show full JSON for API development
                with st.expander(" Full API Response (for developers)"):
                    st.json(result)
                    
            elif result:
                st.error(f" {result['message']}")
            else:
                st.error("Failed to process image")
    else:
        st.info("� Upload an image to get started")

# API Information
st.markdown("---")
st.subheader(" API Integration")
st.write("This app can be used as an API endpoint for your applications.")

# Show API endpoint 
st.code("""
# API Endpoint (after deployment):
POST https://your-app-name.streamlit.app/api/detect

# Example usage:
import requests
import json

files = {'image': open('food_image.jpg', 'rb')}
response = requests.post('https://your-app-name.streamlit.app/api/detect', files=files)
result = response.json()
print(result)
""")
