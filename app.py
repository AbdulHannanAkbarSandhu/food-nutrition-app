import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="NutriVision - Food Detection", 
    page_icon="🍎",
    layout="wide"
)

# API Configuration
st.sidebar.header("🔗 API Configuration")
api_url = st.sidebar.text_input(
    "YOLO API URL:", 
    value="https://your-ngrok-url.ngrok.io/detect",
    help="Enter your Google Colab ngrok URL here"
)

def detect_food_real(image, api_url):
    """Call the real YOLO API"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Make API request
        files = {'image': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "detections": [],
                "status": "error",
                "message": f"API request failed: {response.status_code}"
            }
            
    except Exception as e:
        return {
            "detections": [],
            "status": "error",
            "message": f"API connection failed: {str(e)}"
        }

def draw_bounding_boxes(image, detections):
    """Draw bounding boxes on the image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for detection in detections:
        bbox = detection["bbox"]
        x, y, w, h = bbox
        
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        
        # Draw label
        label = f"{detection['class']} ({detection['confidence']:.2f})"
        draw.text((x, y - 20), label, fill="red")
    
    return img_copy

# Main app
st.title("🍎 NutriVision - Real AI Food Detection")
st.write("Upload an image to detect food items using your trained YOLOv8 model")

# Test API connection
if st.sidebar.button("🔧 Test API Connection"):
    try:
        health_url = api_url.replace('/detect', '/health')
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected!")
            st.sidebar.json(response.json())
        else:
            st.sidebar.error("❌ API Connection Failed")
    except:
        st.sidebar.error("❌ Cannot reach API")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of food items"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)

with col2:
    st.header("🔍 Detection Results")
    
    if uploaded_file is not None:
        if st.button('🚀 Detect Food Items', type="primary"):
            if not api_url or "your-ngrok-url" in api_url:
                st.error("❌ Please enter your real ngrok API URL in the sidebar")
            else:
                with st.spinner('Calling your YOLO API...'):
                    result = detect_food_real(image, api_url)
                    
                if result and result["status"] == "success":
                    st.success(f"✅ {result['message']}")
                    
                    if result["detections"]:
                        # Show image with bounding boxes
                        img_with_boxes = draw_bounding_boxes(image, result["detections"])
                        st.image(img_with_boxes, caption='Detected Food Items', use_column_width=True)
                        
                        # Show detection results
                        st.subheader("🍽️ Detected Food Items:")
                        
                        for detection in result["detections"]:
                            with st.container():
                                st.write(f"**{detection['class'].title()}**")
                                st.write(f"Confidence: {detection['confidence']:.1%}")
                                st.write(f"Location: {detection['bbox']}")
                                st.write("---")
                    
                    # Show full JSON
                    with st.expander("📄 Full API Response"):
                        st.json(result)
                        
                else:
                    st.error(f"❌ {result.get('message', 'Detection failed')}")
    else:
        st.info("👆 Upload an image to get started")

# Instructions
st.markdown("---")
st.subheader("📋 Setup Instructions")
st.write("1. **Run the Google Colab notebook** with your YOLO model")
st.write("2. **Copy the ngrok URL** from Colab output")
st.write("3. **Paste the URL** in the sidebar (replace 'your-ngrok-url')")
st.write("4. **Click 'Test API Connection'** to verify")
st.write("5. **Upload an image** and click 'Detect Food Items'")
