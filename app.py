from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import openai
import os
from PIL import Image
import io
import json
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Food Detection API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
try:
    model = YOLO("best.pt")  # Your trained model
    logger.info("YOLOv8 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# API Keys (set as environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_APP_KEY = os.getenv("NUTRITIONIX_APP_KEY")

# Initialize OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class FoodDetectionService:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
    
    def detect_food_items(self, image: np.ndarray) -> List[Dict]:
        """Detect food items using YOLOv8"""
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            results = model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detections.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "class_id": class_id
                        })
            
            return self.merge_overlapping_detections(detections)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    def merge_overlapping_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge overlapping detections of the same class"""
        if not detections:
            return []
        
        # Group by class_name
        grouped = {}
        for det in detections:
            class_name = det["class_name"]
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(det)
        
        merged_detections = []
        for class_name, class_detections in grouped.items():
            if len(class_detections) == 1:
                merged_detections.extend(class_detections)
            else:
                # For multiple detections of same class, keep the highest confidence one
                # In production, you might want more sophisticated merging
                best_detection = max(class_detections, key=lambda x: x["confidence"])
                best_detection["count"] = len(class_detections)
                merged_detections.append(best_detection)
        
        return merged_detections

    async def determine_portion_type(self, food_name: str) -> Dict[str, Any]:
        """Use GPT to determine if food should be measured in grams or count"""
        if not OPENAI_API_KEY:
            # Fallback logic if no OpenAI API
            return self.fallback_portion_logic(food_name)
        
        try:
            prompt = f"""
            For the food item "{food_name}", determine:
            1. Should this be measured in "grams" or "count" (pieces)?
            2. What is a typical serving size?
            3. Provide a brief explanation.
            
            Respond in JSON format:
            {{
                "measurement_type": "grams" or "count",
                "typical_serving": number,
                "serving_unit": "grams" or "pieces",
                "explanation": "brief explanation",
                "alternative_names": ["alternative1", "alternative2"]
            }}
            
            Examples:
            - Apple: count (1 piece = ~150g)
            - Rice: grams (typical serving 100-150g)
            - Banana: count (1 piece = ~120g)
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"GPT portion determination failed: {e}")
            return self.fallback_portion_logic(food_name)
    
    def fallback_portion_logic(self, food_name: str) -> Dict[str, Any]:
        """Fallback logic when GPT is not available"""
        count_foods = ["apple", "banana", "orange", "egg", "cookie", "donut", "slice", "piece"]
        
        is_count = any(item in food_name.lower() for item in count_foods)
        
        if is_count:
            return {
                "measurement_type": "count",
                "typical_serving": 1,
                "serving_unit": "pieces",
                "explanation": f"Typically measured as individual pieces",
                "alternative_names": [food_name]
            }
        else:
            return {
                "measurement_type": "grams",
                "typical_serving": 100,
                "serving_unit": "grams",
                "explanation": f"Typically measured by weight",
                "alternative_names": [food_name]
            }

    async def get_nutrition_data(self, food_name: str, quantity: float, unit: str) -> Dict[str, Any]:
        """Get nutrition data from Nutritionix API"""
        if not NUTRITIONIX_APP_ID or not NUTRITIONIX_APP_KEY:
            return self.mock_nutrition_data(food_name, quantity, unit)
        
        try:
            headers = {
                'x-app-id': NUTRITIONIX_APP_ID,
                'x-app-key': NUTRITIONIX_APP_KEY,
                'Content-Type': 'application/json'
            }
            
            # Format query for API
            if unit == "pieces":
                query = f"{quantity} {food_name}"
            else:
                query = f"{quantity}g {food_name}"
            
            data = {
                "query": query
            }
            
            response = requests.post(
                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('foods'):
                    food = result['foods'][0]
                    return {
                        "food_name": food.get('food_name', food_name),
                        "serving_qty": food.get('serving_qty', quantity),
                        "serving_unit": food.get('serving_unit', unit),
                        "calories": food.get('nf_calories', 0),
                        "protein": food.get('nf_protein', 0),
                        "fat": food.get('nf_total_fat', 0),
                        "carbs": food.get('nf_total_carbohydrate', 0),
                        "fiber": food.get('nf_dietary_fiber', 0),
                        "sugar": food.get('nf_sugars', 0),
                        "sodium": food.get('nf_sodium', 0),
                    }
            
            # If API fails, try alternative name or return mock data
            return self.mock_nutrition_data(food_name, quantity, unit)
            
        except Exception as e:
            logger.error(f"Nutrition API error: {e}")
            return self.mock_nutrition_data(food_name, quantity, unit)
    
    def mock_nutrition_data(self, food_name: str, quantity: float, unit: str) -> Dict[str, Any]:
        """Mock nutrition data when API is not available"""
        # Basic estimates per 100g or per piece
        mock_data = {
            "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14},
            "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23},
            "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
            "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
        }
        
        base_nutrition = mock_data.get(food_name.lower(), 
                                     {"calories": 100, "protein": 2, "fat": 1, "carbs": 20})
        
        # Scale based on quantity
        multiplier = quantity / 100 if unit == "grams" else quantity
        
        return {
            "food_name": food_name,
            "serving_qty": quantity,
            "serving_unit": unit,
            "calories": round(base_nutrition["calories"] * multiplier, 1),
            "protein": round(base_nutrition["protein"] * multiplier, 1),
            "fat": round(base_nutrition["fat"] * multiplier, 1),
            "carbs": round(base_nutrition["carbs"] * multiplier, 1),
            "fiber": round(2 * multiplier, 1),
            "sugar": round(5 * multiplier, 1),
            "sodium": round(50 * multiplier, 1),
        }

# Initialize service
food_service = FoodDetectionService()

@app.get("/")
async def root():
    return {"message": "AI Food Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "openai_configured": OPENAI_API_KEY is not None,
        "nutritionix_configured": NUTRITIONIX_APP_ID is not None and NUTRITIONIX_APP_KEY is not None
    }

@app.post("/detect-food")
async def detect_food(file: UploadFile = File(...)):
    """Main endpoint for food detection and nutrition analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Detect food items
        detections = food_service.detect_food_items(image_np)
        
        if not detections:
            return JSONResponse({
                "success": True,
                "message": "No food items detected",
                "results": []
            })
        
        # Process each detection
        results = []
        for detection in detections:
            food_name = detection["class_name"]
            
            # Get portion recommendation
            portion_info = await food_service.determine_portion_type(food_name)
            
            # Get nutrition data
            nutrition_data = await food_service.get_nutrition_data(
                food_name, 
                portion_info["typical_serving"], 
                portion_info["serving_unit"]
            )
            
            # Combine all information
            result = {
                "food_name": food_name,
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "portion_recommendation": portion_info,
                "nutrition": nutrition_data,
                "detection_count": detection.get("count", 1)
            }
            
            results.append(result)
        
        return JSONResponse({
            "success": True,
            "message": f"Detected {len(results)} food items",
            "results": results,
            "image_info": {
                "width": image.width,
                "height": image.height
            }
        })
        
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/custom-portion")
async def update_nutrition_with_custom_portion(request_data: dict):
    """Update nutrition data based on user's custom portion input"""
    try:
        food_name = request_data.get("food_name")
        custom_quantity = request_data.get("quantity")
        custom_unit = request_data.get("unit")
        
        if not all([food_name, custom_quantity, custom_unit]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Get updated nutrition data
        nutrition_data = await food_service.get_nutrition_data(
            food_name, custom_quantity, custom_unit
        )
        
        return JSONResponse({
            "success": True,
            "nutrition": nutrition_data
        })
        
    except Exception as e:
        logger.error(f"Custom portion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
