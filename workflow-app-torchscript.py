from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn
import json
import requests
import os
from typing import List, Dict, Any
import logging
from pydantic import BaseModel
import torch
import tempfile
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Detection Workflow API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID", "")
NUTRITIONIX_APP_KEY = os.getenv("NUTRITIONIX_APP_KEY", "")

# Request models
class PortionRequest(BaseModel):
    food_name: str
    
class NutritionRequest(BaseModel):
    food_name: str
    quantity: float
    unit: str

class TorchScriptYOLODetector:
    """TorchScript YOLO detection using converted model"""
    def __init__(self, model_path='best.torchscript', conf_threshold=0.5, iou_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        
        # Your 24 food classes from the conversion
        self.class_names = {
            0: 'Beijing Beef', 1: 'Chow Mein', 2: 'Fried Rice', 3: 'Hashbrown', 
            4: 'Kung Pao Chicken', 5: 'String Bean Chicken Breast', 6: 'burger', 
            7: 'carrot_eggs', 8: 'chicken waffle', 9: 'chicken_nuggets', 
            10: 'chinese_cabbage', 11: 'chinese_sausage', 12: 'curry', 13: 'drink', 
            14: 'french fries', 15: 'fried_chicken', 16: 'fried_dumplings', 
            17: 'fried_eggs', 18: 'ketchup', 19: 'mango chicken pocket', 
            20: 'mung_bean_sprouts', 21: 'rice', 22: 'tostitos cheese dip sauce', 
            23: 'water_spinach'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load TorchScript model"""
        try:
            logger.info(f"Loading TorchScript model from {self.model_path}")
            
            # Load TorchScript model (no Ultralytics needed)
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model.eval()
            
            logger.info(f"✅ TorchScript model loaded with {len(self.class_names)} food classes")
            logger.info(f"📋 Available classes: {list(self.class_names.values())}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load TorchScript model: {e}")
            raise
    
    def preprocess_image(self, image_np):
        """Preprocess image for YOLO inference"""
        original_shape = image_np.shape[:2]  # (height, width)
        
        # Convert BGR to RGB if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_np
        
        # Resize to 640x640
        img_resized = cv2.resize(img_rgb, (640, 640))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor: HWC -> CHW
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
        
        # Add batch dimension
        img_batch = img_tensor.unsqueeze(0)
        
        return img_batch, original_shape
    
    def postprocess_predictions(self, predictions, original_shape):
        """Post-process TorchScript model predictions"""
        detections = []
        
        # TorchScript model output: [1, 28, 8400]
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension -> [28, 8400]
        
        # Transpose: [28, 8400] -> [8400, 28]
        predictions = predictions.transpose(0, 1)
        
        for detection in predictions:
            # Extract bbox and class scores
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]  # 24 class scores
            
            # Get best class
            max_score = torch.max(class_scores)
            
            if max_score > self.conf_threshold:
                class_id = torch.argmax(class_scores).item()
                
                # Convert from center format to corner format
                x1 = float(x_center - width / 2)
                y1 = float(y_center - height / 2)
                x2 = float(x_center + width / 2)
                y2 = float(y_center + height / 2)
                
                # Scale back to original image size
                orig_height, orig_width = original_shape
                x1 = (x1 / 640) * orig_width
                y1 = (y1 / 640) * orig_height
                x2 = (x2 / 640) * orig_width
                y2 = (y2 / 640) * orig_height
                
                # Clamp to image bounds
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))
                
                detection_dict = {
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                    'confidence': float(max_score),
                    'bbox': [x1, y1, x2, y2]
                }
                
                detections.append(detection_dict)
        
        return detections
    
    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return detections
        
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections 
                if calculate_iou(current['bbox'], det['bbox']) < self.iou_threshold
            ]
        
        return keep
    
    def group_by_class_and_select_best(self, detections):
        """Group detections by class and select highest confidence, noting multiple portions"""
        if not detections:
            return []
        
        # Group by class name
        class_groups = defaultdict(list)
        for detection in detections:
            class_groups[detection['class_name']].append(detection)
        
        final_detections = []
        for class_name, class_detections in class_groups.items():
            # Sort by confidence (highest first)
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Take the highest confidence detection
            best_detection = class_detections[0]
            
            # Note if multiple portions were detected
            detected_count = len(class_detections)
            multiple_portions = detected_count > 1
            
            # Create final detection with portion info
            final_detection = {
                "class": class_name,
                "conf": best_detection['confidence'],
                "detected_count": detected_count,
                "multiple_portions": multiple_portions,
                "mode": "grams",  # Default mode
                "user_value": 100,  # Default value, will be updated by ChatGPT
                "x1": int(best_detection['bbox'][0]),
                "y1": int(best_detection['bbox'][1]),
                "x2": int(best_detection['bbox'][2]),
                "y2": int(best_detection['bbox'][3])
            }
            
            final_detections.append(final_detection)
            
            # Log the detection
            if multiple_portions:
                logger.info(f"🍽️  {class_name}: {detected_count} portions detected (showing highest confidence: {best_detection['confidence']:.3f})")
            else:
                logger.info(f"🍽️  {class_name}: single portion detected (confidence: {best_detection['confidence']:.3f})")
        
        return final_detections
    
    def detect_food(self, image_np):
        """Main detection function"""
        try:
            # Preprocess image
            img_tensor, original_shape = self.preprocess_image(image_np)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
            
            # Post-process predictions
            raw_detections = self.postprocess_predictions(predictions, original_shape)
            
            # Apply NMS to remove overlapping detections
            nms_detections = self.apply_nms(raw_detections)
            
            # Group by class and select best, noting multiple portions
            final_detections = self.group_by_class_and_select_best(nms_detections)
            
            logger.info(f"🎯 TorchScript Detection Summary: {len(raw_detections)} raw -> {len(nms_detections)} after NMS -> {len(final_detections)} final foods")
            
            return final_detections
            
        except Exception as e:
            logger.error(f"❌ TorchScript detection failed: {e}")
            return []

class WorkflowFoodDetectionService:
    def __init__(self):
        self.confidence_threshold = 0.5  # Configurable confidence threshold
        
        # Initialize TorchScript YOLO detector
        try:
            self.yolo_detector = TorchScriptYOLODetector(
                model_path='best.torchscript', 
                conf_threshold=self.confidence_threshold,
                iou_threshold=0.45
            )
            logger.info("✅ TorchScript YOLO detector initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TorchScript YOLO detector: {e}")
            self.yolo_detector = None
    
    def real_yolo_detection(self, image: np.ndarray) -> List[Dict]:
        """Real YOLO detection using TorchScript model"""
        if self.yolo_detector is None:
            logger.error("TorchScript detector not available, falling back to mock")
            return self.mock_yolo_detection(image)
        
        try:
            detections = self.yolo_detector.detect_food(image)
            
            if detections:
                logger.info(f"✅ TorchScript YOLO detected {len(detections)} unique food items")
                for detection in detections:
                    portion_note = f" (multiple portions)" if detection.get('multiple_portions', False) else ""
                    logger.info(f"   - {detection['class']}: confidence {detection['conf']:.3f}{portion_note}")
            else:
                logger.info("ℹ️  No food items detected above confidence threshold")
            
            return detections
            
        except Exception as e:
            logger.error(f"TorchScript YOLO detection error: {e}")
            return self.mock_yolo_detection(image)
    
    def mock_yolo_detection(self, image: np.ndarray) -> List[Dict]:
        """Fallback mock detection"""
        try:
            height, width = image.shape[:2]
            
            mock_detections = [
                {
                    "class": "rice",
                    "conf": 0.9936683714389801,
                    "detected_count": 1,
                    "multiple_portions": False,
                    "mode": "grams",
                    "user_value": 100,
                    "x1": 199,
                    "y1": 92, 
                    "x2": 1334,
                    "y2": 901
                }
            ]
            
            logger.info("⚠️  Using mock detection (TorchScript not available)")
            return mock_detections
            
        except Exception as e:
            logger.error(f"Mock detection error: {e}")
            return []
    
    async def get_chatgpt_portion_recommendation(self, food_name: str) -> Dict[str, Any]:
        """Get ONLY portion recommendation from ChatGPT"""
        if not OPENAI_API_KEY:
            return self.fallback_portion_recommendation(food_name)
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are a nutrition expert. For the food item "{food_name}", provide a portion size recommendation.

ONLY respond with a JSON object in this exact format:
{{
    "measurement_type": "grams" or "count",
    "recommended_amount": <number>,
    "unit": "grams" or "pieces" or "cups",
    "explanation": "Brief explanation why this measurement and amount is recommended for {food_name}"
}}

Guidelines:
- For rice, pasta, grains: use grams (typical serving 150-200g)
- For fruits like apple, banana: use count (1 piece)
- For vegetables: use grams (typical serving 100-150g)
- For meat/protein: use grams (typical serving 100-120g)

Be specific about the amount and give a clear, helpful explanation."""
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                try:
                    # Try to parse JSON response
                    recommendation = json.loads(content)
                    return {
                        "success": True,
                        "recommendation": recommendation,
                        "source": "chatgpt"
                    }
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract information
                    return self.parse_chatgpt_text_response(content, food_name)
            else:
                logger.error(f"ChatGPT API error: {response.status_code}")
                return self.fallback_portion_recommendation(food_name)
                
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            return self.fallback_portion_recommendation(food_name)
    
    def parse_chatgpt_text_response(self, content: str, food_name: str) -> Dict[str, Any]:
        """Parse ChatGPT response even if not perfect JSON"""
        # Extract measurement type
        if "grams" in content.lower() or "g" in content.lower():
            measurement_type = "grams"
            unit = "grams"
            # Try to extract number
            import re
            numbers = re.findall(r'\d+', content)
            recommended_amount = int(numbers[0]) if numbers else 100
        else:
            measurement_type = "count"
            unit = "pieces"
            recommended_amount = 1
        
        return {
            "success": True,
            "recommendation": {
                "measurement_type": measurement_type,
                "recommended_amount": recommended_amount,
                "unit": unit,
                "explanation": content[:200]  # First 200 chars as explanation
            },
            "source": "chatgpt_parsed"
        }
    
    def fallback_portion_recommendation(self, food_name: str) -> Dict[str, Any]:
        """Fallback when ChatGPT is not available"""
        # Smart fallback based on food type
        count_foods = {
            "apple": (1, "pieces", "Apples are typically consumed as whole fruits"),
            "banana": (1, "pieces", "Bananas are typically consumed as individual fruits"),
            "orange": (1, "pieces", "Oranges are typically consumed as whole fruits"),
            "egg": (2, "pieces", "A typical serving is 1-2 eggs")
        }
        
        weight_foods = {
            "rice": (150, "grams", "A standard serving of cooked rice is 150-200 grams"),
            "chicken": (120, "grams", "A typical serving of chicken is 100-120 grams"),
            "vegetables": (100, "grams", "A standard serving of vegetables is 100-150 grams"),
            "pasta": (100, "grams", "A typical serving of dry pasta is 80-100 grams"),
            "bread": (30, "grams", "One slice of bread is typically 25-30 grams")
        }
        
        food_lower = food_name.lower()
        
        if food_lower in count_foods:
            amount, unit, explanation = count_foods[food_lower]
            measurement_type = "count"
        elif food_lower in weight_foods:
            amount, unit, explanation = weight_foods[food_lower]
            measurement_type = "grams"
        else:
            # Default fallback
            measurement_type = "grams"
            amount = 100
            unit = "grams"
            explanation = f"Standard serving size recommended for {food_name}"
        
        return {
            "success": True,
            "recommendation": {
                "measurement_type": measurement_type,
                "recommended_amount": amount,
                "unit": unit,
                "explanation": explanation
            },
            "source": "fallback"
        }
    
    async def get_nutritionix_data(self, food_name: str, quantity: float, unit: str) -> Dict[str, Any]:
        """Get precise nutrition data from Nutritionix"""
        if not NUTRITIONIX_APP_ID or not NUTRITIONIX_APP_KEY:
            return self.fallback_nutrition_data(food_name, quantity, unit)
        
        try:
            headers = {
                'x-app-id': NUTRITIONIX_APP_ID,
                'x-app-key': NUTRITIONIX_APP_KEY,
                'Content-Type': 'application/json'
            }
            
            # Format query for Nutritionix API
            if unit in ["pieces", "piece"]:
                query = f"{quantity} {food_name}"
            elif unit == "grams":
                query = f"{quantity}g {food_name}"
            elif unit == "cups":
                query = f"{quantity} cup {food_name}"
            else:
                query = f"{quantity} {unit} {food_name}"
            
            data = {"query": query}
            
            response = requests.post(
                "https://trackapi.nutritionix.com/v2/natural/nutrients",
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('foods') and len(result['foods']) > 0:
                    food = result['foods'][0]
                    return {
                        "success": True,
                        "nutrition": {
                            "food_name": food.get('food_name', food_name),
                            "serving_qty": quantity,
                            "serving_unit": unit,
                            "calories": round(food.get('nf_calories', 0)),
                            "protein": round(food.get('nf_protein', 0), 1),
                            "total_fat": round(food.get('nf_total_fat', 0), 1),
                            "saturated_fat": round(food.get('nf_saturated_fat', 0), 1),
                            "carbohydrates": round(food.get('nf_total_carbohydrate', 0), 1),
                            "dietary_fiber": round(food.get('nf_dietary_fiber', 0), 1),
                            "sugars": round(food.get('nf_sugars', 0), 1),
                            "sodium": round(food.get('nf_sodium', 0), 1),
                            "potassium": round(food.get('nf_potassium', 0), 1)
                        },
                        "source": "nutritionix"
                    }
            
            # Fallback if API fails
            return self.fallback_nutrition_data(food_name, quantity, unit)
            
        except Exception as e:
            logger.error(f"Nutritionix API error: {e}")
            return self.fallback_nutrition_data(food_name, quantity, unit)
    
    def fallback_nutrition_data(self, food_name: str, quantity: float, unit: str) -> Dict[str, Any]:
        """Fallback nutrition data when API is not available"""
        # Enhanced nutrition database per 100g
        nutrition_db = {
            "rice": {"calories": 130, "protein": 2.7, "total_fat": 0.3, "carbohydrates": 28, "dietary_fiber": 0.4, "sugars": 0.1, "sodium": 5, "potassium": 35},
            "chicken": {"calories": 165, "protein": 31, "total_fat": 3.6, "carbohydrates": 0, "dietary_fiber": 0, "sugars": 0, "sodium": 74, "potassium": 256},
            "vegetables": {"calories": 25, "protein": 2, "total_fat": 0.2, "carbohydrates": 5, "dietary_fiber": 2.5, "sugars": 2, "sodium": 10, "potassium": 150},
            "apple": {"calories": 52, "protein": 0.3, "total_fat": 0.2, "carbohydrates": 14, "dietary_fiber": 2.4, "sugars": 10, "sodium": 1, "potassium": 107},
            "banana": {"calories": 89, "protein": 1.1, "total_fat": 0.3, "carbohydrates": 23, "dietary_fiber": 2.6, "sugars": 12, "sodium": 1, "potassium": 358}
        }
        
        base_nutrition = nutrition_db.get(food_name.lower(), nutrition_db["vegetables"])
        
        # Calculate multiplier based on unit
        if unit in ["pieces", "piece"]:
            # Estimate weight per piece
            piece_weights = {"apple": 150, "banana": 120, "orange": 180}
            piece_weight = piece_weights.get(food_name.lower(), 100)
            multiplier = (quantity * piece_weight) / 100
        elif unit == "grams":
            multiplier = quantity / 100
        elif unit == "cups":
            # Estimate grams per cup
            cup_weights = {"rice": 150, "vegetables": 80}
            cup_weight = cup_weights.get(food_name.lower(), 100)
            multiplier = (quantity * cup_weight) / 100
        else:
            multiplier = quantity / 100
        
        return {
            "success": True,
            "nutrition": {
                "food_name": food_name,
                "serving_qty": quantity,
                "serving_unit": unit,
                "calories": round(base_nutrition["calories"] * multiplier),
                "protein": round(base_nutrition["protein"] * multiplier, 1),
                "total_fat": round(base_nutrition["total_fat"] * multiplier, 1),
                "saturated_fat": round(base_nutrition["total_fat"] * 0.3 * multiplier, 1),
                "carbohydrates": round(base_nutrition["carbohydrates"] * multiplier, 1),
                "dietary_fiber": round(base_nutrition["dietary_fiber"] * multiplier, 1),
                "sugars": round(base_nutrition["sugars"] * multiplier, 1),
                "sodium": round(base_nutrition["sodium"] * multiplier, 1),
                "potassium": round(base_nutrition["potassium"] * multiplier, 1)
            },
            "source": "estimated"
        }

# Initialize service
food_service = WorkflowFoodDetectionService()

@app.get("/")
async def root():
    return {"message": "Food Detection Workflow API", "status": "running", "version": "3.0.0"}

@app.get("/health")
async def health():
    yolo_status = "✅ TorchScript YOLO Ready" if food_service.yolo_detector is not None else "❌ YOLO Not Available (using mock)"
    
    return {
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "yolo_model": yolo_status,
        "confidence_threshold": food_service.confidence_threshold,
        "openai_configured": bool(OPENAI_API_KEY),
        "nutritionix_configured": bool(NUTRITIONIX_APP_ID and NUTRITIONIX_APP_KEY),
        "version": "3.0.0",
        "workflow": "torchscript_yolo_detection -> chatgpt_recommendation -> user_choice -> nutritionix_data"
    }

@app.post("/detect-food")
async def detect_food(file: UploadFile = File(...)):
    """Step 1: Food detection using TorchScript YOLO model - returns detected items with bounding boxes"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Get detections using TorchScript YOLO
        detections = food_service.real_yolo_detection(image_np)
        
        if not detections:
            return {
                "success": True,
                "message": "No food items detected above confidence threshold",
                "detections": [],
                "model_info": {
                    "confidence_threshold": food_service.confidence_threshold,
                    "using_torchscript_yolo": food_service.yolo_detector is not None
                },
                "next_step": "none"
            }
        
        # Add portion information to message
        total_portions = sum(det.get('detected_count', 1) for det in detections)
        unique_foods = len(detections)
        
        if total_portions > unique_foods:
            message = f"Detected {unique_foods} unique food types with {total_portions} total portions"
        else:
            message = f"Detected {unique_foods} food items"
        
        return {
            "success": True,
            "message": message,
            "detections": detections,
            "image_info": {
                "width": image.width,
                "height": image.height
            },
            "model_info": {
                "confidence_threshold": food_service.confidence_threshold,
                "using_torchscript_yolo": food_service.yolo_detector is not None,
                "total_portions_detected": total_portions
            },
            "next_step": "get_portion_recommendations"
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-portion-recommendation")
async def get_portion_recommendation(request: PortionRequest):
    """Step 2: Get ChatGPT portion recommendation for a detected food item"""
    try:
        recommendation = await food_service.get_chatgpt_portion_recommendation(request.food_name)

        return {
            "success": True,
            "food_name": request.food_name,
            "chatgpt_recommendation": recommendation,
            "next_step": "get_nutrition_data"
        }

    except Exception as e:
        logger.error(f"Portion recommendation error: {e}")
        raise

@app.post("/get-nutrition-data")
async def get_nutrition_data(request: NutritionRequest):
    """Step 3: Get nutrition data from Nutritionix based on user's final portion choice"""
    try:
        nutrition_data = await food_service.get_nutritionix_data(
            request.food_name, 
            request.quantity, 
            request.unit
        )
        
        return {
            "success": True,
            "food_name": request.food_name,
            "portion": {
                "quantity": request.quantity,
                "unit": request.unit
            },
            "nutrition_data": nutrition_data,
            "next_step": "display_results"
        }
        
    except Exception as e:
        logger.error(f"Nutrition data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint to update confidence threshold
@app.post("/update-confidence-threshold")
async def update_confidence_threshold(threshold: float):
    """Update the confidence threshold for YOLO detection"""
    try:
        if not 0.1 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 1.0")
        
        food_service.confidence_threshold = threshold
        if food_service.yolo_detector:
            food_service.yolo_detector.conf_threshold = threshold
        
        return {
            "success": True,
            "message": f"Confidence threshold updated to {threshold}",
            "new_threshold": threshold
        }
        
    except Exception as e:
        logger.error(f"Threshold update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded TorchScript model"""
    try:
        if food_service.yolo_detector and food_service.yolo_detector.class_names:
            return {
                "model_loaded": True,
                "model_path": food_service.yolo_detector.model_path,
                "model_type": "TorchScript",
                "confidence_threshold": food_service.confidence_threshold,
                "iou_threshold": food_service.yolo_detector.iou_threshold,
                "total_classes": len(food_service.yolo_detector.class_names),
                "food_classes": list(food_service.yolo_detector.class_names.values()),
                "class_mapping": food_service.yolo_detector.class_names
            }
        else:
            return {
                "model_loaded": False,
                "message": "TorchScript model not loaded, using mock detection"
            }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/get-portion-recommendation")
async def get_portion_recommendation(request: PortionRequest):
    """Step 2: Get ChatGPT portion recommendation for a detected food item"""
    try:
        recommendation = await food_service.get_chatgpt_portion_recommendation(request.food_name)
        
        return {
            "success": True,
            "food_name": request.food_name,
            "chatgpt_recommendation": recommendation,
            "next_step": "get_nutrition_data"
        }
        
    except Exception as e:
        logger.error(f"Portion recommendation error: {e}")
        raise
