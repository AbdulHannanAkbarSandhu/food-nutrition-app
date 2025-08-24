# Food Detection & Nutrition Analysis System

A production-ready full-stack application that detects food items from images and provides detailed nutritional analysis. Built with custom YOLOv8 object detection, React frontend, and FastAPI backend.

## Live Demo

- **Frontend**: [https://food-detection-frontend-m8n8j92rr.vercel.app](https://food-detection-frontend-m8n8j92rr.vercel.app)
- **Backend API**: Deployed on Azure Container Apps

## System Architecture

### Frontend
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS
- **Deployment**: Vercel with serverless API functions
- **Features**: Drag-and-drop upload, 3-step workflow, responsive design

### Backend  
- **Framework**: FastAPI (Python)
- **ML Model**: Custom YOLOv8 converted to TorchScript
- **Deployment**: Azure Container Apps with auto-scaling
- **APIs**: OpenAI GPT-3.5-turbo, Nutritionix nutrition database

### Model
- **Architecture**: YOLOv8n fine-tuned for food detection
- **Classes**: 24 food categories (excluding problematic class)
- **Accuracy**: 90%+ detection accuracy, mAP50: 0.725
- **Format**: TorchScript for deployment stability

## Project Structure

```
food-nutrition-app/
├── food-detection-frontend/          # React frontend
│   ├── api/                         # Vercel serverless functions
│   │   ├── detect-food.js          # Food detection API proxy
│   │   ├── get-portion-recommendation.js
│   │   └── get-nutrition-data.js
│   ├── src/
│   │   ├── WorkflowApp.jsx         # Main application component
│   │   └── main.jsx
│   ├── .env.example                # Environment template
│   └── package.json
├── foodie/                         # Training dataset & results
│   ├── data.yaml                   # Dataset configuration
│   ├── train/images & labels/
│   ├── valid/images & labels/
│   ├── test/images & labels/
│   └── yolo_results/              # Training outputs
├── workflow-app-torchscript.py     # FastAPI backend
├── workflow-Dockerfile-torchscript # Docker configuration
├── workflow-requirements-torchscript.txt
├── best.torchscript               # Trained model (TorchScript)
└── training and validation.ipynb  # Model training notebook
```

## Model Training Results

### Final Metrics (Class 5 Excluded)
- **mAP50-95**: 0.601, baseline model had less than 0.5.
- **mAP50**: 0.725 , base line model barely had 0.5.
- **Precision**: 0.823
- **Recall**: 0.627

### Key Improvements Achieved
- **Hashbrown**: Precision 0.80 → 0.93, Recall 0.75 → 0.87
- **French Fries**: Precision 0.76 → 0.94, Recall 0.73 → 0.80
- **Chinese Sausage**: Precision 0.81 → 0.97, Recall 0.75 → 0.85
- **Reduced confusion** between visually similar classes (curry/drinks, nuggets/fried chicken)

### Training Methodology
1. **Baseline Training**: YOLOv8n with default settings
2. **Data Augmentation**: Custom pipeline addressing class imbalance
3. **Loss Rebalancing**: Increased classification loss weight for rare classes
4. **Optimizer Selection**: AdamW over SGD for better convergence
5. **Class Exclusion**: Removed problematic "String Bean Chicken Breast" class

## Food Categories Detected

Beijing Beef, Chow Mein, Fried Rice, Hashbrown, Kung Pao Chicken, Burger, Carrot Eggs, Chicken Waffle, Chicken Nuggets, Chinese Cabbage, Chinese Sausage, Curry, Drink, French Fries, Fried Chicken, Fried Dumplings, Fried Eggs, Ketchup, Mango Chicken Pocket, Mung Bean Sprouts, Rice, Tostitos Cheese Dip Sauce, Water Spinach

## Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Azure account (for backend deployment)
- Vercel account (for frontend deployment)
- install "ultralytics==8.3.177" (for model training)

### Local Development

#### Backend Setup
```bash
# Install dependencies
pip install -r workflow-requirements-torchscript.txt

# Run FastAPI server
python workflow-app-torchscript.py
```

#### Frontend Setup
```bash
cd food-detection-frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local with your backend URL

# Start development server
npm run dev
```

## Environment Variables

### Frontend (.env.local)
```
AZURE_BACKEND_URL=https://your-backend-url.azurecontainerapps.io
```

### Backend
Set in your deployment environment:
- `OPENAI_API_KEY`: For portion size recommendations
- `NUTRITIONIX_APP_ID`: For nutrition data
- `NUTRITIONIX_APP_KEY`: For nutrition data

## Deployment

### Backend (Azure Container Apps)
```bash
# Build and push Docker image
docker build -f workflow-Dockerfile-torchscript -t food-detection .
docker tag food-detection your-registry.azurecr.io/food-detection
docker push your-registry.azurecr.io/food-detection

# Deploy to Azure Container Apps
az containerapp create \
  --name food-detection-app \
  --resource-group your-rg \
  --image your-registry.azurecr.io/food-detection:latest
```

### Frontend (Vercel)
```bash
cd food-detection-frontend

# Set environment variables in Vercel dashboard
# Deploy
vercel --prod
```

## API Endpoints

### Food Detection
- **POST** `/detect-food`
- **Input**: Multipart form data with image file
- **Output**: Detected food items with confidence scores and bounding boxes

### Portion Recommendations  
- **POST** `/get-portion-recommendation`
- **Input**: JSON with detected food items
- **Output**: AI-generated portion size recommendations

### Nutrition Analysis
- **POST** `/get-nutrition-data`  
- **Input**: JSON with food items and portions
- **Output**: Complete nutritional breakdown

## Training Environment

- **Framework**: Ultralytics YOLOv8
- **Optimizer**: AdamW with cosine learning rate schedule
- **Augmentations**: Mosaic, Mixup, Copy-Paste, Erasing
- **Hardware**: Google Colab GPU (Tesla T4/V100)
- **Training Time**: ~4 hours for final model

## Key Technical Decisions

### Model Deployment
- **TorchScript conversion** eliminated Ultralytics dependency issues
- **Azure Container Apps** provided better stability than Container Instances
- **Auto-scaling** handles variable load efficiently

### Frontend Security
- **Serverless proxy functions** hide backend URLs from client
- **Environment variable validation** prevents configuration errors
- **CORS handling** through Vercel API routes

### Class Handling
- **Dynamic exclusion** of problematic classes without dataset modification
- **Confidence thresholding** for reliable predictions
- **Multiple detection grouping** for complex images

## Performance Characteristics

- **Detection Speed**: ~2-3 seconds per image
- **Model Size**: 12.4MB (TorchScript format)
- **Memory Usage**: ~500MB backend RAM
- **Throughput**: 10+ concurrent requests supported

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Ultralytics YOLOv8 framework
- Roboflow dataset platform
- OpenAI API for intelligent recommendations
- Nutritionix database for nutrition data
