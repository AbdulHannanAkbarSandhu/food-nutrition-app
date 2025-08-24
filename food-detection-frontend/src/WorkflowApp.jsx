import React, { useState, useRef, useCallback } from 'react';
import { Upload, Camera, Loader2, AlertCircle, CheckCircle, Edit3, X, Apple, BarChart3, Brain, Target } from 'lucide-react';

// Update with your new workflow API URL
const API_BASE_URL = '/api';

const WorkflowFoodDetectionApp = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [currentStep, setCurrentStep] = useState(1); // 1: Upload, 2: Detection, 3: Recommendation, 4: Results
  const [detections, setDetections] = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const [finalResults, setFinalResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [processingFood, setProcessingFood] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
      // Reset workflow
      setCurrentStep(1);
      setDetections([]);
      setRecommendations({});
      setFinalResults([]);
      setError(null);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleImageUpload(file);
  }, [handleImageUpload]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  // Step 1: Food Detection
  const detectFood = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setCurrentStep(2);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await fetch(`${API_BASE_URL}/detect-food`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success && data.detections.length > 0) {
        setDetections(data.detections);
        setCurrentStep(3);
        // Automatically get recommendations for all detected foods
        await getRecommendationsForAll(data.detections);
      } else {
        setError("No food items detected in the image");
        setCurrentStep(1);
      }
      
    } catch (err) {
      setError(`Detection failed: ${err.message}`);
      setCurrentStep(1);
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Get ChatGPT Recommendations for All Detected Foods
  const getRecommendationsForAll = async (detectedFoods) => {
    setLoading(true);
    const newRecommendations = {};

    try {
      for (const detection of detectedFoods) {
        setProcessingFood(detection.class);
        
        const response = await fetch(`${API_BASE_URL}/get-portion-recommendation`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            food_name: detection.class
          }),
        });

        if (response.ok) {
          const data = await response.json();
          newRecommendations[detection.class] = {
            ...data.chatgpt_recommendation,
            detection: detection
          };
        }
      }
      
      setRecommendations(newRecommendations);
      setProcessingFood(null);
      
    } catch (err) {
      setError(`Recommendation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Step 3: Get Final Nutrition Data
  const getFinalNutrition = async (foodName, quantity, unit, isCustom = false) => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-nutrition-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          food_name: foodName,
          quantity: parseFloat(quantity),
          unit: unit
        }),
      });

      if (response.ok) {
        const data = await response.json();
        
        // Update final results
        setFinalResults(prev => {
          const updated = prev.filter(item => item.food_name !== foodName);
          updated.push({
            food_name: foodName,
            portion: data.portion,
            nutrition: data.nutrition_data.nutrition,
            detection: recommendations[foodName]?.detection,
            recommendation_used: !isCustom,
            source: data.nutrition_data.source
          });
          return updated;
        });
        
        // Check if all foods have final results
        if (finalResults.length + 1 >= detections.length) {
          setCurrentStep(4);
        }
      }
    } catch (err) {
      setError(`Nutrition lookup failed: ${err.message}`);
    }
  };

  // Accept ChatGPT Recommendation
  const acceptRecommendation = async (foodName) => {
    const rec = recommendations[foodName];
    if (rec && rec.recommendation) {
      await getFinalNutrition(
        foodName, 
        rec.recommendation.recommended_amount,
        rec.recommendation.unit,
        false
      );
    }
  };

  // Custom Portion Input Component
  const CustomPortionInput = ({ foodName, recommendation, onSubmit, onCancel }) => {
    const [quantity, setQuantity] = useState(recommendation?.recommended_amount || 100);
    const [unit, setUnit] = useState(recommendation?.unit || 'grams');

    const handleSubmit = () => {
      onSubmit(foodName, quantity, unit, true);
    };

    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-3">
        <h4 className="font-medium text-blue-800 mb-3">Custom Portion for {foodName}</h4>
        <div className="flex items-center gap-3">
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            className="w-24 px-3 py-2 border border-blue-300 rounded-lg text-center"
            min="0.1"
            step="0.1"
          />
          <select
            value={unit}
            onChange={(e) => setUnit(e.target.value)}
            className="px-3 py-2 border border-blue-300 rounded-lg"
          >
            <option value="grams">grams</option>
            <option value="pieces">pieces</option>
            <option value="cups">cups</option>
            <option value="ounces">ounces</option>
          </select>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <CheckCircle className="w-4 h-4" />
            Use This
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
        </div>
      </div>
    );
  };

  const [editingPortion, setEditingPortion] = useState({});

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Apple className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                AI Food Detection & Nutrition Workflow
              </h1>
              <p className="text-gray-600 mt-1">
                3-Step Process: Detection → AI Recommendations → Nutrition Analysis
              </p>
            </div>
          </div>
          
          {/* Progress Steps */}
          <div className="mt-6 flex items-center justify-center">
            <div className="flex items-center space-x-8">
              {[
                { step: 1, label: "Upload & Detect", icon: Camera, active: currentStep >= 1 },
                { step: 2, label: "AI Recommendations", icon: Brain, active: currentStep >= 3 },
                { step: 3, label: "Final Results", icon: Target, active: currentStep >= 4 }
              ].map(({ step, label, icon: Icon, active }) => (
                <div key={step} className={`flex items-center ${active ? 'text-blue-600' : 'text-gray-400'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${active ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="ml-2 font-medium">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Step 1: Upload and Detection */}
        {currentStep <= 2 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Camera className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-semibold text-gray-800">
                {currentStep === 1 ? "Upload Image" : "Detecting Food Items..."}
              </h2>
            </div>
            
            {currentStep === 1 && (
              <>
                <div
                  className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-500 hover:bg-blue-50 transition-all cursor-pointer group"
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {imagePreview ? (
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setImage(null);
                          setImagePreview(null);
                        }}
                        className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 hover:bg-red-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <div className="group-hover:scale-105 transition-transform">
                      <Camera className="w-16 h-16 text-blue-600 mx-auto mb-4" />
                      <h3 className="text-lg font-medium text-gray-700 mb-2">
                        Drop your food image here
                      </h3>
                      <p className="text-gray-500">or click to browse files</p>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files[0])}
                  className="hidden"
                />

                <button
                  onClick={detectFood}
                  disabled={!image || loading}
                  className="w-full mt-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-3"
                >
                  <BarChart3 className="w-5 h-5" />
                  Start Food Detection
                </button>
              </>
            )}

            {currentStep === 2 && loading && (
              <div className="text-center py-8">
                <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
                <p className="text-lg text-gray-600">Analyzing your image...</p>
              </div>
            )}
          </div>
        )}

        {/* Step 2: ChatGPT Recommendations */}
        {currentStep === 3 && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Brain className="w-6 h-6 text-green-600" />
              <h2 className="text-2xl font-semibold text-gray-800">AI Portion Recommendations</h2>
            </div>

            {loading && (
              <div className="text-center py-4 mb-6">
                <Loader2 className="w-8 h-8 text-green-600 animate-spin mx-auto mb-2" />
                <p className="text-green-600">Getting AI recommendations for {processingFood}...</p>
              </div>
            )}

            <div className="space-y-6">
              {detections.map((detection, index) => {
                const recommendation = recommendations[detection.class];
                const isEditing = editingPortion[detection.class];
                
                return (
                  <div key={index} className="border border-gray-200 rounded-lg p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-gray-800 capitalize">
                          {detection.class}
                        </h3>
                        <p className="text-sm text-gray-500">
                          Confidence: {(detection.conf * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-sm text-gray-400">
                        Bounding Box: [{detection.x1}, {detection.y1}, {detection.x2}, {detection.y2}]
                      </div>
                    </div>

                    {recommendation && recommendation.success && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                        <div className="flex items-start gap-3">
                          <Brain className="w-5 h-5 text-green-600 mt-0.5" />
                          <div className="flex-1">
                            <h4 className="font-medium text-green-800 mb-2">ChatGPT Recommendation:</h4>
                            <p className="text-green-700 mb-3">{recommendation.recommendation.explanation}</p>
                            <div className="flex items-center gap-4 mb-3">
                              <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full font-medium">
                                {recommendation.recommendation.recommended_amount} {recommendation.recommendation.unit}
                              </span>
                              <span className="text-sm text-green-600">
                                ({recommendation.recommendation.measurement_type} measurement)
                              </span>
                            </div>
                            
                            {!isEditing && (
                              <div className="flex gap-3">
                                <button
                                  onClick={() => acceptRecommendation(detection.class)}
                                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
                                >
                                  <CheckCircle className="w-4 h-4" />
                                  Use This Recommendation
                                </button>
                                <button
                                  onClick={() => setEditingPortion(prev => ({ ...prev, [detection.class]: true }))}
                                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                                >
                                  <Edit3 className="w-4 h-4" />
                                  Custom Portion
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    {isEditing && (
                      <CustomPortionInput
                        foodName={detection.class}
                        recommendation={recommendation?.recommendation}
                        onSubmit={getFinalNutrition}
                        onCancel={() => setEditingPortion(prev => ({ ...prev, [detection.class]: false }))}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Step 3: Final Results */}
        {currentStep === 4 && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center gap-2 mb-6">
              <Target className="w-6 h-6 text-purple-600" />
              <h2 className="text-2xl font-semibold text-gray-800">Final Nutrition Results</h2>
            </div>

            <div className="space-y-6">
              {finalResults.map((result, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-800 capitalize">
                        {result.food_name}
                      </h3>
                      <p className="text-sm text-gray-500">
                        {result.portion.quantity} {result.portion.unit} • 
                        {result.recommendation_used ? ' ChatGPT Recommendation' : ' Custom Portion'} • 
                        Data from {result.source}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div className="text-center p-4 bg-red-50 rounded-lg">
                      <div className="text-2xl font-bold text-red-600">
                        {result.nutrition.calories}
                      </div>
                      <div className="text-sm text-gray-600">Calories</div>
                    </div>
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {result.nutrition.protein}g
                      </div>
                      <div className="text-sm text-gray-600">Protein</div>
                    </div>
                    <div className="text-center p-4 bg-yellow-50 rounded-lg">
                      <div className="text-2xl font-bold text-yellow-600">
                        {result.nutrition.carbohydrates}g
                      </div>
                      <div className="text-sm text-gray-600">Carbs</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">
                        {result.nutrition.total_fat}g
                      </div>
                      <div className="text-sm text-gray-600">Fat</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="text-center">
                      <span className="font-medium text-gray-700">Fiber:</span>
                      <span className="ml-1 text-gray-600">{result.nutrition.dietary_fiber}g</span>
                    </div>
                    <div className="text-center">
                      <span className="font-medium text-gray-700">Sugar:</span>
                      <span className="ml-1 text-gray-600">{result.nutrition.sugars}g</span>
                    </div>
                    <div className="text-center">
                      <span className="font-medium text-gray-700">Sodium:</span>
                      <span className="ml-1 text-gray-600">{result.nutrition.sodium}mg</span>
                    </div>
                  </div>
                </div>
              ))}

              {/* Total Summary */}
              {finalResults.length > 1 && (
                <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200">
                  <h3 className="font-bold text-green-800 mb-4 text-lg">Total Nutritional Summary</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-green-600">
                        {finalResults.reduce((sum, item) => sum + item.nutrition.calories, 0)}
                      </div>
                      <div className="text-sm text-gray-600">Total Calories</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {finalResults.reduce((sum, item) => sum + item.nutrition.protein, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Total Protein</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-yellow-600">
                        {finalResults.reduce((sum, item) => sum + item.nutrition.carbohydrates, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Total Carbs</div>
                    </div>
                    <div className="text-center">
                      <div className="text-3xl font-bold text-purple-600">
                        {finalResults.reduce((sum, item) => sum + item.nutrition.total_fat, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Total Fat</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-8 text-center">
              <button
                onClick={() => {
                  setCurrentStep(1);
                  setImage(null);
                  setImagePreview(null);
                  setDetections([]);
                  setRecommendations({});
                  setFinalResults([]);
                  setError(null);
                }}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 mx-auto"
              >
                <Upload className="w-5 h-5" />
                Analyze Another Image
              </button>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-red-800">Error</h4>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkflowFoodDetectionApp;
