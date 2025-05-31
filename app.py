# python_symptom_service/app.py
from fastapi import FastAPI, HTTPException, Depends
import os
import uvicorn
from typing import List

from symptom_mapper import SymptomMapper
from symptom_mapper import SymptomRequest as EmbeddingSymptomRequest

from ensemble_classifier import SymptomClassifier
from ensemble_classifier import ClassifierSymptomRequest, ClassifierResponse

from bilstm_predictor import BiLSTMMedicinePredictor
from bilstm_predictor import BiLSTMPredictionRequest, BiLSTMPredictionResponse

app = FastAPI(
    title="AI Disease Prediction Service",
    description="Provides APIs for symptom mapping, classification, and BiLSTM prediction.",
    version="1.2.1"
)

symptom_mapper_instance: SymptomMapper | None = None
classifier_instance: SymptomClassifier | None = None
bilstm_predictor_instance: BiLSTMMedicinePredictor | None = None

async def get_symptom_mapper():
    if symptom_mapper_instance is None:
        raise HTTPException(status_code=503, detail="SymptomMapper is not available or failed to load.")
    return symptom_mapper_instance

async def get_classifier():
    if classifier_instance is None:
        raise HTTPException(status_code=503, detail="Symptom Classifier is not available or failed to load.")
    return classifier_instance

async def get_bilstm_predictor():
    if bilstm_predictor_instance is None:
        raise HTTPException(status_code=503, detail="BiLSTM Medicine Predictor is not available or failed to load.")
    return bilstm_predictor_instance

@app.on_event("startup")
async def load_models_on_startup():
    global symptom_mapper_instance, classifier_instance, bilstm_predictor_instance
    
    print("INFO: Starting model loading process...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    try:
        print("INFO: Initializing SymptomMapper...")
        st_model_name = os.environ.get('ST_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embeddings_file = os.path.join(MODELS_DIR, os.environ.get('EMBEDDINGS_FILE_NAME', 'symptom_embeddings.npy'))
        labels_file = os.path.join(MODELS_DIR, os.environ.get('LABELS_FILE_NAME', 'symptom_labels.txt'))
        if not os.path.exists(embeddings_file) or not os.path.exists(labels_file):
             raise FileNotFoundError(f"SymptomMapper data files not found ({embeddings_file}, {labels_file}).")
        symptom_mapper_instance = SymptomMapper(st_model_name, embeddings_file, labels_file)
        print(f"SUCCESS: SymptomMapper initialized.")
    except Exception as e:
        symptom_mapper_instance = None
        print(f"ERROR: Failed to load SymptomMapper: {e}")

    try:
        print("INFO: Initializing Symptom Classifier...")
        classifier_model_file = os.path.join(MODELS_DIR, os.environ.get('CLASSIFIER_MODEL_FILE', 'ensemble_symptom_classifier.joblib'))
        label_encoder_file = os.path.join(MODELS_DIR, os.environ.get('LABEL_ENCODER_FILE', 'label_encoder.joblib'))
        if not os.path.exists(classifier_model_file) or not os.path.exists(label_encoder_file):
            raise FileNotFoundError(f"Classifier model files not found ({classifier_model_file}, {label_encoder_file}).")
        classifier_instance = SymptomClassifier(classifier_model_file, label_encoder_file)
        print("SUCCESS: Symptom Classifier initialized.")
    except Exception as e:
        classifier_instance = None
        print(f"ERROR: Failed to load Symptom Classifier: {e}")

    try:
        print("INFO: Initializing BiLSTM Medicine Predictor...")
        bilstm_model_file = os.path.join(MODELS_DIR, os.environ.get('BILSTM_MODEL_FILE', 'bilstm_medicine_model.keras'))
        
        if not os.path.exists(bilstm_model_file):
            raise FileNotFoundError(f"BiLSTM Keras model file not found: {bilstm_model_file}")
        
        bilstm_predictor_instance = BiLSTMMedicinePredictor(model_path=bilstm_model_file)
        print("SUCCESS: BiLSTM Medicine Predictor initialized.")
    except Exception as e:
        bilstm_predictor_instance = None
        print(f"ERROR: Failed to load BiLSTM Medicine Predictor: {e}")

    print("INFO: Model loading process finished.")

@app.post("/map_symptoms", summary="Map user symptoms via text embedding", response_model=List[dict])
async def map_symptoms_route(request_data: EmbeddingSymptomRequest, mapper: SymptomMapper = Depends(get_symptom_mapper)):
    try:
        matches = mapper.map_symptoms(request_data.symptoms_text, request_data.top_n, request_data.threshold)
        return matches
    except Exception as e:
        print(f"ERROR: Exception in /map_symptoms: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error: {str(e)}")

@app.post("/classify_symptoms", summary="Classify disease based on symptoms", response_model=ClassifierResponse)
async def classify_symptoms_route(request_data: ClassifierSymptomRequest, classifier: SymptomClassifier = Depends(get_classifier)):
    try:
        if not request_data.symptoms:
            raise HTTPException(status_code=400, detail="Symptoms list cannot be empty.")
        predicted_disease_name = classifier.predict_disease(request_data.symptoms)
        return ClassifierResponse(predicted_disease=predicted_disease_name)
    except ValueError as ve:
        print(f"VALUE_ERROR: in /classify_symptoms: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"ERROR: Exception in /classify_symptoms: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error: {str(e)}")

@app.post("/predict_medicine_usage", summary="Predict next month's medicine usage using BiLSTM", response_model=BiLSTMPredictionResponse)
async def predict_medicine_usage_route(
    request_data: BiLSTMPredictionRequest,
    predictor: BiLSTMMedicinePredictor = Depends(get_bilstm_predictor)
):
    try:
        if not request_data.historical_data:
            raise HTTPException(status_code=400, detail="Historical data is required for BiLSTM prediction.")

        result = predictor.predict_next_month(
            med_id=request_data.id_medicine,
            historical_data_input=request_data.historical_data
        )

        if "error" in result:
            status_code = 400
            if "Not enough historical data" in result["error"] or "Could not create prediction sequence" in result["error"]:
                status_code = 422 
            raise HTTPException(status_code=status_code, detail=result["error"])
            
        return BiLSTMPredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR: Exception in /predict_medicine_usage: {e}")
        # print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An internal error occurred during BiLSTM prediction: {str(e)}")

@app.get("/", summary="Service Information and Health Check")
async def read_root():
    model_status = {
        "symptom_mapper": "Loaded" if symptom_mapper_instance else "Failed/Unavailable",
        "symptom_classifier": "Loaded" if classifier_instance else "Failed/Unavailable",
        "bilstm_medicine_predictor": "Loaded" if bilstm_predictor_instance else "Failed/Unavailable",
    }
    overall_status = "healthy"
    if any(status == "Failed/Unavailable" for status in model_status.values()):
        overall_status = "degraded"
        
    return {
        "service_name": app.title,
        "version": app.version,
        "status": overall_status,
        "models_status": model_status
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)