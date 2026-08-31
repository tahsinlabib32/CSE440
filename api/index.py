from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import unicodedata
import re
import os

app = FastAPI()



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
NB_MODEL_PATH = os.path.join(BASE_DIR, "nbmodel.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    nb_model = joblib.load(NB_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    nb_model = None
    vectorizer = None

class PredictionRequest(BaseModel):
    case_title: str
    case_text: str
    explain: bool = False

class SingleModelResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    explanation: Optional[list] = None

class PredictionResponse(BaseModel):
    lr: SingleModelResponse
    nb: SingleModelResponse

def clean_legal_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.get("/api/health")
def health_check():
    if model is None or nb_model is None or vectorizer is None:
        return {"status": "error", "message": "Models failed to load."}
    return {"status": "ok", "message": "API is running and models are loaded."}

@app.get("/api/random")
def get_random_sample():
    import json, random
    sample_path = os.path.join(BASE_DIR, "random_samples.json")
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        return random.choice(samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not load random samples.")

@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or nb_model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Models are not fully loaded.")
    
    try:
        model_text = "TITLE: " + request.case_title + " TEXT: " + request.case_text
        cleaned_text = clean_legal_text(model_text)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Input text is empty after cleaning.")

        features = vectorizer.transform([cleaned_text])
        dense_features = features.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        present_features_mask = dense_features > 0
        
        import numpy as np
        
        def run_model(m, is_lr=False, is_nb=False):
            pred = m.predict(features)[0]
            conf = None
            if hasattr(m, "predict_proba"):
                conf = float(max(m.predict_proba(features)[0]))
                
            expl = None
            if request.explain:
                class_idx = list(m.classes_).index(pred)
                
                if is_lr and hasattr(m, "coef_"):
                    class_coef = m.coef_[class_idx]
                    contributions = dense_features * class_coef
                    top_indices = np.argsort(contributions)[-10:][::-1]
                    expl = [{"word": str(feature_names[i]), "score": float(contributions[i])} for i in top_indices if contributions[i] > 0]
                    
                elif is_nb and hasattr(m, "feature_log_prob_"):
                    class_log_prob = m.feature_log_prob_[class_idx]
                    masked_log_prob = np.where(present_features_mask, class_log_prob, -np.inf)
                    top_indices = np.argsort(masked_log_prob)[-10:][::-1]
                    expl = [{"word": str(feature_names[i]), "score": float(class_log_prob[i])} for i in top_indices if masked_log_prob[i] != -np.inf]
            
            return SingleModelResponse(prediction=pred, confidence=conf, explanation=expl)

        lr_res = run_model(model, is_lr=True)
        nb_res = run_model(nb_model, is_nb=True)
        
        return PredictionResponse(lr=lr_res, nb=nb_res)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
