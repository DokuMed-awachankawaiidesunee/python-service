from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import BaseModel
import os

class SymptomRequest(BaseModel):
    symptoms_text: str
    top_n: int = 10
    threshold: float = 0.4

class SymptomMapper:
    def __init__(self, model_name_or_path, embeddings_path, labels_path):
        print(f"DEBUG: SymptomMapper init: model_path='{model_name_or_path}', embeddings='{embeddings_path}', labels='{labels_path}'")
        if not os.path.exists(embeddings_path):
             raise FileNotFoundError(f"Embeddings file not found at SymptomMapper init: {embeddings_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at SymptomMapper init: {labels_path}")

        self.model = SentenceTransformer(model_name_or_path)
        self.predefined_symptom_embeddings = np.load(embeddings_path)
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.predefined_symptoms = [line.strip() for line in f.readlines()]
        print(f"DEBUG: SymptomMapper loaded {len(self.predefined_symptoms)} symptoms.")


    def map_symptoms(self, user_input_text: str, top_n: int = 10, threshold: float = 0.5):
        if not user_input_text.strip():
            return []
        
        user_embedding = self.model.encode([user_input_text])
        similarities = cosine_similarity(
            user_embedding,
            self.predefined_symptom_embeddings
        )[0]

        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                results.append((self.predefined_symptoms[i], float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        matched_symptoms = [{"symptom": symptom, "score": score} for symptom, score in results[:top_n]]
        return matched_symptoms