# python_symptom_service/bilstm_predictor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from pydantic import BaseModel
from typing import List, Dict, Any

class BiLSTMInputData(BaseModel):
    id_hospital: int
    id_medicine: int
    input_year: int
    input_month: int
    usage_qty: int

class BiLSTMPredictionRequest(BaseModel):
    id_medicine: int
    historical_data: List[BiLSTMInputData]

class BiLSTMPredictionResponse(BaseModel):
    id_medicine: int
    predicted_next_month_usage: float
    last_actual_month: str
    prediction_for_month: str

WINDOW_SIZE = 12

class BiLSTMMedicinePredictor:
    def __init__(self, model_path: str):
        print(f"DEBUG: BiLSTMMedicinePredictor init: model_path='{model_path}'")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BiLSTM model not found at: {model_path}")
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        print("DEBUG: BiLSTM model loaded.")

    def _prepare_data_for_medicine(self, med_id: int, historical_data_input: List[BiLSTMInputData]) -> pd.DataFrame | None:
        if not historical_data_input:
            print(f"ERROR: No historical_data_input provided for medicine {med_id}")
            return None

        historical_data_dicts = [item.dict() for item in historical_data_input]
        df = pd.DataFrame(historical_data_dicts)

        df_med = df[df['id_medicine'] == med_id].copy()

        if df_med.empty:
            print(f"WARNING: No data found for medicine_id {med_id} in the provided historical_data.")
            return None

        try:
            df_med['date'] = pd.to_datetime(df_med['input_year'].astype(str) + '-' +
                                            df_med['input_month'].astype(str).str.zfill(2) + '-01')
        except KeyError as e:
            print(f"ERROR: Missing 'input_year' or 'input_month' in historical data for medicine {med_id}: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Could not create 'date' column for medicine {med_id}: {e}")
            return None

        df_med.sort_values('date', inplace=True)
        df_med.set_index('date', inplace=True)
        
        if 'usage_qty' not in df_med.columns:
            print(f"ERROR: 'usage_qty' column missing in historical data for medicine {med_id}")
            return None
            
        series_usage = df_med[['usage_qty']].copy()
        return series_usage

    def _create_sequences_for_prediction(self, data_array: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray | None:
        if len(data_array) < window_size:
            print(f"DEBUG: Not enough data to create sequence. Have {len(data_array)}, need {window_size}")
            return None
        
        sequence = data_array[-window_size:]
        return sequence.reshape((1, window_size, 1))

    def predict_next_month(self, med_id: int, historical_data_input: List[BiLSTMInputData]) -> Dict[str, Any]:
        series_df = self._prepare_data_for_medicine(med_id, historical_data_input)

        if series_df is None or series_df.empty:
            return {"error": f"Could not prepare data for medicine_id {med_id} from provided input."}

        if len(series_df) < WINDOW_SIZE:
            return {"error": f"Not enough historical data for medicine_id {med_id}. Need at least {WINDOW_SIZE} months, got {len(series_df)}."}

        try:
            scaled_values = self.scaler.fit_transform(series_df[['usage_qty']])
        except Exception as e:
            print(f"ERROR: Failed to scale usage_qty for medicine {med_id}: {e}")
            return {"error": f"Failed to process usage data for medicine_id {med_id}."}
            
        last_sequence_scaled = self._create_sequences_for_prediction(scaled_values)

        if last_sequence_scaled is None:
            return {"error": f"Could not create prediction sequence for medicine_id {med_id}. Need at least {WINDOW_SIZE} months of data."}

        try:
            predicted_scaled_value = self.model.predict(last_sequence_scaled)
            predicted_usage_qty = self.scaler.inverse_transform(predicted_scaled_value)[0][0]
        except Exception as e:
            print(f"ERROR: Model prediction or inverse transform failed for medicine {med_id}: {e}")
            return {"error": f"Model prediction failed for medicine_id {med_id}."}

        last_actual_date = series_df.index[-1]
        prediction_for_date = last_actual_date + pd.DateOffset(months=1)

        return {
            "id_medicine": med_id,
            "predicted_next_month_usage": float(predicted_usage_qty),
            "last_actual_month": last_actual_date.strftime('%Y-%m'),
            "prediction_for_month": prediction_for_date.strftime('%Y-%m')
        }