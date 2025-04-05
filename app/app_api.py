# app/predict_api.py

# This script sets up a FastAPI app that serves predictions using a pre-trained Keras model.
# FastAPI’s built-in Swagger UI allows testing of the API endpoints.

from fastapi import FastAPI
from pydantic import BaseModel, create_model
import numpy as np
import pandas as pd
from tensorflow import keras
import uvicorn
from datetime import datetime
import joblib

model = keras.models.load_model("models/current_model.keras")
transform_pipeline = joblib.load("models/transform_pipeline.pkl")

feature_types = {
    "channel_id": int
}


numeric_features = [
    "ch9__f_1", "ch9__f_2", "ch9__f_3", "ch9__f_4", "ch9__f_5", "ch9__f_6", "ch9__f_12",
    "ch3__f_1", "ch3__f_2", "ch3__f_3", "ch3__f_4", "ch3__f_5", "ch3__f_6", "ch3__f_7", "ch3__f_8", "ch3__f_9", "ch3__f_12",
    "ch54__f_1", "ch54__f_2", "ch54__f_3", "ch54__f_4", "ch54__f_5", "ch54__f_6", "ch54__f_7", "ch54__f_8", "ch54__f_9", "ch54__f_12",
    "ch4__f_1", "ch4__f_2", "ch4__f_3", "ch4__f_4", "ch4__f_5", "ch4__f_6", "ch4__f_7", "ch4__f_8", "ch4__f_9", "ch4__f_12"
]
feature_types.update({col: int for col in numeric_features})

categorical_features = [
    "ch3__f_10", "ch3__f_11",
    "ch4__f_10", "ch4__f_11",
    "ch9__f_10", "ch9__f_11",
    "ch54__f_10", "ch54__f_11"
]
feature_types.update({col: str for col in categorical_features})

Features = create_model("Features", **{k: (v, ...) for k, v in feature_types.items()})


class PredictionInput(BaseModel):
    timeslot_datetime_from: str
    features: Features


app = FastAPI(
    title="TV Share Predictor",
    description="Predicts share_15_54 based on channel features",
    version="1.0.0",)

@app.post("/predict")
def predict(data: list[PredictionInput]):
    records = []
    for d in data:
        row = d.features.dict()
        dt = datetime.fromisoformat(d.timeslot_datetime_from)
        row["hour"] = dt.hour
        row["day"] = dt.weekday()  # 0 = Monday

        records.append(row)

    df = pd.DataFrame(records)

    try:

        df_transformed = transform_pipeline.transform(df)
        preds = model.predict(df_transformed).flatten().tolist()

        return {"predictions": preds}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)