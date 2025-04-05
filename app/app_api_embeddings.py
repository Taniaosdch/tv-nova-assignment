from fastapi import FastAPI
from pydantic import BaseModel, create_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import uvicorn


########################################
# 1. Load pipeline + model
########################################

model = load_model("models/current_model.keras")
transform_pipeline = joblib.load("models/transform_pipeline.pkl")

########################################
# 2. Define columns + Pydantic schema
########################################

# The columns your pipeline transforms
categorical_low = ["channel_id", "day"]
numeric_cols = [
    "ch9__f_1", "ch9__f_2", "ch9__f_3", "ch9__f_4", "ch9__f_5", "ch9__f_6", "ch9__f_12",
    "ch3__f_1", "ch3__f_2", "ch3__f_3", "ch3__f_4", "ch3__f_5", "ch3__f_6", "ch3__f_7", "ch3__f_8", "ch3__f_9", "ch3__f_12",
    "ch54__f_1", "ch54__f_2", "ch54__f_3", "ch54__f_4", "ch54__f_5", "ch54__f_6", "ch54__f_7", "ch54__f_8", "ch54__f_9", "ch54__f_12",
    "ch4__f_1", "ch4__f_2", "ch4__f_3", "ch4__f_4", "ch4__f_5", "ch4__f_6", "ch4__f_7", "ch4__f_8", "ch4__f_9", "ch4__f_12"
]

# The columns that go to embedding
categorical_high = [
    "ch3__f_10", "ch4__f_10", "ch54__f_10",
    "ch3__f_11", "ch4__f_11", "ch54__f_11"
]


feature_types = {col: float for col in numeric_cols}
feature_types.update({col: str for col in categorical_high})
feature_types.update({col: int for col in categorical_low})

Features = create_model(
    "Features",  
    **{k: (v, ...) for k, v in feature_types.items()}
)

class PredictionInput(BaseModel):
    timeslot_datetime_from: str
    features: Features



app = FastAPI(
    title="TV Share Predictor",
    version="2.0.0",
    description="""
    Predict share_15_54 using a hybrid embeding model.
    We load transform_pipeline.pkl for numeric + low-cat, 
    and pass high-cat columns as strings for embedding.
    """,
)



def prepare_input(df: pd.DataFrame) -> dict:
    """
    1) Use transform_pipeline to transform numeric + low-cat
    2) Send high-cat as tf.string
    """

    X_transformed = transform_pipeline.transform(df) 
    inputs = {
        "numerical_features": tf.convert_to_tensor(X_transformed, dtype=tf.float32)
    }

    for col in categorical_high:
        inputs[col] = tf.convert_to_tensor(
            df[col].astype(str).values.reshape(-1, 1),
            dtype=tf.string
        )

    return inputs


@app.post("/predict")
def predict(data: list[PredictionInput]):
    """
    timeslot_datetime_from -> we can compute 'hour' or 'day' if needed 
    or let the user provide 'day' in features.
    """

    # Convert JSON payload to DataFrame
    records = []
    for item in data:
        row = item.features.dict()
        dt = datetime.fromisoformat(item.timeslot_datetime_from)
        row["hour"] = dt.hour
        row["day"] = dt.strftime("%A")

        records.append(row)

    df = pd.DataFrame(records)

    inputs = prepare_input(df)

    preds = model.predict(inputs).flatten().tolist()
    return {"predictions": preds}


if __name__ == "__main__":
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
