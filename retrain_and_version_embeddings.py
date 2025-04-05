# This script simulates "daily retraining" of the model (`current_model.keras`)
# using data from a REST API (placeholder here).
#
# It saves versioned copies of the model and preprocessing pipeline
# and updates the current active model (current_model.keras) and transformer (transform_pipeline.pkl).

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import StringLookup, Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import shutil

# ======================================
# 1. Simulate or load daily data
# ======================================
# E.g. replace with API fetch if needed
# response = requests.get("https://name.com/api/data")
# url = response.json()["dataset_url"]
# df = pd.read_csv(url)

df = pd.read_csv("data.csv").drop_duplicates()
df = df.drop(["ch9__f_7", "ch9__f_8", "ch9__f_9", "ch9__f_10", "ch9__f_11"], axis=1, errors="ignore")

df["timeslot_datetime_from"] = pd.to_datetime(df["timeslot_datetime_from"])
df["hour"] = df["timeslot_datetime_from"].dt.hour
df["day"] = df["timeslot_datetime_from"].dt.day_name()


df_clean = df.dropna()

# =====================================
# 2. Train/test split
# =====================================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    df_clean.drop(columns=["share_15_54", "timeslot_datetime_from", "main_ident", "share_15_54_3mo_mean"], errors="ignore"),
    df_clean["share_15_54"],
    test_size=0.2,
    random_state=42
)
print("Train full shape:", X_train_full.shape, y_train_full.shape)
print("Test shape:", X_test.shape, y_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    random_state=42
)
print("X_train/X_valid:", X_train.shape, X_valid.shape)

# =====================================
# 3. Define columns for pipeline
# =====================================
categorical_high = [
    "ch3__f_10", "ch4__f_10", "ch54__f_10",
    "ch3__f_11", "ch4__f_11", "ch54__f_11"
]
categorical_low = ["channel_id", "day"]

# numeric columns automatically discovered
num_vars = X_train.select_dtypes(include=["int64", "float64"]).columns

# Build pipeline for numeric + low-cat
transform_pipeline = ColumnTransformer([
    ("scaler", StandardScaler(), num_vars),
    ("one_hot", OneHotEncoder(handle_unknown="ignore"), categorical_low),
])
transform_pipeline.fit(X_train)

# For debugging or reference
X_transformed = transform_pipeline.transform(X_train)
print("X_transformed shape:", X_transformed.shape)

# =====================================
# 4. Create embedding inputs
# =====================================
embedding_inputs = []
embedding_encoded = []

for col in categorical_high:
    vocab = df[col].dropna().unique().astype(str)
    lookup = StringLookup(vocabulary=vocab.tolist(), output_mode="int", oov_token="[UNK]")
    inp = Input(shape=(1,), dtype=tf.string, name=col)
    embedding_inputs.append(inp)

    idx = lookup(inp)
    emb = Embedding(input_dim=len(vocab) + 2, output_dim=4)(idx)
    emb_flat = Flatten()(emb)
    embedding_encoded.append(emb_flat)

# numeric + low-cat input
numerical_input = Input(shape=(X_transformed.shape[1],), name="numerical_features")

all_inputs = [numerical_input] + embedding_inputs
all_features = Concatenate()([numerical_input] + embedding_encoded)

hidden_1 = Dense(128, activation="relu")(all_features)
hidden_2 = Dense(64, activation="relu")(hidden_1)
concat = Concatenate()([hidden_2, all_features])
output = Dense(1)(concat)

model = Model(inputs=all_inputs, outputs=output)
model.compile(optimizer="adam", loss="mse")

# =====================================
# 5. Prepare input function
# =====================================
def prepare_input(X_df):
    # transform numeric + low-cat
    transformed_num = transform_pipeline.transform(X_df)
    input_dict = {
        "numerical_features": tf.convert_to_tensor(transformed_num, dtype=tf.float32),
    }
    for ch_col in categorical_high:
        input_dict[ch_col] = tf.convert_to_tensor(
            X_df[ch_col].astype(str).values.reshape(-1, 1),
            dtype=tf.string
        )
    return input_dict

train_inputs = prepare_input(X_train)
valid_inputs = prepare_input(X_valid)
test_inputs = prepare_input(X_test)

# =====================================
# 6. Train the model
# =====================================
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath="models/model_{epoch:02d}_{val_loss:.2f}.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

history = model.fit(
    train_inputs, y_train,
    validation_data=(valid_inputs, y_valid),
    epochs=30,
    batch_size=32,
    callbacks=[model_checkpoint, early_stop]
)

# =====================================
# 7. Evaluate
# IMPORTANT: pass multi-input dict, not X_test_transformed
# =====================================
eval_loss = model.evaluate(test_inputs, y_test)
print("Test Loss:", eval_loss)

# =====================================
# 8. Save model + pipeline
# =====================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("models", exist_ok=True)

model_path = f"models/model_{timestamp}.keras"
pipeline_path = f"models/pipeline_{timestamp}.pkl"

model.save(model_path)
joblib.dump(transform_pipeline, pipeline_path)

# update current_model.keras + transform_pipeline
shutil.copy(model_path, "models/current_model.keras")
shutil.copy(pipeline_path, "models/transform_pipeline.pkl")

# =====================================
# 9. Logging
# =====================================
os.makedirs("logs", exist_ok=True)

val_loss_min = float(min(history.history["val_loss"]))
metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_file": model_path,
    "val_loss": val_loss_min,
    "test_loss": float(eval_loss),
    "num_rows": len(df_clean)
}
with open("logs/training_log.jsonl", "a") as f:
    json.dump(metrics, f)
    f.write("\n")

print(f"✅ Model retrained. Best val_loss: {val_loss_min:.4f} | Test_loss: {eval_loss:.4f}")
