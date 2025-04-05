
# 📺 TV Share Predictor

This project is a machine learning model designed to predict TV audience share for the 15–54 demographic. It uses a trained Keras neural network with embeddings for high-cardinality categorical variables. The project includes a FastAPI REST API for testing, and Docker support for deployment.

## Features
- Neural network model with Keras
- FastAPI REST API with `/predict` endpoint and Swagger UI
- Scripts for easy model retraining and versioning
- REST API testing utilities and example inputs
- Dockerized deployment

## Installation
Install dependencies using `pip install -r requirements.txt`.

- Test the FastAPI is for inference only. It loads the latest current_model.keras to generate predictions:
- Swagger UI at `http://localhost:8000/docs`
- Simply run in terminal ```uvicorn app.app_api_embeddings:app --reload```
- Example input JSON in `example_requests/test_input.json`


The ```retrain_and_version_embeddings.py``` script is used solely for training and versioning a new embedding-based model whenever updated data becomes available.

- Loads fresh data (e.g., data.csv or an API feed).
- Cleans and preprocesses the data (scaling numeric features, one-hot encoding low-cardinality columns).
- Creates embeddings for high-cardinality categorical columns.
- Trains the Keras model, logs the results, and evaluates on a test set.
- Saves both the updated model (.keras file) and a fitted transform_pipeline.pkl so inference stays consistent.
- Overwrites the current model (current_model.keras) and pipeline (transform_pipeline.pkl) if all goes well.

FastAPI service will automatically use the latest model on next startup.