FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
WORKDIR /app

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
