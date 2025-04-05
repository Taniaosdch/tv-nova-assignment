import requests
import json

# Load example input
with open("example_requests/test_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Call the local API
response = requests.post("http://localhost:8000/predict", json=data)

# Check status and output
print("Status:", response.status_code)
print("Response:", response.json())
