import json

# lower values for features give lower share
with open("example_requests/test_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)
response = requests.post("http://localhost:8000/predict", json=data)


print("Status:", response.status_code)
print("Response:", response.json())
