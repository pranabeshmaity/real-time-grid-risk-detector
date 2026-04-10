import requests
import json

data = {
    "voltages": [1.0] * 118,
    "frequencies": [60.0] * 118,
    "powers": [0.0] * 118
}

response = requests.post("http://localhost:8000/api/v1/predictions/single", json=data)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))
