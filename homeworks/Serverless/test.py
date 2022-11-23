import requests

data = {
    "url": "https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg"
}

url = "http://localhost:8080/2015-03-31/functions/function/invocations"

results = requests.post(url, json=data).json()

print(results)