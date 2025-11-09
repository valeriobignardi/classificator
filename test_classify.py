#!/usr/bin/env python3
"""
Test classificazione con tenant Wopta (GPT-5)
"""
import requests
import json

url = "http://localhost:5000/classify/new/wopta"
headers = {"Content-Type": "application/json"}

payload = {
    "text": "Ciao, vorrei informazioni sulla mia polizza assicurativa"
}

print(f"📤 Richiesta a {url}")
print(f"📦 Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"📊 Status Code: {response.status_code}")
    print(f"📄 Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Errore: {e}")
