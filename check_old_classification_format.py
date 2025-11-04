#!/usr/bin/env python3
"""
Verifica formato vecchie classificazioni
Autore: Valerio Bignardi
Data: 2025-11-04
"""

import os
from pymongo import MongoClient
import json

# Connessione MongoDB
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['classificazioni']
collection = db['humanitas_015007d9-d413-11ef-86a5-96000228e7fe']

# Caso specifico
session_id = "00e77bad-c42b-4690-a581-bf81b63d971f"
case = collection.find_one({"session_id": session_id})

print(f"🔍 ANALISI COMPLETA DOCUMENTO MongoDB")
print(f"=" * 80)

if case:
    # Rimuovi _id per JSON serialization
    case_copy = dict(case)
    case_copy['_id'] = str(case_copy['_id'])
    
    print(json.dumps(case_copy, indent=2, default=str))
    
    print(f"\n" + "=" * 80)
    print(f"📊 CHIAVI PRESENTI NEL DOCUMENTO:")
    print(f"=" * 80)
    for key in sorted(case.keys()):
        value = case[key]
        if isinstance(value, str) and len(value) > 100:
            value_display = value[:100] + "..."
        elif isinstance(value, dict):
            value_display = f"<dict con {len(value)} chiavi>"
        elif isinstance(value, list):
            value_display = f"<list con {len(value)} elementi>"
        else:
            value_display = value
        print(f"   {key}: {value_display}")

else:
    print("❌ Caso non trovato")

client.close()
