#!/usr/bin/env python3
"""
Script per verificare casi con LLM Prediction = N/A
Autore: Valerio Bignardi
Data: 2025-11-04
"""

import os
import sys
from pymongo import MongoClient
from bson import ObjectId
import json

# Connessione MongoDB
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['classificazioni']

# Collection Humanitas
collection_name = 'humanitas_015007d9-d413-11ef-86a5-96000228e7fe'
collection = db[collection_name]

print(f"🔍 Analisi casi con LLM Prediction = N/A")
print(f"📊 Collection: {collection_name}\n")

# Caso specifico dallo screenshot
session_id = "00e77bad-c42b-4690-a581-bf81b63d971f"
print(f"=" * 80)
print(f"🎯 CASO SPECIFICO: {session_id}")
print(f"=" * 80)

case = collection.find_one({"session_id": session_id})
if case:
    print(f"\n📋 Dati MongoDB per session_id: {session_id}")
    print(f"   🆔 _id: {case.get('_id')}")
    print(f"   📝 Classification: {case.get('classification', 'N/A')}")
    print(f"   🎯 Confidence: {case.get('confidence', 'N/A')}")
    print(f"   🔧 Method: {case.get('classification_method', 'N/A')}")
    print(f"   📊 Review Status: {case.get('review_status', 'N/A')}")
    print(f"   📅 Created: {case.get('classified_at', case.get('timestamp', 'N/A'))}")
    
    print(f"\n🤖 LLM PREDICTION:")
    print(f"   Label: {case.get('llm_prediction', 'CAMPO NON PRESENTE')}")
    print(f"   Confidence: {case.get('llm_confidence', 'CAMPO NON PRESENTE')}")
    print(f"   Reasoning: {case.get('llm_reasoning', 'CAMPO NON PRESENTE')}")
    
    print(f"\n🧠 ML PREDICTION:")
    print(f"   Label: {case.get('ml_prediction', 'CAMPO NON PRESENTE')}")
    print(f"   Confidence: {case.get('ml_confidence', 'CAMPO NON PRESENTE')}")
    
    print(f"\n📦 METADATA:")
    metadata = case.get('metadata', {})
    print(f"   Representative: {metadata.get('representative', 'N/A')}")
    print(f"   Propagated: {metadata.get('propagated', 'N/A')}")
    print(f"   Outlier: {metadata.get('outlier', 'N/A')}")
    print(f"   Cluster ID: {metadata.get('cluster_id', 'N/A')}")
    print(f"   Method: {metadata.get('method', 'N/A')}")
    
    # Verifica se c'è ensemble_details
    ensemble_details = case.get('ensemble_details', {})
    if ensemble_details:
        print(f"\n🎯 ENSEMBLE DETAILS:")
        print(f"   Weights: {ensemble_details.get('weights_used', 'N/A')}")
        print(f"   LLM Available: {ensemble_details.get('llm_available', 'N/A')}")
        print(f"   ML Available: {ensemble_details.get('ml_available', 'N/A')}")
else:
    print(f"❌ Caso non trovato in MongoDB!")

print(f"\n" + "=" * 80)
print(f"📊 STATISTICA GENERALE: Casi con review_status='pending' e classification='altro'")
print(f"=" * 80)

# Query per tutti i casi ALTRO pending
altro_cases = list(collection.find({
    "review_status": "pending",
    "classification": "altro"
}).limit(10))

print(f"\n🔍 Trovati {len(altro_cases)} casi 'altro' in review (primi 10):\n")

for idx, case in enumerate(altro_cases, 1):
    session_id = case.get('session_id', 'N/A')
    llm_pred = case.get('llm_prediction', None)
    ml_pred = case.get('ml_prediction', None)
    method = case.get('classification_method', 'N/A')
    
    print(f"{idx}. Session: {session_id[:20]}...")
    print(f"   📝 LLM Prediction: {llm_pred if llm_pred else '❌ ASSENTE'}")
    print(f"   🧠 ML Prediction: {ml_pred if ml_pred else '❌ ASSENTE'}")
    print(f"   🔧 Method: {method}")
    print()

# Conta quanti casi ALTRO hanno llm_prediction vuoto/null
print(f"=" * 80)
print(f"📊 CONTEGGIO: Casi 'altro' senza LLM prediction")
print(f"=" * 80)

# Casi senza campo llm_prediction
count_no_llm_field = collection.count_documents({
    "review_status": "pending",
    "classification": "altro",
    "llm_prediction": {"$exists": False}
})

# Casi con llm_prediction = null o ""
count_llm_null = collection.count_documents({
    "review_status": "pending",
    "classification": "altro",
    "$or": [
        {"llm_prediction": None},
        {"llm_prediction": ""},
        {"llm_prediction": "N/A"}
    ]
})

# Totale casi altro pending
total_altro = collection.count_documents({
    "review_status": "pending",
    "classification": "altro"
})

print(f"\n📊 Casi 'altro' in pending: {total_altro}")
print(f"❌ Senza campo llm_prediction: {count_no_llm_field}")
print(f"❌ Con llm_prediction vuoto/null/N/A: {count_llm_null}")
print(f"✅ Con llm_prediction valido: {total_altro - count_no_llm_field - count_llm_null}")

client.close()
