#!/usr/bin/env python3
"""
Test script per verificare i campi nel database MongoDB
Author: Valerio Bignardi
Date: 2025-01-27
"""

import sys
import os
import yaml
sys.path.append('.')
from mongo_classification_reader import MongoClassificationReader
from Utils.tenant import Tenant

def test_mongodb_fields():
    """
    Scopo: Testa i campi di classificazione nel database MongoDB
    """
    
    # Carica configurazione
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Crea oggetto tenant per Wopta
    tenant = Tenant(
        tenant_id='16c222a9-f293-11ef-9315-96000228e7fe',
        tenant_name='Wopta',
        tenant_slug='wopta',
        tenant_database='wopta',
        tenant_status=1
    )

    # Inizializza reader MongoDB
    mongo_reader = MongoClassificationReader(
        tenant=tenant,
        mongodb_url=config['mongodb']['url'],
        database_name=config['mongodb']['database']
    )

    # Test query per verificare i campi classificazione
    mongo_reader.connect()
    collection = mongo_reader.db[mongo_reader.collection_name]
    test_query = {'classification_method': {'$exists': True}}
    sessions = list(collection.find(test_query).limit(3))

    print(f'=== TEST CAMPI CLASSIFICAZIONE ===')
    print(f'Documenti con classification_method: {len(sessions)}')

    if sessions:
        for i, session in enumerate(sessions):
            print(f'\n--- Documento {i+1} ---')
            print(f'ID: {session.get("_id", "N/A")}')
            print(f'Classification: {session.get("classification", "N/A")}')
            print(f'Classification Method: {session.get("classification_method", "N/A")}')
            print(f'ML Prediction: {session.get("ml_prediction", "N/A")}')
            print(f'LLM Prediction: {session.get("llm_prediction", "N/A")}')
            print(f'ML Confidence: {session.get("ml_confidence", "N/A")}')
            print(f'LLM Confidence: {session.get("llm_confidence", "N/A")}')
    else:
        print('🔍 Nessun documento con classification_method. Verifico documenti recenti...')
        recent_docs = list(collection.find({}).sort('timestamp', -1).limit(3))
        for i, doc in enumerate(recent_docs):
            print(f'\n--- Documento recente {i+1} ---')
            print(f'ID: {doc.get("_id", "N/A")}')
            print(f'Classification: {doc.get("classification", "N/A")}')
            print(f'Has classification_method: {"classification_method" in doc}')
            print(f'Keys disponibili: {sorted(list(doc.keys()))}')
            
    # Test delle statistiche sui campi
    print(f'\n=== STATISTICHE CAMPI ===')
    total_docs = collection.count_documents({})
    classification_docs = collection.count_documents({'classification': {'$exists': True}})
    ml_prediction_docs = collection.count_documents({'ml_prediction': {'$exists': True}})
    llm_prediction_docs = collection.count_documents({'llm_prediction': {'$exists': True}})
    classification_method_docs = collection.count_documents({'classification_method': {'$exists': True}})
    
    print(f'Totale documenti: {total_docs}')
    print(f'Con campo classification: {classification_docs}')
    print(f'Con campo ml_prediction: {ml_prediction_docs}')
    print(f'Con campo llm_prediction: {llm_prediction_docs}')
    print(f'Con campo classification_method: {classification_method_docs}')

if __name__ == "__main__":
    test_mongodb_fields()
