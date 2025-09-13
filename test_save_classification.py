#!/usr/bin/env python3
"""
Test script per verificare la funzione save_classification_result
Author: Valerio Bignardi
Date: 2025-01-27
"""

import sys
import os
import yaml
sys.path.append('.')
from mongo_classification_reader import MongoClassificationReader
from Utils.tenant import Tenant

def test_save_classification_result():
    """
    Scopo: Testa la funzione save_classification_result con diversi scenari
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

    print(f'=== TEST SAVE_CLASSIFICATION_RESULT ===')
    
    # Test scenario 1: ML + LLM result (ensemble)
    print(f'\\n--- Test 1: Ensemble (ML + LLM) ---')
    test_session_id_1 = "test_ensemble_1"
    ml_result_1 = {
        'predicted_label': 'RICHIESTA_INFORMAZIONI',
        'confidence': 0.85
    }
    llm_result_1 = {
        'predicted_label': 'RICHIESTA_INFORMAZIONI', 
        'confidence': 0.90
    }
    classification_1 = 'RICHIESTA_INFORMAZIONI'
    method_1 = 'ENSEMBLE'
    
    success_1 = mongo_reader.save_classification_result(
        session_id=test_session_id_1,
        client_name='wopta',
        ml_result=ml_result_1,
        llm_result=llm_result_1,
        final_decision={'classification': classification_1, 'method': method_1}
    )
    print(f'Ensemble save result: {success_1}')
    
    # Test scenario 2: Solo LLM result  
    print(f'\\n--- Test 2: Solo LLM ---')
    test_session_id_2 = "test_llm_only_2"
    ml_result_2 = {}  # Vuoto
    llm_result_2 = {
        'predicted_label': 'APERTURA_POLIZZA',
        'confidence': 0.75
    }
    classification_2 = 'APERTURA_POLIZZA'
    method_2 = 'LLM_ONLY'
    
    success_2 = mongo_reader.save_classification_result(
        session_id=test_session_id_2,
        client_name='wopta',
        ml_result=ml_result_2,
        llm_result=llm_result_2,
        final_decision={'classification': classification_2, 'method': method_2}
    )
    print(f'LLM-only save result: {success_2}')
    
    # Test scenario 3: ML result vuoto ma esistente
    print(f'\\n--- Test 3: ML result vuoto ---')
    test_session_id_3 = "test_empty_ml_3"
    ml_result_3 = {'confidence': 0.0}  # Senza predicted_label
    llm_result_3 = {
        'predicted_label': 'INFO_COPERTURA_MASSIMA',
        'confidence': 0.80
    }
    classification_3 = 'INFO_COPERTURA_MASSIMA'
    method_3 = 'HYBRID'
    
    success_3 = mongo_reader.save_classification_result(
        session_id=test_session_id_3,
        client_name='wopta',
        ml_result=ml_result_3,
        llm_result=llm_result_3,
        final_decision={'classification': classification_3, 'method': method_3}
    )
    print(f'Empty ML save result: {success_3}')
    
    # Verifica i risultati nel database
    print(f'\\n=== VERIFICA SALVATAGGIO ===')
    mongo_reader.connect()
    collection = mongo_reader.db[mongo_reader.collection_name]
    
    for test_id in [test_session_id_1, test_session_id_2, test_session_id_3]:
        doc = collection.find_one({'_id': test_id})
        if doc:
            print(f'\\n--- {test_id} ---')
            print(f'Classification: {doc.get("classification", "N/A")}')
            print(f'Classification Method: {doc.get("classification_method", "N/A")}')
            print(f'ML Prediction: {doc.get("ml_prediction", "N/A")}')
            print(f'LLM Prediction: {doc.get("llm_prediction", "N/A")}')
            print(f'ML Confidence: {doc.get("ml_confidence", "N/A")}')
            print(f'LLM Confidence: {doc.get("llm_confidence", "N/A")}')
        else:
            print(f'\\n❌ Documento {test_id} non trovato!')
    
    # Cleanup - rimuovi i documenti di test
    print(f'\\n=== CLEANUP ===')
    for test_id in [test_session_id_1, test_session_id_2, test_session_id_3]:
        result = collection.delete_one({'_id': test_id})
        print(f'Removed {test_id}: {result.deleted_count} doc')

if __name__ == "__main__":
    test_save_classification_result()
