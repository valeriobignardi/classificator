#!/usr/bin/env python3
"""
Test performance delle query MongoDB per identificare la causa dei timeout
Autore: Valerio Bignardi  
Data: 2025-09-01
"""

import time
import sys
import os

# Aggiungi path del progetto
sys.path.append('/home/ubuntu/classificatore')

def test_mongodb_performance():
    """
    Testa le performance delle query MongoDB sui casi di revisione
    """
    print("🔍 TEST PERFORMANCE MONGODB - CASI DI REVISIONE")
    print("="*80)
    
    try:
        from Utils.tenant import Tenant
        
        # Connessione al tenant humanitas
        print("🔧 Inizializzazione tenant...")
        start_time = time.time()
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        print(f"✅ Tenant risolto in {time.time() - start_time:.3f}s")
        
        # Test connessione diretta MongoDB
        print("\n🔧 Test connessione MongoDB diretta...")
        start_time = time.time()
        
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['classificazioni']
        collection_name = f"humanitas_{tenant.tenant_id.replace('-', '_')}"
        collection = db[collection_name]
        
        print(f"✅ Connessione MongoDB in {time.time() - start_time:.3f}s")
        print(f"📊 Collection: {collection_name}")
        
        # Test 1: Count totale documenti
        print("\n📊 TEST 1: Count totale documenti")
        start_time = time.time()
        total_count = collection.count_documents({})
        elapsed = time.time() - start_time
        print(f"   📈 Documenti totali: {total_count} ({elapsed:.3f}s)")
        
        # Test 2: Count casi pending (come fa l'API)
        print("\n📊 TEST 2: Count casi pending")
        start_time = time.time()
        pending_count = collection.count_documents({'review_status': 'pending'})
        elapsed = time.time() - start_time
        print(f"   📈 Casi pending: {pending_count} ({elapsed:.3f}s)")
        
        # Test 3: Query complessa come l'API review/cases
        print("\n📊 TEST 3: Query complessa API review/cases")
        start_time = time.time()
        
        # Simula la query dell'API
        query = {'review_status': 'pending'}
        projection = {
            '_id': 1,
            'conversation_id': 1, 
            'classification_type': 1,
            'predicted_label': 1,
            'confidence': 1,
            'conversation_text': 1,
            'timestamp': 1,
            'cluster_id': 1
        }
        
        cases = list(collection.find(query, projection).limit(20).sort('timestamp', -1))
        elapsed = time.time() - start_time
        print(f"   📈 Query API simulata: {len(cases)} risultati ({elapsed:.3f}s)")
        
        # Test 4: Query con aggregation (più pesante)
        print("\n📊 TEST 4: Aggregation pesante")
        start_time = time.time()
        
        pipeline = [
            {'$match': {'review_status': 'pending'}},
            {'$group': {'_id': '$classification_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        
        agg_results = list(collection.aggregate(pipeline))
        elapsed = time.time() - start_time
        print(f"   📈 Aggregation: {len(agg_results)} risultati ({elapsed:.3f}s)")
        
        # Test 5: Index analysis
        print("\n📊 TEST 5: Analisi indici")
        indexes = list(collection.list_indexes())
        print(f"   📈 Indici presenti: {len(indexes)}")
        for idx in indexes:
            print(f"     - {idx.get('name', 'unknown')}: {idx.get('key', {})}")
        
        # Suggerimento indici
        print("\n💡 SUGGERIMENTI OTTIMIZZAZIONE:")
        if not any('review_status' in str(idx.get('key', {})) for idx in indexes):
            print("   ⚠️  Manca indice su 'review_status' - crea: db.collection.createIndex({'review_status': 1})")
        if not any('timestamp' in str(idx.get('key', {})) for idx in indexes):
            print("   ⚠️  Manca indice su 'timestamp' - crea: db.collection.createIndex({'timestamp': -1})")
            
        # Chiudi connessione
        client.close()
        print(f"\n✅ Test completato successfully!")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mongodb_performance()
