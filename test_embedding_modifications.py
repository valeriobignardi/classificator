#!/usr/bin/env python3
"""
Test script per verificare le modifiche per il salvataggio embedding in MongoDB
"""
import numpy as np
import sys
import os

# Add path for imports
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Pipeline')

def test_mongo_classification_reader():
    """Test per verificare che la modifica a mongo_classification_reader funzioni"""
    try:
        from mongo_classification_reader import MongoClassificationReader
        print("✅ Import MongoClassificationReader successful")
        
        # Test signature del metodo save_classification_result
        import inspect
        signature = inspect.signature(MongoClassificationReader.save_classification_result)
        print(f"📋 Signature save_classification_result:")
        print(f"   Parameters: {list(signature.parameters.keys())}")
        
        # Verifica che i nuovi parametri embedding e embedding_model esistano
        if 'embedding' in signature.parameters and 'embedding_model' in signature.parameters:
            print("✅ I parametri embedding e embedding_model sono presenti nella signature")
        else:
            print("❌ Parametri embedding/embedding_model MANCANTI dalla signature")
            
    except Exception as e:
        print(f"❌ Errore test mongo_classification_reader: {e}")

def test_pipeline_modifications():
    """Test per verificare che le modifiche alla pipeline siano syntatticamente corrette"""
    try:
        # Verifica che il file pipeline possa essere importato senza errori di sintassi
        import ast
        with open('/home/ubuntu/classificatore/Pipeline/end_to_end_pipeline.py', 'r') as f:
            content = f.read()
        
        # Parse del codice per verificare sintassi
        ast.parse(content)
        print("✅ Pipeline syntax check passed")
        
        # Verifica che le modifiche siano presenti
        if 'session_embedding =' in content and '_get_embedder_name()' in content:
            print("✅ Le modifiche per embedding sono presenti nel file")
        else:
            print("❌ Modifiche embedding NON TROVATE nel file")
            
        if 'embedding=session_embedding' in content and 'embedding_model=embedding_model' in content:
            print("✅ I parametri embedding sono passati alla funzione save_classification_result")
        else:
            print("❌ I parametri embedding NON SONO passati alla funzione save_classification_result")
            
    except Exception as e:
        print(f"❌ Errore test pipeline modifications: {e}")

def test_embedding_array_conversion():
    """Test per verificare la conversione numpy array in MongoDB"""
    try:
        # Simula un embedding numpy array
        test_embedding = np.random.rand(768).astype(np.float32)
        print(f"📊 Test embedding shape: {test_embedding.shape}")
        print(f"📊 Test embedding dtype: {test_embedding.dtype}")
        
        # Test conversione a list
        embedding_list = test_embedding.tolist()
        print(f"✅ Conversion to list successful, length: {len(embedding_list)}")
        
        # Test che sia JSON serializable
        import json
        json.dumps(embedding_list)
        print("✅ Embedding list is JSON serializable")
        
    except Exception as e:
        print(f"❌ Errore test embedding conversion: {e}")

if __name__ == "__main__":
    print("🧪 TEST EMBEDDING MODIFICATIONS")
    print("=" * 50)
    
    print("\n1. Testing MongoClassificationReader modifications:")
    test_mongo_classification_reader()
    
    print("\n2. Testing Pipeline modifications:")
    test_pipeline_modifications()
    
    print("\n3. Testing embedding array conversion:")
    test_embedding_array_conversion()
    
    print("\n🏁 TEST COMPLETE")
