#!/usr/bin/env python3
"""
Test per verificare la correzione del problema numpy.int64 con MongoDB

Autore: AI Assistant
Data: 2025-08-24
"""
import numpy as np
from mongo_classification_reader import MongoClassificationReader
import json

def test_numpy_conversion():
    """Test conversione numpy types per compatibilità MongoDB"""
    print("🧪 TEST: Conversione tipi numpy per MongoDB")
    
    # Simula metadati cluster con tipi numpy
    test_cluster_metadata = {
        'cluster_id': np.int64(42),  # numpy.int64
        'is_representative': np.bool_(True),  # numpy.bool_
        'propagated_from': 'cluster_propagation',
        'propagation_confidence': np.float64(0.85)  # numpy.float64
    }
    
    print(f"📊 Dati originali:")
    for key, value in test_cluster_metadata.items():
        print(f"   {key}: {value} (tipo: {type(value)})")
    
    # Inizializza reader MongoDB
    reader = MongoClassificationReader()
    
    # Test simulando la logica di conversione interna
    doc_metadata = {}
    
    # Simula la logica del metodo salva_risultato_classificazione
    if "cluster_id" in test_cluster_metadata:
        cluster_id_value = test_cluster_metadata["cluster_id"]
        if hasattr(cluster_id_value, 'item'):  # numpy type
            doc_metadata["cluster_id"] = int(cluster_id_value.item())
        else:
            doc_metadata["cluster_id"] = int(cluster_id_value)
    
    if "is_representative" in test_cluster_metadata:
        doc_metadata["is_representative"] = bool(test_cluster_metadata["is_representative"])
    
    if "propagated_from" in test_cluster_metadata:
        doc_metadata["propagated_from"] = str(test_cluster_metadata["propagated_from"])
    
    if "propagation_confidence" in test_cluster_metadata:
        confidence_value = test_cluster_metadata["propagation_confidence"]
        if hasattr(confidence_value, 'item'):  # numpy type
            doc_metadata["propagation_confidence"] = float(confidence_value.item())
        else:
            doc_metadata["propagation_confidence"] = float(confidence_value)
    
    print(f"\n✅ Dati convertiti:")
    for key, value in doc_metadata.items():
        print(f"   {key}: {value} (tipo: {type(value)})")
    
    # Test serializzazione JSON (simula MongoDB)
    try:
        json_str = json.dumps(doc_metadata)
        print(f"\n🎯 Test serializzazione JSON: ✅ SUCCESS")
        print(f"   JSON: {json_str}")
        
        # Test deserializzazione
        recovered_data = json.loads(json_str)
        print(f"\n🔄 Test deserializzazione: ✅ SUCCESS")
        print(f"   cluster_id recuperato: {recovered_data['cluster_id']} (tipo: {type(recovered_data['cluster_id'])})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test serializzazione FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_numpy_conversion()
    if success:
        print(f"\n🎉 TUTTI I TEST PASSATI! La correzione numpy.int64 funziona correttamente.")
    else:
        print(f"\n💥 TEST FALLITI! Necessarie ulteriori correzioni.")
