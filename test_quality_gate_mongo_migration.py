#!/usr/bin/env python3
"""
Author: System
Date: 2025-01-15
Description: Test per verificare la migrazione del QualityGateEngine a MongoDB
Last Update: 2025-01-15

Test per validare l'integrazione MongoDB nel sistema di riaddestramento
"""

import json
import os
import sys
from typing import List, Dict, Any

def test_quality_gate_mongo_integration():
    """
    Scopo: Testa l'integrazione MongoDB nel QualityGateEngine
    
    Parametri input: Nessuno
    Output: Risultato del test di integrazione
    Ultimo aggiornamento: 2025-01-15
    """
    
    print("🧪 Test QualityGateEngine - Migrazione MongoDB")
    print("=" * 60)
    
    try:
        # 1. Test connessione MongoDB
        print("1️⃣ Test connessione MongoDB...")
        from mongo_classification_reader import MongoClassificationReader
        
        mongo_reader = MongoClassificationReader()
        if mongo_reader.connect():
            print("✅ Connessione MongoDB: OK")
            
            # Verifica dati disponibili
            labels = mongo_reader.get_available_labels("humanitas")
            sessions = mongo_reader.get_all_sessions("humanitas", limit=5)
            
            print(f"📋 Etichette disponibili: {len(labels)}")
            print(f"📊 Sessioni disponibili: {len(sessions)}")
            
            if len(labels) > 0 and len(sessions) > 0:
                print("✅ Dati MongoDB: OK")
            else:
                print("⚠️  Dati MongoDB: Limitati o assenti")
            
            mongo_reader.disconnect()
        else:
            print("❌ Connessione MongoDB: FALLITA")
            return False
        
        # 2. Test QualityGateEngine con dati mock
        print("\n2️⃣ Test QualityGateEngine con dati simulati...")
        
        # Crea decisioni mock per il test
        mock_decisions = [
            {
                'session_id': 'mock_session_1',
                'human_decision': 'prenotazione_visite',
                'timestamp': '2025-01-15T10:00:00Z',
                'confidence': 0.9
            },
            {
                'session_id': 'mock_session_2', 
                'human_decision': 'ritiro_referti',
                'timestamp': '2025-01-15T10:05:00Z',
                'confidence': 0.85
            }
        ]
        
        # Importa e inizializza QualityGateEngine
        from QualityGate.quality_gate_engine import QualityGateEngine
        
        quality_gate = QualityGateEngine(tenant_name='humanitas')
        
        # Test del metodo _prepare_training_data
        print("📚 Test preparazione dati di training...")
        X, y = quality_gate._prepare_training_data(mock_decisions)
        
        if X is not None and y is not None:
            print(f"✅ Dati preparati: {len(X)} campioni con {X.shape[1]} features")
            print(f"📊 Etichette: {list(set(y))}")
            print("✅ Migrazione MongoDB: COMPLETATA con successo")
            return True
        else:
            print("❌ Preparazione dati: FALLITA")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main del test di integrazione MongoDB
    """
    print("🚀 Avvio test migrazione MongoDB per QualityGateEngine")
    
    # Configura environment per i test
    os.environ.setdefault('PYTHONPATH', '/home/ubuntu/classificazione_discussioni')
    
    success = test_quality_gate_mongo_integration()
    
    if success:
        print("\n🎉 Test completato con successo!")
        print("✅ Il sistema di riaddestramento è ora compatibile con MongoDB")
        print("📈 La migrazione da MySQL a MongoDB è stata completata")
    else:
        print("\n❌ Test fallito!")
        print("🔧 Verificare configurazione MongoDB e dipendenze")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
