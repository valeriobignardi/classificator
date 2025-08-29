#!/usr/bin/env python3
"""
Author: Valerio Bignardi
Date: 2025-08-29
Description: Test per verificare che il fix del metodo _determine_classification_type funzioni
Last Update: 2025-08-29

Test del fix per l'errore 'MongoClassificationReader' object has no attribute '_determine_classification_type'
"""

from mongo_classification_reader import MongoClassificationReader
import datetime

def test_save_classification_result():
    """
    Test del salvataggio risultato classificazione per verificare il fix
    """
    print("🧪 Test salvataggio classificazione con metadati cluster...")
    
    # Crea reader
    reader = MongoClassificationReader(
        tenant_name="humanitas"
    )
    
    try:
        # Test 1: Metodo esiste
        print("\n1️⃣ Test esistenza metodo...")
        assert hasattr(reader, '_determine_classification_type'), "❌ Metodo non trovato"
        print("✅ Metodo _determine_classification_type trovato")
        
        # Test 2: Metodo funziona con diversi metadata
        print("\n2️⃣ Test funzionamento metodo...")
        
        # RAPPRESENTANTE
        repr_metadata = {'is_representative': True, 'cluster_id': 10}
        result = reader._determine_classification_type(repr_metadata)
        assert result == "RAPPRESENTANTE", f"❌ Expected RAPPRESENTANTE, got {result}"
        print(f"✅ Rappresentante: {result}")
        
        # PROPAGATO
        prop_metadata = {'propagated_from': 'session_123', 'cluster_id': 10}
        result = reader._determine_classification_type(prop_metadata)
        assert result == "PROPAGATO", f"❌ Expected PROPAGATO, got {result}"
        print(f"✅ Propagato: {result}")
        
        # OUTLIER
        outlier_metadata = {'cluster_id': -1, 'outlier_score': 0.8}
        result = reader._determine_classification_type(outlier_metadata)
        assert result == "OUTLIER", f"❌ Expected OUTLIER, got {result}"
        print(f"✅ Outlier: {result}")
        
        # NORMALE (no metadata)
        result = reader._determine_classification_type(None)
        assert result == "NORMALE", f"❌ Expected NORMALE, got {result}"
        print(f"✅ Normale: {result}")
        
        print("\n3️⃣ Test save_classification_result completo...")
        
        # Test del salvataggio completo (questo causava l'errore originale)
        test_session_id = f"test_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success = reader.save_classification_result(
            session_id=test_session_id,
            client_name="humanitas",
            ml_result={"predicted_label": "richiesta_info", "confidence": 0.85},
            llm_result={"predicted_label": "richiesta_info", "confidence": 0.90},
            final_decision={"predicted_label": "richiesta_info", "confidence": 0.87, "method": "ensemble"},
            conversation_text="Vorrei sapere informazioni sui vostri servizi",
            needs_review=False,
            cluster_metadata={
                'cluster_id': 15,
                'is_representative': True,
                'confidence': 0.92
            },
            classified_by="test_fix"
        )
        
        print(f"✅ Salvataggio classificazione completato: {success}")
        
        # Cleanup: rimuovi il record di test
        if reader.connect():
            collection = reader.db[reader.get_collection_name()]
            collection.delete_one({"session_id": test_session_id})
            reader.disconnect()
            print("✅ Pulizia record di test completata")
        
        print("\n🎉 TUTTI I TEST PASSATI! Il fix è funzionante!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FALLITO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_save_classification_result()
    if success:
        print("\n✅ Fix verificato con successo!")
    else:
        print("\n❌ Fix non funzionante, rivedere il codice")
