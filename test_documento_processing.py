#!/usr/bin/env python3
"""
File: test_documento_processing_pipeline.py
Autore: Valerio Bignardi
Data: 2025-09-08
Descrizione: Test per verificare l'implementazione DocumentoProcessing nel pipeline

Storia delle modifiche:
2025-09-08 - Test implementazione oggetti unificati
"""

import sys
import os

# Aggiungi percorsi per gli import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))

def test_documento_processing_basic():
    """
    Test base della classe DocumentoProcessing
    
    Scopo: Verificare creazione e metodi base della classe
    """
    print("🧪 [TEST 1] Test DocumentoProcessing - Creazione base")
    
    try:
        from documento_processing import DocumentoProcessing
        
        # Test creazione oggetto base
        doc = DocumentoProcessing(
            session_id="test_123",
            testo_completo="Test conversazione per verifica funzionalità"
        )
        
        print(f"✅ Oggetto creato: {doc}")
        print(f"   📝 Session ID: {doc.session_id}")
        print(f"   📊 Tipo iniziale: {doc.get_document_type()}")
        
        # Test set clustering info
        doc.set_clustering_info(cluster_id=5, cluster_size=10, is_outlier=False)
        print(f"   🎯 Dopo clustering: Cluster {doc.cluster_id}, Size {doc.cluster_size}")
        
        # Test set as representative
        doc.set_as_representative("test_selection")
        print(f"   👥 Dopo representative: {doc.get_document_type()}")
        print(f"   ✅ Needs review: {doc.needs_review}")
        
        # Test conversione MongoDB
        mongo_meta = doc.to_mongo_metadata()
        print(f"   💾 Metadati MongoDB: {mongo_meta}")
        
        # Test decisione classificazione
        classification_decision = doc.to_classification_decision()
        print(f"   🎯 Decisione classificazione: {classification_decision}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE nel test base: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_documento_processing_propagation():
    """
    Test propagazione nella classe DocumentoProcessing
    
    Scopo: Verificare funzionamento propagazione intelligente
    """
    print("\n🧪 [TEST 2] Test DocumentoProcessing - Propagazione")
    
    try:
        from documento_processing import DocumentoProcessing
        
        # Test creazione documento per propagazione
        doc = DocumentoProcessing(
            session_id="test_prop_456", 
            testo_completo="Test documento per propagazione"
        )
        
        # Imposta clustering
        doc.set_clustering_info(cluster_id=3, cluster_size=8, is_outlier=False)
        
        # Imposta come propagato
        doc.set_as_propagated(
            propagated_from=3,
            propagated_label="richiesta_informazioni",
            consensus=0.85,
            reason="strong_consensus"
        )
        
        print(f"✅ Documento propagato creato: {doc}")
        print(f"   🔄 Tipo: {doc.get_document_type()}")
        print(f"   🏷️ Label propagata: {doc.propagated_label}")
        print(f"   🎯 Consensus: {doc.propagation_consensus}")
        print(f"   ❌ Needs review: {doc.needs_review}")  # Deve essere False
        print(f"   🔍 Reasoning: {doc.reasoning}")
        
        # Test metadati per salvataggio
        mongo_meta = doc.to_mongo_metadata()
        review_info = doc.get_review_info()
        classification_decision = doc.to_classification_decision()
        
        print(f"   💾 MongoDB meta: {mongo_meta}")
        print(f"   📝 Review info: {review_info}")
        print(f"   🎯 Classification: {classification_decision}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE nel test propagazione: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_documento_processing_outlier():
    """
    Test outlier nella classe DocumentoProcessing
    
    Scopo: Verificare gestione outlier
    """
    print("\n🧪 [TEST 3] Test DocumentoProcessing - Outlier")
    
    try:
        from documento_processing import DocumentoProcessing
        
        # Test creazione outlier
        doc = DocumentoProcessing(
            session_id="test_outlier_789",
            testo_completo="Test documento outlier"
        )
        
        # Imposta come outlier
        doc.set_clustering_info(cluster_id=-1, cluster_size=0, is_outlier=True)
        
        print(f"✅ Outlier creato: {doc}")
        print(f"   🔍 Tipo: {doc.get_document_type()}")
        print(f"   🎯 Cluster ID: {doc.cluster_id}")
        print(f"   🚫 Is outlier: {doc.is_outlier}")
        print(f"   📝 Selection reason: {doc.selection_reason}")
        
        # Gli outlier non sono né rappresentanti né propagati
        print(f"   👥 Is representative: {doc.is_representative}")
        print(f"   🔄 Is propagated: {doc.is_propagated}")
        
        # Test metadati
        mongo_meta = doc.to_mongo_metadata()
        print(f"   💾 Metadati MongoDB: {mongo_meta}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE nel test outlier: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """
    Esegue tutti i test
    
    Scopo: Verifica completa implementazione DocumentoProcessing
    """
    print("🚀 [INIZIO TEST] Verifica implementazione DocumentoProcessing")
    
    results = []
    
    # Test base
    results.append(test_documento_processing_basic())
    
    # Test propagazione
    results.append(test_documento_processing_propagation())
    
    # Test outlier
    results.append(test_documento_processing_outlier())
    
    # Statistiche finali
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 [RISULTATI TEST]")
    print(f"   ✅ Test passati: {passed}/{total}")
    print(f"   {'🎉 TUTTI I TEST PASSATI!' if passed == total else '❌ ALCUNI TEST FALLITI!'}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
