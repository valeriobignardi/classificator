#!/usr/bin/env python3
"""
Test completo della pipeline di classificazione con eredità campi
Autore: Valerio Bignardi
Data: 2024-12-28
Scopo: Verificare che i documenti propagati ereditino correttamente tutti i campi
"""

import sys
import os
sys.path.append(os.getcwd())

from mongo_classification_reader import MongoClassificationReader
from Models.documento_processing import DocumentoProcessing
try:
    from Utils.tenant_manager import TenantManager
except ImportError:
    from Utils.mock_tenant_manager import TenantManager

def test_complete_pipeline():
    """
    Test completo della pipeline con un nuovo documento
    """
    print("🧪 TEST PIPELINE COMPLETA")
    print("=" * 50)
    
    # 1. Inizializza il tenant
    tenant_manager = TenantManager()
    tenant = tenant_manager.get_tenant_by_slug('wopta')
    
    if not tenant:
        print("❌ Tenant Wopta non trovato")
        return False
    
    print(f"✅ Tenant trovato: {tenant.tenant_name}")
    
    # 2. Crea un documento di test
    doc_test = DocumentoProcessing(
        session_id="test_pipeline_001",
        testo_completo="Buongiorno, vorrei aprire una nuova polizza vita per mia moglie. Potete inviarmi il preventivo?",
        is_representative=False,
        cluster_id=1001
    )
    
    print(f"📄 Documento di test creato:")
    print(f"   ID: {doc_test.session_id}")
    print(f"   Testo: {doc_test.testo_completo[:80]}...")
    
    # 3. Simula un rappresentante del cluster con classificazione completa
    rappresentante = DocumentoProcessing(
        session_id="rappresentante_test_001",
        testo_completo="Salve, desidero stipulare una polizza vita per proteggere la mia famiglia.",
        is_representative=True,
        cluster_id=1001,
        predicted_label="RICHIESTA_STIPULA_POLIZZA_VITA",
        classification_method="ENSEMBLE",
        ml_prediction="RICHIESTA_STIPULA_POLIZZA_VITA",
        ml_confidence=0.92,
        llm_prediction="RICHIESTA_STIPULA_POLIZZA_VITA", 
        llm_confidence=0.88
    )
    
    print(f"👑 Rappresentante del cluster:")
    print(f"   ID: {rappresentante.session_id}")
    print(f"   Classification: {rappresentante.predicted_label}")
    print(f"   ML Prediction: {rappresentante.ml_prediction} (conf: {rappresentante.ml_confidence})")
    print(f"   LLM Prediction: {rappresentante.llm_prediction} (conf: {rappresentante.llm_confidence})")
    print(f"   Method: {rappresentante.classification_method}")
    
    # 4. Propaga la classificazione
    doc_test.set_as_propagated(
        propagated_from=rappresentante.cluster_id,
        propagated_label=rappresentante.predicted_label,
        consensus=0.9,
        reason="test_propagation",
        ml_prediction=rappresentante.ml_prediction,
        ml_confidence=rappresentante.ml_confidence,
        llm_prediction=rappresentante.llm_prediction,
        llm_confidence=rappresentante.llm_confidence,
        classification_method=rappresentante.classification_method
    )
    
    print(f"🔄 Dopo propagazione:")
    print(f"   Classification: {doc_test.predicted_label}")
    print(f"   ML Prediction: {doc_test.ml_prediction} (conf: {doc_test.ml_confidence})")
    print(f"   LLM Prediction: {doc_test.llm_prediction} (conf: {doc_test.llm_confidence})")
    print(f"   Method: {doc_test.classification_method}")
    print(f"   Is Propagated: {doc_test.is_propagated}")
    
    # 5. Salva in MongoDB
    mongo_reader = MongoClassificationReader(tenant)
    
    final_decision = doc_test.to_classification_decision()
    print(f"📋 Final decision generato:")
    for key, value in final_decision.items():
        print(f"   {key}: {value}")
    
    # Salva il documento
    result = mongo_reader.save_classification_result(
        session_id=doc_test.session_id,
        client_name="Test Client",
        final_decision=final_decision,
        conversation_text=doc_test.testo_completo,
        needs_review=False,
        classified_by="test_pipeline"
    )
    
    if result:
        print("✅ Documento salvato in MongoDB")
        
        # 6. Verifica lettura da MongoDB
        mongo_doc = mongo_reader.get_case_by_id(doc_test.session_id)
        
        if mongo_doc:
            print(f"🔍 Documento letto da MongoDB:")
            print(f"   ID: {mongo_doc.get('_id')}")
            print(f"   Classification: {mongo_doc.get('classification', 'N/A')}")
            print(f"   ML Prediction: {mongo_doc.get('ml_prediction', 'N/A')}")
            print(f"   LLM Prediction: {mongo_doc.get('llm_prediction', 'N/A')}")
            print(f"   Method: {mongo_doc.get('classification_method', 'N/A')}")
            
            # Verifica che tutti i campi siano presenti
            campi_richiesti = ['classification', 'ml_prediction', 'llm_prediction', 'classification_method']
            tutti_presenti = all(mongo_doc.get(campo) for campo in campi_richiesti)
            
            if tutti_presenti:
                print("🎉 SUCCESSO! Tutti i campi ereditati correttamente!")
                return True
            else:
                print("❌ Alcuni campi mancanti nel documento MongoDB")
                return False
        else:
            print("❌ Documento non trovato in MongoDB")
            return False
    else:
        print("❌ Errore nel salvataggio")
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n✅ TEST PIPELINE COMPLETO: SUCCESSO!")
    else:
        print("\n❌ TEST PIPELINE COMPLETO: FALLITO!")
