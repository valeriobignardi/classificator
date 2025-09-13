#!/usr/bin/env python3
"""
Test script per verificare fix propagazione campi classificazione
Author: Valerio Bignardi
Date: 2025-01-27
"""

import sys
import os
sys.path.append('.')
from Models.documento_processing import DocumentoProcessing

def test_propagation_fix():
    """
    Test per verificare che la propagazione erediti tutti i campi
    """
    print("=== TEST FIX PROPAGAZIONE CAMPI CLASSIFICAZIONE ===")
    
    # 1. Crea un documento rappresentante con tutti i campi
    rep_doc = DocumentoProcessing(
        session_id="rep_001",
        testo_completo="Testo rappresentante",
        predicted_label="RICHIESTA_STIPULA_POLIZZA",
        confidence=0.92,
        classification_method="ensemble",
        ml_prediction="RICHIESTA_STIPULA_POLIZZA",
        ml_confidence=0.88,
        llm_prediction="RICHIESTA_STIPULA_POLIZZA", 
        llm_confidence=0.95
    )
    
    print(f"🎯 Rappresentante creato:")
    print(f"   Classification: {rep_doc.predicted_label}")
    print(f"   ML: {rep_doc.ml_prediction} (conf: {rep_doc.ml_confidence})")
    print(f"   LLM: {rep_doc.llm_prediction} (conf: {rep_doc.llm_confidence})")
    print(f"   Method: {rep_doc.classification_method}")
    
    # 2. Crea un documento propagato
    prop_doc = DocumentoProcessing(
        session_id="prop_001", 
        testo_completo="Testo propagato"
    )
    
    # 3. Applica propagazione con nuovi parametri
    prop_doc.set_as_propagated(
        propagated_from=1,
        propagated_label=rep_doc.predicted_label,
        consensus=0.8,
        ml_prediction=rep_doc.ml_prediction,
        ml_confidence=rep_doc.ml_confidence,
        llm_prediction=rep_doc.llm_prediction,
        llm_confidence=rep_doc.llm_confidence,
        classification_method=rep_doc.classification_method
    )
    
    print(f"\n🔄 Propagato creato:")
    print(f"   Classification: {prop_doc.predicted_label}")
    print(f"   ML: {prop_doc.ml_prediction} (conf: {prop_doc.ml_confidence})")
    print(f"   LLM: {prop_doc.llm_prediction} (conf: {prop_doc.llm_confidence})")
    print(f"   Method: {prop_doc.classification_method}")
    
    # 4. Test to_classification_decision
    decision = prop_doc.to_classification_decision()
    print(f"\n📊 Classification Decision:")
    for key, value in decision.items():
        print(f"   {key}: {value}")
    
    # 5. Verifica che tutti i campi siano ereditati
    errors = []
    if prop_doc.ml_prediction != rep_doc.ml_prediction:
        errors.append(f"ML prediction non ereditata: {prop_doc.ml_prediction} != {rep_doc.ml_prediction}")
    if prop_doc.llm_prediction != rep_doc.llm_prediction:
        errors.append(f"LLM prediction non ereditata: {prop_doc.llm_prediction} != {rep_doc.llm_prediction}")
    if prop_doc.ml_confidence != rep_doc.ml_confidence:
        errors.append(f"ML confidence non ereditata: {prop_doc.ml_confidence} != {rep_doc.ml_confidence}")
    if prop_doc.llm_confidence != rep_doc.llm_confidence:
        errors.append(f"LLM confidence non ereditata: {prop_doc.llm_confidence} != {rep_doc.llm_confidence}")
    
    if errors:
        print(f"\n❌ ERRORI RILEVATI:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print(f"\n✅ TUTTI I CAMPI EREDITATI CORRETTAMENTE!")
        return True

if __name__ == "__main__":
    success = test_propagation_fix()
    sys.exit(0 if success else 1)
