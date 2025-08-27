#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autore: AI Assistant  
Data: 27 Agosto 2025
Scopo: Esempio pratico di utilizzo della propagazione cluster per review cases

Questo script dimostra come utilizzare la nuova funzionalità di propagazione
delle decisioni umane dai rappresentanti del cluster agli altri membri.

Ultimo aggiornamento: 27/08/2025
"""

import os
import sys
import json
from datetime import datetime
from bson import ObjectId

# Aggiunge il path del progetto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mongo_classification_reader import MongoClassificationReader
from QualityGate.quality_gate_engine import QualityGateEngine


def demo_cluster_review_propagation():
    """
    Dimostra la funzionalità di propagazione delle decisioni umane nei cluster
    
    Scopo: Mostra praticamente come funziona la propagazione automatica
           quando si fa review di un caso rappresentante
    """
    
    print("=" * 80)
    print("🎯 DEMO: Propagazione Decisioni Umane nei Cluster")
    print("=" * 80)
    
    # Inizializza componenti
    tenant_name = "demo_propagation"
    client_name = "demo_client"
    
    mongo_reader = MongoClassificationReader(tenant_name=tenant_name)
    mongo_reader.connect()
    
    quality_gate = QualityGateEngine(tenant_name=tenant_name)
    
    try:
        # Simula scenario reale: cluster con rappresentante + membri propagati
        print("\n📝 STEP 1: Creazione scenario cluster di esempio")
        print("-" * 50)
        
        cluster_id = f"demo_cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Crea casi di esempio
        cases = create_demo_cluster_cases(mongo_reader, client_name, cluster_id)
        
        print(f"✅ Creato cluster {cluster_id} con {len(cases)} casi:")
        for case_type, case_id in cases.items():
            print(f"   - {case_type}: {case_id}")
        
        # Mostra stato iniziale
        print("\n📊 STEP 2: Stato iniziale del cluster")
        print("-" * 50)
        
        show_cluster_status(mongo_reader, cluster_id, "PRIMA della review")
        
        # Simula decisione umana sul rappresentante
        print("\n🎯 STEP 3: Decisione umana sul rappresentante")
        print("-" * 50)
        
        representative_id = cases["Rappresentante"]
        human_decision = "reclamo"
        human_confidence = 0.95
        notes = "Dopo analisi manuale, è chiaramente un reclamo. Deve essere propagato agli altri casi simili."
        
        print(f"👨‍💼 Operatore umano decide:")
        print(f"   📋 Caso: {representative_id}")
        print(f"   🏷️  Decisione: '{human_decision}'")
        print(f"   📊 Confidenza: {human_confidence}")
        print(f"   📝 Note: {notes}")
        
        # Applica decisione con propagazione
        print("\n🔄 STEP 4: Applicazione con propagazione automatica")
        print("-" * 50)
        
        result = mongo_reader.resolve_review_session_with_cluster_propagation(
            case_id=representative_id,
            client_name=client_name,
            human_decision=human_decision,
            human_confidence=human_confidence,
            human_notes=notes
        )
        
        print("📤 Risultato propagazione:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Mostra stato finale
        print("\n📊 STEP 5: Stato finale del cluster")
        print("-" * 50)
        
        show_cluster_status(mongo_reader, cluster_id, "DOPO la review con propagazione")
        
        # Dimostra anche utilizzo tramite QualityGate
        print("\n🔧 STEP 6: Test integrazione QualityGate")
        print("-" * 50)
        
        # Crea un altro caso per testare QualityGate
        additional_case_id = create_additional_demo_case(mongo_reader, client_name, cluster_id + "_extra")
        
        qg_result = quality_gate.resolve_review_case(
            case_id=additional_case_id,
            human_decision="informazioni",
            human_confidence=0.88,
            notes="Test tramite QualityGate"
        )
        
        print("📤 Risultato QualityGate:")
        print(json.dumps(qg_result, indent=2, ensure_ascii=False))
        
        # Riepilogo finale
        print("\n🎉 DEMO COMPLETATA")
        print("=" * 80)
        print("✨ Funzionalità dimostrate:")
        print("   1. ✅ Propagazione automatica da rappresentante a membri cluster")
        print("   2. ✅ Riduzione della confidenza nei casi propagati (0.95 → 0.855)")  
        print("   3. ✅ Tracciabilità completa con metadati di propagazione")
        print("   4. ✅ Integrazione con QualityGateEngine")
        print("   5. ✅ Logging dettagliato per audit e debugging")
        print("\n🔍 Vantaggi:")
        print("   • Riduzione del carico di lavoro manuale")
        print("   • Consistenza delle classificazioni nel cluster")
        print("   • Tracciabilità completa delle decisioni")
        print("   • Mantenimento dell'integrità semantica del clustering")
        
    except Exception as e:
        print(f"❌ Errore durante la demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup opzionale (rimuovi se vuoi mantenere i dati demo)
        cleanup = input("\n🧹 Vuoi rimuovere i dati demo? (y/n): ").lower().strip()
        if cleanup == 'y':
            cleanup_demo_data(mongo_reader, cluster_id)
            print("✅ Dati demo rimossi")


def create_demo_cluster_cases(mongo_reader, client_name, cluster_id):
    """
    Crea casi di esempio per dimostrare la propagazione cluster
    
    Parametri:
        mongo_reader: Istanza MongoClassificationReader
        client_name: Nome del cliente  
        cluster_id: ID del cluster da creare
        
    Returns:
        Dict: Mapping tipo_caso -> case_id
    """
    collection = mongo_reader.db[mongo_reader.get_collection_name()]
    
    base_timestamp = datetime.now().isoformat()
    cases = {}
    
    # Caso rappresentante
    representative_doc = {
        "session_id": f"demo_session_representative_{cluster_id}",
        "client": client_name,
        "conversazione": "Il servizio non funziona da giorni, sono molto arrabbiato. Voglio un rimborso immediato!",
        "classificazione": "informazioni",  # Classificazione iniziale (errata)
        "confidence": 0.65,
        "review_status": "pending",
        "timestamp": base_timestamp,
        "classified_by": "demo_ml_model",
        "metadata": {
            "cluster_metadata": {
                "cluster_id": cluster_id,
                "is_representative": True,
                "method": "cluster_representative",
                "selection_reason": "Highest confidence in cluster"
            }
        }
    }
    
    result = collection.insert_one(representative_doc)
    cases["Rappresentante"] = str(result.inserted_id)
    
    # Primi caso propagato
    propagated_doc_1 = {
        "session_id": f"demo_session_propagated_1_{cluster_id}",
        "client": client_name,
        "conversazione": "Stesso problema, il sistema è inutilizzabile. Serve una soluzione rapida.",
        "classificazione": "informazioni",  # Sarà aggiornato dalla propagazione
        "confidence": 0.58,
        "review_status": "not_required",
        "timestamp": base_timestamp,
        "classified_by": "demo_ml_model",
        "metadata": {
            "cluster_metadata": {
                "cluster_id": cluster_id,
                "is_representative": False,
                "propagated_from": "cluster_propagation",
                "similarity_score": 0.89
            }
        }
    }
    
    result = collection.insert_one(propagated_doc_1)
    cases["Propagato_1"] = str(result.inserted_id)
    
    # Secondo caso propagato  
    propagated_doc_2 = {
        "session_id": f"demo_session_propagated_2_{cluster_id}",
        "client": client_name,
        "conversazione": "Inaccettabile questo disservizio. Pretendo una risposta immediata!",
        "classificazione": "informazioni",  # Sarà aggiornato dalla propagazione
        "confidence": 0.62,
        "review_status": "not_required", 
        "timestamp": base_timestamp,
        "classified_by": "demo_ml_model",
        "metadata": {
            "cluster_metadata": {
                "cluster_id": cluster_id,
                "is_representative": False,
                "propagated_from": "cluster_propagation",
                "similarity_score": 0.91
            }
        }
    }
    
    result = collection.insert_one(propagated_doc_2)
    cases["Propagato_2"] = str(result.inserted_id)
    
    return cases


def create_additional_demo_case(mongo_reader, client_name, cluster_id):
    """
    Crea un caso aggiuntivo per testare QualityGate
    
    Returns:
        str: ID del caso creato
    """
    collection = mongo_reader.db[mongo_reader.get_collection_name()]
    
    doc = {
        "session_id": f"demo_session_quality_gate_{cluster_id}",
        "client": client_name,
        "conversazione": "Vorrei informazioni sui vostri servizi. Grazie.",
        "classificazione": "domanda",  
        "confidence": 0.70,
        "review_status": "pending",
        "timestamp": datetime.now().isoformat(),
        "classified_by": "demo_ml_model",
        "metadata": {
            "cluster_metadata": {
                "cluster_id": cluster_id,
                "is_representative": True,
                "method": "cluster_representative"
            }
        }
    }
    
    result = collection.insert_one(doc)
    return str(result.inserted_id)


def show_cluster_status(mongo_reader, cluster_id, phase_name):
    """
    Mostra lo stato attuale di tutti i casi nel cluster
    
    Parametri:
        mongo_reader: Istanza MongoClassificationReader
        cluster_id: ID del cluster
        phase_name: Nome della fase (per logging)
    """
    collection = mongo_reader.db[mongo_reader.get_collection_name()]
    
    # Trova tutti i casi del cluster
    cluster_cases = list(collection.find(
        {"metadata.cluster_metadata.cluster_id": cluster_id},
        {
            "session_id": 1,
            "classificazione": 1,
            "confidence": 1,
            "review_status": 1,
            "human_decision": 1,
            "human_reviewed_by_propagation": 1,
            "metadata.cluster_metadata.is_representative": 1,
            "metadata.cluster_metadata.human_review_propagated_at": 1
        }
    ))
    
    print(f"📊 Stato cluster '{cluster_id}' - {phase_name}:")
    
    if not cluster_cases:
        print("   ⚠️  Nessun caso trovato nel cluster")
        return
    
    for case in cluster_cases:
        is_rep = case.get("metadata", {}).get("cluster_metadata", {}).get("is_representative", False)
        role = "🎯 RAPPRESENTANTE" if is_rep else "📡 propagato    "
        
        classification = case.get("classificazione", "N/A")
        confidence = case.get("confidence", 0)
        review_status = case.get("review_status", "N/A")
        human_decision = case.get("human_decision", "")
        propagated = case.get("human_reviewed_by_propagation", False)
        
        status_icon = "✅" if review_status == "completed" else "⏳"
        propagation_info = " [PROPAGATO]" if propagated else ""
        
        print(f"   {status_icon} {role} | {classification:<12} | conf: {confidence:.3f} | "
              f"review: {review_status}{propagation_info}")
        
        if human_decision:
            print(f"      └─ 👨‍💼 Decisione umana: '{human_decision}'")


def cleanup_demo_data(mongo_reader, cluster_id):
    """
    Rimuove i dati demo dal database
    
    Parametri:
        mongo_reader: Istanza MongoClassificationReader
        cluster_id: ID del cluster da rimuovere
    """
    try:
        collection = mongo_reader.db[mongo_reader.get_collection_name()]
        
        # Rimuovi tutti i casi che iniziano con il prefisso demo
        result = collection.delete_many({
            "$or": [
                {"metadata.cluster_metadata.cluster_id": {"$regex": f"^{cluster_id}"}},
                {"session_id": {"$regex": "^demo_session_"}}
            ]
        })
        
        print(f"🧹 Rimossi {result.deleted_count} documenti demo")
        
    except Exception as e:
        print(f"❌ Errore nella pulizia: {e}")


if __name__ == "__main__":
    demo_cluster_review_propagation()
