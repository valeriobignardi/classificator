#!/usr/bin/env python3
"""
File: test_bug_rappresentanti_sovrascritti.py
Autore: Valerio Bignardi
Data creazione: 2025-01-31
Descrizione: Test per dimostrare il bug dei rappresentanti sovrascritti

Scopo: 
- Creare un documento rappresentante e uno outlier
- Testare il salvataggio in MongoDB
- Verificare se i metadati vengono sovrascritti
- Debug completo del flusso di salvataggio
"""

import sys
import os
import json
from datetime import datetime
from pymongo import MongoClient
import yaml

# Aggiungi path per import
sys.path.append(os.path.dirname(__file__))

from Models.documento_processing import DocumentoProcessing
from mongo_classification_reader import MongoClassificationReader
from Utils.tenant import Tenant

def load_config():
    """Carica configurazione dal file config.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Errore caricamento config: {e}")
        return None

def create_test_tenant():
    """
    Crea un tenant di test per Wopta
    
    Returns:
        Tenant: Oggetto tenant di test
    """
    return Tenant(
        tenant_id="16c222a9-f293-11ef-9315-96000228e7fe",
        tenant_name="Wopta",
        tenant_slug="wopta",
        tenant_database="wopta_16c222a9-f293-11ef-9315-96000228e7fe",
        tenant_status=1
    )

def create_test_representative():
    """
    Crea un documento rappresentante di test
    
    Returns:
        DocumentoProcessing: Documento rappresentante
    """
    print(f"\n🧪 [TEST] Creazione documento RAPPRESENTANTE...")
    
    doc = DocumentoProcessing(
        session_id="test_repr_12345",
        testo_completo="Vorrei informazioni sulla polizza vita e i benefici in caso di morte",
        embedding=[0.1] * 768  # Embedding fake
    )
    
    # Imposta come rappresentante
    doc.cluster_id = 42
    doc.cluster_size = 25
    doc.is_representative = True
    doc.is_outlier = False
    doc.is_propagated = False
    doc.selection_reason = "alta_centralità"
    doc.predicted_label = "INFO_CASO_MORTE"
    doc.confidence = 0.95
    doc.classification_method = "intelligent_clustering_representative"
    doc.reasoning = "Richiesta informazioni benefici caso morte"
    
    print(f"   ✅ Rappresentante creato:")
    print(f"      session_id: {doc.session_id}")
    print(f"      cluster_id: {doc.cluster_id}")
    print(f"      is_representative: {doc.is_representative}")
    print(f"      is_outlier: {doc.is_outlier}")
    print(f"      predicted_label: {doc.predicted_label}")
    print(f"      confidence: {doc.confidence}")
    
    return doc

def create_test_outlier():
    """
    Crea un documento outlier di test
    
    Returns:
        DocumentoProcessing: Documento outlier
    """
    print(f"\n🧪 [TEST] Creazione documento OUTLIER...")
    
    doc = DocumentoProcessing(
        session_id="test_outlier_67890",
        testo_completo="Ciao, volevo solo salutare e augurare buona giornata",
        embedding=[0.2] * 768  # Embedding fake
    )
    
    # Imposta come outlier
    doc.cluster_id = -1
    doc.cluster_size = 1
    doc.is_representative = False
    doc.is_outlier = True
    doc.is_propagated = False
    doc.outlier_score = 0.9
    doc.predicted_label = "ALTRO"
    doc.confidence = 0.1
    doc.classification_method = "intelligent_clustering_outlier"
    doc.reasoning = "Testo non correlato al dominio assicurativo"
    
    print(f"   ✅ Outlier creato:")
    print(f"      session_id: {doc.session_id}")
    print(f"      cluster_id: {doc.cluster_id}")
    print(f"      is_representative: {doc.is_representative}")
    print(f"      is_outlier: {doc.is_outlier}")
    print(f"      predicted_label: {doc.predicted_label}")
    print(f"      confidence: {doc.confidence}")
    
    return doc

def test_to_mongo_metadata(doc):
    """
    Testa la funzione to_mongo_metadata
    
    Args:
        doc: Documento da testare
    """
    print(f"\n🔍 [TEST] Testing to_mongo_metadata per {doc.session_id}...")
    
    metadata = doc.to_mongo_metadata()
    
    print(f"   📋 Metadati generati:")
    for key, value in metadata.items():
        print(f"      {key}: {value} (type: {type(value).__name__})")
    
    return metadata

def test_save_classification_result(tenant, doc, mongo_reader):
    """
    Testa il salvataggio con debug completo
    
    Args:
        tenant: Oggetto tenant
        doc: Documento da salvare
        mongo_reader: Reader MongoDB
    """
    print(f"\n💾 [TEST] Testing save_classification_result per {doc.session_id}...")
    
    # Ottieni metadati
    cluster_metadata = doc.to_mongo_metadata()
    final_decision = doc.to_classification_decision()
    review_info = doc.get_review_info()
    
    print(f"   📋 INPUT PARAMETERS:")
    print(f"      session_id: {doc.session_id}")
    print(f"      client_name: {tenant.tenant_slug}")
    print(f"      final_decision: {final_decision}")
    print(f"      cluster_metadata: {cluster_metadata}")
    print(f"      needs_review: {review_info['needs_review']}")
    print(f"      review_reason: {review_info['review_reason']}")
    print(f"      classified_by: {review_info['classified_by']}")
    
    # Debug: verifica cosa viene passato
    print(f"\n🔍 [DEBUG] VERIFICA PARAMETRI PRIMA DEL SALVATAGGIO:")
    print(f"   cluster_metadata is None: {cluster_metadata is None}")
    print(f"   cluster_metadata keys: {list(cluster_metadata.keys()) if cluster_metadata else 'N/A'}")
    print(f"   is_representative in metadata: {'is_representative' in cluster_metadata if cluster_metadata else False}")
    if cluster_metadata and 'is_representative' in cluster_metadata:
        print(f"   is_representative value: {cluster_metadata['is_representative']}")
    
    # Salva in MongoDB
    success = mongo_reader.save_classification_result(
        session_id=doc.session_id,
        client_name=tenant.tenant_slug,
        final_decision=final_decision,
        conversation_text=doc.testo_completo,
        needs_review=review_info['needs_review'],
        review_reason=review_info['review_reason'],
        classified_by=review_info['classified_by'],
        notes=review_info['notes'],
        cluster_metadata=cluster_metadata,
        embedding=doc.embedding
    )
    
    print(f"   💾 Salvataggio risultato: {'SUCCESS' if success else 'FAILED'}")
    return success

def verify_saved_document(tenant, session_id, mongo_reader, expected_doc):
    """
    Verifica il documento salvato in MongoDB
    
    Args:
        tenant: Oggetto tenant
        session_id: ID sessione da verificare
        mongo_reader: Reader MongoDB
        expected_doc: Documento originale per confronto
    """
    print(f"\n🔍 [VERIFICA] Controllo documento salvato {session_id}...")
    
    try:
        # Connetti direttamente a MongoDB per verificare
        client = MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection_name = f"wopta_{tenant.tenant_id}"
        collection = db[collection_name]
        
        # Trova il documento
        doc = collection.find_one({"session_id": session_id})
        
        if not doc:
            print(f"   ❌ Documento {session_id} NON TROVATO!")
            return False
        
        print(f"   ✅ Documento trovato! Verifica metadati...")
        
        # Verifica campi principali
        print(f"\n   📋 CAMPI PRINCIPALI:")
        print(f"      session_id: {doc.get('session_id')}")
        print(f"      classification: {doc.get('classification')}")
        print(f"      confidence: {doc.get('confidence')}")
        print(f"      classification_method: {doc.get('classification_method')}")
        print(f"      classification_type: {doc.get('classification_type')}")
        print(f"      review_status: {doc.get('review_status')}")
        
        # Verifica metadati cluster
        metadata = doc.get('metadata', {})
        print(f"\n   🎯 METADATI CLUSTER:")
        print(f"      cluster_id: {metadata.get('cluster_id', 'NON PRESENTE')}")
        print(f"      is_representative: {metadata.get('is_representative', 'NON PRESENTE')}")
        print(f"      representative: {metadata.get('representative', 'NON PRESENTE')}")
        print(f"      outlier: {metadata.get('outlier', 'NON PRESENTE')}")
        print(f"      propagated: {metadata.get('propagated', 'NON PRESENTE')}")
        
        # Confronta con valori attesi
        print(f"\n   ⚖️  CONFRONTO ATTESO vs SALVATO:")
        expected_cluster_id = expected_doc.cluster_id
        expected_is_repr = expected_doc.is_representative
        expected_is_outlier = expected_doc.is_outlier
        
        saved_cluster_id = metadata.get('cluster_id')
        saved_is_repr = metadata.get('is_representative', False)
        saved_is_outlier = metadata.get('outlier', False)
        
        print(f"      cluster_id: {expected_cluster_id} → {saved_cluster_id} {'✅' if expected_cluster_id == saved_cluster_id else '❌'}")
        print(f"      is_representative: {expected_is_repr} → {saved_is_repr} {'✅' if expected_is_repr == saved_is_repr else '❌'}")
        print(f"      is_outlier: {expected_is_outlier} → {saved_is_outlier} {'✅' if expected_is_outlier == saved_is_outlier else '❌'}")
        
        # Verifica se c'è stata sovrascrittura
        bug_detected = False
        if expected_is_repr and not saved_is_repr:
            print(f"   🚨 BUG DETECTED: RAPPRESENTANTE SOVRASCRITTO!")
            bug_detected = True
        
        if expected_cluster_id != -1 and saved_cluster_id == -1:
            print(f"   🚨 BUG DETECTED: CLUSTER_ID SOVRASCRITTO!")
            bug_detected = True
        
        if not bug_detected:
            print(f"   ✅ Nessun bug rilevato per questo documento")
        
        client.close()
        return not bug_detected
        
    except Exception as e:
        print(f"   ❌ Errore verifica: {e}")
        return False

def cleanup_test_data(tenant):
    """
    Pulisce i dati di test
    
    Args:
        tenant: Oggetto tenant
    """
    print(f"\n🧹 [CLEANUP] Rimozione dati di test...")
    
    try:
        client = MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection_name = f"wopta_{tenant.tenant_id}"
        collection = db[collection_name]
        
        # Rimuovi documenti di test
        result = collection.delete_many({
            "session_id": {"$in": ["test_repr_12345", "test_outlier_67890"]}
        })
        
        print(f"   ✅ Rimossi {result.deleted_count} documenti di test")
        client.close()
        
    except Exception as e:
        print(f"   ❌ Errore cleanup: {e}")

def main():
    """
    Funzione principale del test
    """
    print(f"🧪 TEST BUG RAPPRESENTANTI SOVRASCRITTI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # 1. Setup
    config = load_config()
    if not config:
        return
    
    tenant = create_test_tenant()
    
    # 2. Crea documenti di test
    representative_doc = create_test_representative()
    outlier_doc = create_test_outlier()
    
    # 3. Testa to_mongo_metadata
    repr_metadata = test_to_mongo_metadata(representative_doc)
    outlier_metadata = test_to_mongo_metadata(outlier_doc)
    
    # 4. Crea MongoReader
    try:
        mongo_reader = MongoClassificationReader(tenant=tenant)
        print(f"\n✅ MongoClassificationReader inizializzato")
    except Exception as e:
        print(f"❌ Errore inizializzazione MongoReader: {e}")
        return
    
    # 5. Cleanup dati precedenti
    cleanup_test_data(tenant)
    
    # 6. Test salvataggio rappresentante
    print(f"\n{'='*80}")
    print(f"🧪 TEST 1: SALVATAGGIO RAPPRESENTANTE")
    print(f"{'='*80}")
    
    success_repr = test_save_classification_result(tenant, representative_doc, mongo_reader)
    
    # 7. Test salvataggio outlier
    print(f"\n{'='*80}")
    print(f"🧪 TEST 2: SALVATAGGIO OUTLIER")
    print(f"{'='*80}")
    
    success_outlier = test_save_classification_result(tenant, outlier_doc, mongo_reader)
    
    # 8. Verifica documenti salvati
    print(f"\n{'='*80}")
    print(f"🔍 VERIFICA DOCUMENTI SALVATI")
    print(f"{'='*80}")
    
    repr_ok = verify_saved_document(tenant, "test_repr_12345", mongo_reader, representative_doc)
    outlier_ok = verify_saved_document(tenant, "test_outlier_67890", mongo_reader, outlier_doc)
    
    # 9. Risultati finali
    print(f"\n{'='*80}")
    print(f"📊 RISULTATI FINALI")
    print(f"{'='*80}")
    
    print(f"✅ Salvataggio rappresentante: {'SUCCESS' if success_repr else 'FAILED'}")
    print(f"✅ Salvataggio outlier: {'SUCCESS' if success_outlier else 'FAILED'}")
    print(f"✅ Verifica rappresentante: {'PASS' if repr_ok else 'FAIL'}")
    print(f"✅ Verifica outlier: {'PASS' if outlier_ok else 'FAIL'}")
    
    if not repr_ok:
        print(f"\n🚨 BUG CONFERMATO: I rappresentanti vengono sovrascritti!")
        print(f"💡 Il problema è nella funzione save_classification_result")
        print(f"🔧 Soluzione: Correggere la logica quando cluster_metadata è None")
    else:
        print(f"\n✅ Test superato: I metadati sono stati salvati correttamente")
    
    # 10. Cleanup finale
    cleanup_test_data(tenant)
    
    print(f"\n🏁 Test completato!")

if __name__ == "__main__":
    main()
