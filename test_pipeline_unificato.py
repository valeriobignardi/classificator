#!/usr/bin/env python3
"""
File: test_pipeline_unificato.py
Autore: Valerio Bignardi
Data: 2025-09-08
Descrizione: Test completo del pipeline unificato con DocumentoProcessing

Storia delle modifiche:
2025-09-08 - Test implementazione completa pipeline unificato
"""

import sys
import os
import json
from datetime import datetime

# Aggiungi percorsi per gli import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))

def create_test_sessions():
    """
    Crea sessioni di test per verificare il pipeline
    
    Returns:
        Dict: Sessioni di test simulate
    """
    test_sessions = {}
    
    # Simula diverse tipologie di conversazioni
    conversations = [
        {
            "id": "test_001",
            "text": "Ciao, ho bisogno di informazioni sui vostri servizi di riabilitazione. Potreste darmi dettagli sui costi e tempi di attesa?",
            "category": "richiesta_informazioni"
        },
        {
            "id": "test_002", 
            "text": "Salve, vorrei prenotare una visita cardiologica per la prossima settimana. È possibile avere un appuntamento?",
            "category": "prenotazione_visita"
        },
        {
            "id": "test_003",
            "text": "Buongiorno, ho fatto gli esami del sangue la settimana scorsa. Come faccio a ritirare i risultati?",
            "category": "ritiro_referti"
        },
        {
            "id": "test_004",
            "text": "Mi serve assistenza per il pagamento della fattura. Non riesco ad accedere al portale online.",
            "category": "supporto_tecnico"
        },
        {
            "id": "test_005",
            "text": "Vorrei informazioni sui servizi di fisioterapia disponibili nella vostra struttura.",
            "category": "richiesta_informazioni"
        },
        {
            "id": "test_006",
            "text": "Devo disdire la mia prenotazione di domani per la visita oculistica. Come posso fare?",
            "category": "disdetta_appuntamento"
        },
        {
            "id": "test_007",
            "text": "Ho bisogno di una copia del mio referto di radiologia. È possibile averlo via email?",
            "category": "ritiro_referti"
        },
        {
            "id": "test_008", 
            "text": "Salve, vorrei prenotare una visita ortopedica urgente. Ho dolore al ginocchio da giorni.",
            "category": "prenotazione_visita"
        },
        {
            "id": "test_009",
            "text": "Problemi con l'app mobile, non riesco a vedere i miei appuntamenti. Potete aiutarmi?",
            "category": "supporto_tecnico"
        },
        {
            "id": "test_010",
            "text": "Informazioni sui costi delle visite private e tempi per prenotazione cardiologia.",
            "category": "richiesta_informazioni"
        },
        {
            "id": "test_011",
            "text": "C'è stata una comunicazione poco chiara da parte del personale. Vorrei fare un reclamo formale.",
            "category": "reclamo"
        },
        {
            "id": "test_012",
            "text": "Non trovo più il mio numero di tessera sanitaria. Posso prenotare lo stesso?",
            "category": "altro"
        }
    ]
    
    for conv in conversations:
        test_sessions[conv["id"]] = {
            'testo_completo': conv["text"],
            'session_id': conv["id"],
            'expected_category': conv["category"],
            'timestamp': datetime.now().isoformat()
        }
    
    return test_sessions

def test_clustering_unificato():
    """
    Test del clustering con oggetti DocumentoProcessing
    """
    print("🧪 [TEST 1] Clustering unificato con DocumentoProcessing")
    
    try:
        # Import necessari
        print("   📁 Caricamento moduli...")
        
        # Aggiunge percorso Models
        import sys
        sys.path.append('Models')
        from documento_processing import DocumentoProcessing
        
        # Crea sessioni di test
        test_sessions = create_test_sessions()
        print(f"   📊 Create {len(test_sessions)} sessioni di test")
        
        # Simula risultato clustering
        documenti = []
        cluster_assignments = [1, 2, 3, 1, 1, 4, 3, 2, 1, 1, -1, -1]  # Alcuni cluster + outlier
        
        for i, (session_id, session_data) in enumerate(test_sessions.items()):
            doc = DocumentoProcessing(
                session_id=session_id,
                testo_completo=session_data['testo_completo']
            )
            
            cluster_id = cluster_assignments[i % len(cluster_assignments)]
            cluster_size = cluster_assignments.count(cluster_id) if cluster_id != -1 else 0
            is_outlier = (cluster_id == -1)
            
            doc.set_clustering_info(cluster_id, cluster_size, is_outlier)
            documenti.append(doc)
        
        # Simula selezione rappresentanti
        cluster_docs = {}
        for doc in documenti:
            if not doc.is_outlier:
                if doc.cluster_id not in cluster_docs:
                    cluster_docs[doc.cluster_id] = []
                cluster_docs[doc.cluster_id].append(doc)
        
        # Seleziona 1 rappresentante per cluster
        for cluster_id, docs in cluster_docs.items():
            docs[0].set_as_representative(f"test_selection_cluster_{cluster_id}")
        
        # Applica propagazione
        for cluster_id, docs in cluster_docs.items():
            for doc in docs[1:]:  # Non-rappresentanti
                doc.set_as_propagated(
                    propagated_from=cluster_id,
                    propagated_label=f"categoria_cluster_{cluster_id}",
                    consensus=0.8,
                    reason="test_propagation"
                )
        
        # Statistiche
        rappresentanti = [doc for doc in documenti if doc.is_representative]
        propagati = [doc for doc in documenti if doc.is_propagated]
        outliers = [doc for doc in documenti if doc.is_outlier]
        
        print(f"   ✅ Clustering simulato completato:")
        print(f"     📊 Documenti totali: {len(documenti)}")
        print(f"     👥 Rappresentanti: {len(rappresentanti)}")
        print(f"     🔄 Propagati: {len(propagati)}")
        print(f"     🔍 Outliers: {len(outliers)}")
        
        # Test metadati
        if rappresentanti:
            rep = rappresentanti[0]
            print(f"   🔍 Test metadati rappresentante {rep.session_id}:")
            print(f"     - Tipo: {rep.get_document_type()}")
            print(f"     - Needs review: {rep.needs_review}")
            print(f"     - Cluster ID: {rep.cluster_id}")
            
            mongo_meta = rep.to_mongo_metadata()
            print(f"     - MongoDB meta keys: {list(mongo_meta.keys())}")
        
        if propagati:
            prop = propagati[0]
            print(f"   🔍 Test metadati propagato {prop.session_id}:")
            print(f"     - Tipo: {prop.get_document_type()}")
            print(f"     - Label propagata: {prop.propagated_label}")
            print(f"     - Needs review: {prop.needs_review}")
        
        return True, documenti
        
    except Exception as e:
        print(f"   ❌ ERRORE nel test clustering: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def test_selezione_rappresentanti(documenti):
    """
    Test selezione rappresentanti con limite
    """
    print("\n🧪 [TEST 2] Selezione rappresentanti con limite")
    
    try:
        # Simula la funzione di selezione
        rappresentanti_totali = [doc for doc in documenti if doc.is_representative]
        
        print(f"   📊 Rappresentanti totali disponibili: {len(rappresentanti_totali)}")
        
        # Test selezione con limite
        max_budget = 3
        print(f"   🎯 Budget massimo: {max_budget}")
        
        # Simula selezione intelligente (ordina per confidence o cluster size)
        selected = sorted(rappresentanti_totali, 
                         key=lambda d: (d.cluster_size or 0, d.confidence or 0.5), 
                         reverse=True)[:max_budget]
        
        print(f"   ✅ Rappresentanti selezionati: {len(selected)}")
        
        for i, doc in enumerate(selected):
            print(f"     {i+1}. {doc.session_id} - Cluster {doc.cluster_id} "
                  f"(size: {doc.cluster_size})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE nel test selezione: {e}")
        return False

def test_classificazione_salvataggio(documenti):
    """
    Test classificazione e salvataggio (simulato)
    """
    print("\n🧪 [TEST 3] Classificazione e salvataggio (simulato)")
    
    try:
        # Simula classificazione per documenti che ne hanno bisogno
        docs_to_classify = [doc for doc in documenti 
                           if (doc.is_representative or doc.is_outlier or 
                               (doc.is_propagated and not doc.predicted_label))]
        
        print(f"   📊 Documenti da classificare: {len(docs_to_classify)}")
        
        # Simula risultati classificazione
        labels = ["richiesta_informazioni", "prenotazione_visita", "ritiro_referti", 
                 "supporto_tecnico", "altro"]
        
        for i, doc in enumerate(docs_to_classify):
            label = labels[i % len(labels)]
            confidence = 0.7 + (i % 3) * 0.1  # Varia tra 0.7-0.9
            
            doc.set_classification_result(
                predicted_label=label,
                confidence=confidence,
                method="simulated_ensemble",
                reasoning=f"Test classification for {doc.get_document_type()}"
            )
        
        print(f"   ✅ Classificazione simulata completata")
        
        # Test preparazione dati per salvataggio
        saved_count = 0
        for doc in documenti:
            # Simula salvataggio
            mongo_meta = doc.to_mongo_metadata()
            classification_decision = doc.to_classification_decision()
            review_info = doc.get_review_info()
            
            # Verifica che i metadati siano completi
            if mongo_meta and classification_decision and review_info:
                saved_count += 1
        
        print(f"   ✅ Salvataggio simulato: {saved_count}/{len(documenti)} documenti")
        
        # Statistiche finali per tipo
        saved_reps = sum(1 for doc in documenti if doc.is_representative)
        saved_props = sum(1 for doc in documenti if doc.is_propagated)
        saved_outliers = sum(1 for doc in documenti if doc.is_outlier)
        
        print(f"   📊 Breakdown salvataggio:")
        print(f"     👥 Rappresentanti: {saved_reps}")
        print(f"     🔄 Propagati: {saved_props}")
        print(f"     🔍 Outliers: {saved_outliers}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERRORE nel test salvataggio: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_pipeline_test():
    """
    Esegue il test completo del pipeline unificato
    """
    print("🚀 [TEST PIPELINE UNIFICATO] Inizio test completo")
    print("=" * 80)
    
    results = []
    documenti = []
    
    # Test 1: Clustering
    success, documenti = test_clustering_unificato()
    results.append(success)
    
    if success and documenti:
        # Test 2: Selezione rappresentanti
        success = test_selezione_rappresentanti(documenti)
        results.append(success)
        
        # Test 3: Classificazione e salvataggio
        success = test_classificazione_salvataggio(documenti)
        results.append(success)
    else:
        print("❌ Test clustering fallito - saltando test successivi")
        results.extend([False, False])
    
    # Risultati finali
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 80)
    print(f"📊 [RISULTATI FINALI TEST PIPELINE]")
    print(f"   ✅ Test passati: {passed}/{total}")
    print(f"   📈 Percentuale successo: {passed/total*100:.1f}%")
    
    if passed == total:
        print(f"   🎉 TUTTI I TEST PASSATI - Pipeline unificato funziona correttamente!")
    else:
        print(f"   ⚠️ {total-passed} test falliti - Necessarie correzioni")
    
    print("=" * 80)
    
    return passed == total

if __name__ == "__main__":
    success = run_full_pipeline_test()
    exit(0 if success else 1)
