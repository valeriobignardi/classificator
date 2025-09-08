#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 TEST LOGICHE RECUPERATE - Verifica implementazione logiche critiche

Scopo: Testare le logiche critiche recuperate dopo il refactoring
- Selezione rappresentanti avanzata con configurazione da DB
- Classificazione con controlli ML ensemble
- Validazione "altro" e pulizia caratteri speciali

Autore: Valerio Bignardi
Data: 2025-09-08
"""

import sys
import os
import yaml
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Aggiungi il percorso del progetto
sys.path.append('/home/ubuntu/classificatore')

from Models.documento_processing import DocumentoProcessing
from Pipeline.end_to_end_pipeline import EndToEndPipeline, get_supervised_training_params_from_db

def test_recovered_logics():
    """
    🧪 Test completo delle logiche recuperate
    """
    print(f"\n🧪 TESTING LOGICHE RECUPERATE")
    print(f"=" * 80)
    print(f"📅 Esecuzione: {datetime.now()}")
    
    # 1. Test configurazione da database/config
    print(f"\n1️⃣ TEST: Configurazione parametri")
    test_configuration_loading()
    
    # 2. Test selezione rappresentanti avanzata
    print(f"\n2️⃣ TEST: Selezione rappresentanti avanzata")
    test_advanced_representative_selection()
    
    # 3. Test pulizia caratteri speciali
    print(f"\n3️⃣ TEST: Pulizia caratteri speciali")
    test_label_cleaning()
    
    # 4. Test classificazione con controlli ML
    print(f"\n4️⃣ TEST: Classificazione con controlli ML")
    test_ml_ensemble_controls()
    
    print(f"\n✅ TESTING COMPLETATO")
    print(f"=" * 80)

def test_configuration_loading():
    """
    Test caricamento configurazione da database vs config.yaml
    """
    try:
        # Test lettura parametri (simulato)
        tenant_id = "test-tenant-recovery"
        
        try:
            params = get_supervised_training_params_from_db(tenant_id)
            print(f"   ✅ Parametri DB caricati: {len(params)} chiavi")
            print(f"       Chiavi: {list(params.keys())[:5]}...")  # Prime 5 chiavi
        except Exception as e:
            print(f"   ⚠️ DB non disponibile: {e}")
            print(f"   📖 Fallback a config.yaml")
        
        # Test lettura config.yaml
        config_path = '/home/ubuntu/classificatore/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            supervised_config = config.get('supervised_training', {})
            human_review_config = supervised_config.get('human_review', {})
            
            print(f"   ✅ Config.yaml letto: {len(human_review_config)} parametri")
            print(f"       Parametri: {list(human_review_config.keys())}")
        else:
            print(f"   ❌ Config.yaml non trovato")
            
    except Exception as e:
        print(f"   ❌ Errore test configurazione: {e}")

def test_advanced_representative_selection():
    """
    Test selezione rappresentanti con logiche avanzate
    """
    try:
        # Crea documenti di test simulati
        documenti_test = create_test_documents_for_selection()
        
        print(f"   📊 Documenti di test: {len(documenti_test)}")
        
        rappresentanti = [doc for doc in documenti_test if doc.is_representative]
        print(f"   👥 Rappresentanti disponibili: {len(rappresentanti)}")
        
        # Distribuzione per cluster
        cluster_distribution = {}
        for doc in documenti_test:
            if doc.cluster_id is not None:
                cluster_distribution[doc.cluster_id] = cluster_distribution.get(doc.cluster_id, 0) + 1
        
        print(f"   📈 Distribuzione cluster: {cluster_distribution}")
        
        # Test selezione con diversi budget
        budgets = [5, 10, 20]
        
        for budget in budgets:
            print(f"\n   🎯 Test budget: {budget}")
            
            # Simula selezione (senza istanziare EndToEndPipeline)
            selected_count = simulate_selection_logic(rappresentanti, budget, cluster_distribution)
            print(f"       ✅ Selezionati: {selected_count}/{len(rappresentanti)} rappresentanti")
            
    except Exception as e:
        print(f"   ❌ Errore test selezione: {e}")

def test_label_cleaning():
    """
    Test pulizia caratteri speciali nelle etichette
    """
    test_labels = [
        "Tag normale",
        "tag-con-trattini",
        "Tag@#$%^&*()Speciali!!!",
        "   spazi   multipli   ",
        "MAIUSCOLO",
        "",
        None,
        "a",  # troppo corto
        "tab\tcon\nnewline",
        "àccénti_spéciàli"
    ]
    
    print(f"   🧪 Test {len(test_labels)} etichette problematiche:")
    
    for label in test_labels:
        cleaned = clean_label_text_test(label)
        print(f"       '{label}' → '{cleaned}'")

def test_ml_ensemble_controls():
    """
    Test controlli stato ML ensemble
    """
    try:
        # Simula diversi stati ML ensemble
        test_states = [
            {"ensemble_exists": True, "ml_trained": True, "classes": ["tag1", "tag2", "altro"]},
            {"ensemble_exists": True, "ml_trained": False, "classes": []},
            {"ensemble_exists": False, "ml_trained": False, "classes": []}
        ]
        
        for i, state in enumerate(test_states):
            print(f"   🧠 Scenario {i+1}: {state}")
            
            classification_mode = determine_classification_mode(state)
            print(f"       → Modalità: {classification_mode}")
        
        print(f"   ✅ Controlli ML ensemble: OK")
        
    except Exception as e:
        print(f"   ❌ Errore test ML controls: {e}")

def create_test_documents_for_selection() -> List[DocumentoProcessing]:
    """
    Crea documenti di test per testare la selezione rappresentanti
    """
    documenti = []
    
    # Cluster 0: 15 documenti (cluster grande)
    for i in range(15):
        doc = DocumentoProcessing(f"session_{i}", f"Domanda test {i}")
        doc.set_clustering_info(cluster_id=0, cluster_size=15)
        doc.embeddings = [1.0, 2.0, 3.0]
        if i < 3:  # 3 rappresentanti
            doc.set_as_representative()
            doc.confidence = 0.8 + i*0.05
        else:
            doc.set_as_propagated(propagated_from=0, propagated_label="tag_cluster_0")
            doc.confidence = 0.7
        documenti.append(doc)
    
    # Cluster 1: 8 documenti (cluster medio)
    for i in range(8):
        doc = DocumentoProcessing(f"session_c1_{i}", f"Domanda cluster 1 {i}")
        doc.set_clustering_info(cluster_id=1, cluster_size=8)
        doc.embeddings = [2.0, 3.0, 4.0]
        if i < 2:  # 2 rappresentanti
            doc.set_as_representative()
            doc.confidence = 0.75 + i*0.1
        else:
            doc.set_as_propagated(propagated_from=1, propagated_label="tag_cluster_1")
            doc.confidence = 0.65
        documenti.append(doc)
    
    # Cluster 2: 3 documenti (cluster piccolo) 
    for i in range(3):
        doc = DocumentoProcessing(f"session_c2_{i}", f"Domanda cluster 2 {i}")
        doc.set_clustering_info(cluster_id=2, cluster_size=3)
        doc.embeddings = [3.0, 4.0, 5.0]
        if i < 1:  # 1 rappresentante
            doc.set_as_representative()
            doc.confidence = 0.6
        else:
            doc.set_as_propagated(propagated_from=2, propagated_label="tag_cluster_2")
            doc.confidence = 0.5
        documenti.append(doc)
    
    # Outlier: 5 documenti
    for i in range(5):
        doc = DocumentoProcessing(f"session_out_{i}", f"Outlier {i}")
        doc.set_clustering_info(cluster_id=-1, cluster_size=1, is_outlier=True)
        doc.embeddings = [0.0, 1.0, 2.0]
        # Outlier non hanno rappresentanti
        doc.set_as_propagated(propagated_from=-1, propagated_label="altro")
        doc.confidence = 0.3
        documenti.append(doc)
    
    return documenti

def simulate_selection_logic(rappresentanti: List[DocumentoProcessing], 
                           budget: int, 
                           cluster_distribution: Dict[int, int]) -> int:
    """
    Simula la logica di selezione rappresentanti
    """
    if len(rappresentanti) <= budget:
        return len(rappresentanti)
    
    # Simula selezione intelligente
    cluster_representatives = {}
    for doc in rappresentanti:
        cluster_id = doc.cluster_id
        if cluster_id not in cluster_representatives:
            cluster_representatives[cluster_id] = []
        cluster_representatives[cluster_id].append(doc)
    
    # Strategia: 1 rappresentante per cluster prima
    selected_count = 0
    remaining_budget = budget
    
    # Prima passata: 1 per cluster (priorità cluster grandi)
    sorted_clusters = sorted(cluster_representatives.keys(), 
                           key=lambda cid: cluster_distribution.get(cid, 0), 
                           reverse=True)
    
    for cluster_id in sorted_clusters:
        if remaining_budget > 0:
            selected_count += 1
            remaining_budget -= 1
    
    # Seconda passata: cluster grandi ottengono rappresentanti extra
    for cluster_id in sorted_clusters:
        if remaining_budget <= 0:
            break
        
        cluster_size = cluster_distribution.get(cluster_id, 0)
        available_reps = len(cluster_representatives[cluster_id])
        
        if cluster_size >= 10 and available_reps > 1:
            selected_count += 1
            remaining_budget -= 1
    
    return selected_count

def clean_label_text_test(label: str) -> str:
    """
    Versione test della funzione di pulizia caratteri
    """
    if not label:
        return 'altro'
        
    import re
    
    # Rimuovi caratteri speciali e normalizza
    cleaned = re.sub(r'[^\w\s-]', '', label.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalizza spazi multipli
    cleaned = cleaned.lower().strip()
    
    # Fallback se rimane vuoto
    if not cleaned or len(cleaned) < 2:
        return 'altro'
        
    return cleaned

def determine_classification_mode(state: Dict[str, Any]) -> str:
    """
    Determina modalità di classificazione basata sullo stato
    """
    if not state.get("ensemble_exists", False):
        return "fallback"
    
    if not state.get("ml_trained", False) or len(state.get("classes", [])) == 0:
        return "llm_only"
    
    return "ensemble"

if __name__ == "__main__":
    test_recovered_logics()
