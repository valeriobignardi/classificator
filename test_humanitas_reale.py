#!/usr/bin/env python3
"""
Script per testare _select_representatives_for_human_review con dati REALI di Humanitas
Autore: Valerio Bignardi
Data: 2025-01-17
"""

import sys
import os
import yaml
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from Utils.tenant import Tenant

def test_humanitas_real_data():
    """
    Test con dati reali di Humanitas per capire perché ritorna 0 rappresentanti
    """
    print("🔍 Test funzione con dati REALI di Humanitas")
    
    # Crea il file di log vuoto
    debug_log_path = "/home/ubuntu/classificatore/rappresentanti.log"
    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write(f"DEBUG LOG HUMANITAS REALE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    try:
        # Inizializza la pipeline
        print("⚙️ Inizializzazione pipeline...")
        pipeline = EndToEndPipeline()
        
        # Crea tenant di test con UUID reale di Humanitas
        tenant = Tenant(
            tenant_id="015007d9-d413-11ef-86a5-96000228e7fe",
            tenant_name="Humanitas", 
            tenant_slug="humanitas",
            tenant_database="humanitas_015007d9-d413-11ef-86a5-96000228e7fe",
            tenant_status=1
        )
        pipeline.tenant = tenant
        
        print("🔄 Caricamento dati reali dal MongoDB...")
        
        # Recupera i dati reali dal clustering
        from mongo_classification_reader import MongoClassificationReader
        
        reader = MongoClassificationReader(tenant)
        print("📊 Recupero tutti i documenti classificati...")
        
        # Connetti al database
        if not reader.connect():
            print("❌ Impossibile connettersi al MongoDB!")
            return False
        
        # Accedi alla collection
        collection = reader.db[reader.collection_name]
        all_docs = list(collection.find({}))
        print(f"📋 Documenti totali trovati: {len(all_docs)}")
        
        if not all_docs:
            print("❌ Nessun documento trovato nel MongoDB!")
            return False
        
        # Raggruppa per cluster_id
        cluster_groups = {}
        for doc in all_docs:
            cluster_id = doc.get('cluster_id', -1)
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(doc)
        
        print(f"📊 Analisi clustering reale:")
        print(f"  - Cluster totali: {len(cluster_groups)}")
        
        # Crea representatives dict dal clustering reale
        representatives = {}
        suggested_labels = {}
        
        for cluster_id, docs in cluster_groups.items():
            if cluster_id == -1:  # Outlier
                continue
                
            # Converte documenti in formato representatives
            cluster_reps = []
            for doc in docs:
                rep = {
                    'session_id': doc.get('session_id', 'no_session'),
                    'classification_confidence': doc.get('classification_confidence', 0.5),
                    'cluster_id': cluster_id
                }
                cluster_reps.append(rep)
            
            representatives[str(cluster_id)] = cluster_reps
            
            # Usa la prima etichetta trovata come suggested_label
            first_label = docs[0].get('suggested_label', f'cluster_{cluster_id}')
            suggested_labels[str(cluster_id)] = first_label
        
        print(f"📊 Dati estratti dal clustering reale:")
        print(f"  - Representatives: {len(representatives)} cluster")
        total_sessions = sum(len(reps) for reps in representatives.values())
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Suggested labels: {len(suggested_labels)}")
        
        # Mostra distribuzione dimensioni cluster
        cluster_sizes = {}
        for cluster_id, reps in representatives.items():
            size = len(reps)
            cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
        
        print(f"  - Distribuzione dimensioni cluster:")
        for size, count in sorted(cluster_sizes.items()):
            print(f"    • Dimensione {size}: {count} cluster")
        
        # Chiama la funzione con dati reali
        print("🚀 Chiamata _select_representatives_for_human_review con dati REALI...")
        
        result_reps, result_stats = pipeline._select_representatives_for_human_review(
            representatives=representatives,
            suggested_labels=suggested_labels,
            max_sessions=50,
            confidence_threshold=0.7,
            force_review=False,
            disagreement_threshold=0.3,
            all_sessions=None
        )
        
        print("✅ Funzione completata con dati reali!")
        print(f"📋 Risultati:")
        print(f"  - Cluster selezionati: {len(result_reps)}")
        print(f"  - Sessions selezionate: {result_stats.get('total_sessions_for_review', 0)}")
        print(f"  - Stats: {result_stats}")
        
        # Analizza il risultato
        if len(result_reps) == 0:
            print("❌ PROBLEMA CONFERMATO: La funzione ritorna 0 cluster!")
            print("🔍 Il debug nel log dovrebbe mostrare il motivo...")
        else:
            print("✅ La funzione ha selezionato cluster correttamente")
        
        # Mostra contenuto log
        print(f"\n📄 Debug log dettagliato scritto in: {debug_log_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        
        # Mostra log anche in caso di errore
        try:
            with open(debug_log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
                print("\n📄 Log parziale:")
                print("-" * 60)
                print(log_content[-1000:])  # Ultimi 1000 caratteri
                print("-" * 60)
        except:
            pass
        
        return False

if __name__ == "__main__":
    success = test_humanitas_real_data()
    if success:
        print("✅ Test con dati reali completato!")
    else:
        print("❌ Test con dati reali fallito!")
        sys.exit(1)
