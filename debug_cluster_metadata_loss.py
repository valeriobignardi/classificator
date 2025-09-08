#!/usr/bin/env python3
"""
Script per debuggare il processo di salvataggio dei metadati clustering
e identificare dove si perdono i cluster_id e i flag rappresentanti.

Autore: Valerio Bignardi
Data: 2025-09-08
Ultimo aggiornamento: 2025-09-08 - Debug perdita metadati clustering
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from TAGS.tag import Tenant
import logging

# Setup logging dettagliato
logging.basicConfig(level=logging.DEBUG)

def debug_clustering_save_process():
    """
    Debug del processo di salvataggio per identificare dove si perdono 
    i metadati del clustering (cluster_id, is_representative)
    """
    
    print("🔍 DEBUG PROCESSO SALVATAGGIO CLUSTERING HUMANITAS")
    print("=" * 80)
    
    try:
        # Risolvi tenant
        tenant = Tenant.from_slug("humanitas")
        if not tenant:
            print("❌ Tenant Humanitas non trovato!")
            return
            
        print(f"✅ Tenant trovato: {tenant.tenant_name} ({tenant.tenant_id})")
        
        # Inizializza pipeline
        print(f"\n🚀 Inizializzazione pipeline...")
        pipeline = EndToEndPipeline(
            tenant=tenant,
            confidence_threshold=0.7,
            auto_mode=True
        )
        
        print(f"✅ Pipeline inizializzata per tenant: {tenant.tenant_name}")
        
        # Estrai sessioni (poche per debug)
        print(f"\n📥 Estrazione sessioni per debug...")
        sessioni = pipeline.estrai_sessioni(
            giorni_indietro=7, 
            limit=100,  # Solo 100 per debug veloce
            force_full_extraction=True
        )
        
        print(f"📊 Sessioni estratte: {len(sessioni)}")
        
        if len(sessioni) == 0:
            print("❌ Nessuna sessione trovata per il debug!")
            return
        
        # Esegui clustering
        print(f"\n🔍 Esecuzione clustering...")
        cluster_result = pipeline.esegui_clustering(
            num_sessioni=len(sessioni),
            force_reprocess=True
        )
        
        if not cluster_result:
            print("❌ Clustering fallito!")
            return
            
        cluster_labels, embeddings, session_texts, session_ids = cluster_result
        
        print(f"✅ Clustering completato:")
        print(f"   📊 Sessioni processate: {len(session_ids)}")
        print(f"   🎯 Cluster labels: {len(cluster_labels)}")
        print(f"   📈 Embeddings: {embeddings.shape if embeddings is not None else 'None'}")
        
        # Analizza risultati clustering
        unique_clusters = set(cluster_labels)
        outliers = sum(1 for label in cluster_labels if label == -1)
        clustered = len(cluster_labels) - outliers
        
        print(f"\n📊 ANALISI CLUSTERING:")
        print(f"   🎯 Cluster unici: {len(unique_clusters)}")
        print(f"   ✅ Documenti clusterizzati: {clustered}")
        print(f"   🚫 Outlier: {outliers}")
        
        # Mostra distribuzione cluster
        from collections import Counter
        cluster_counts = Counter(cluster_labels)
        print(f"\n🏆 TOP 10 CLUSTER:")
        for cluster_id, count in cluster_counts.most_common(10):
            if cluster_id == -1:
                print(f"   Outlier (ID=-1): {count} documenti")
            else:
                print(f"   Cluster {cluster_id}: {count} documenti")
        
        # Seleziona rappresentanti
        print(f"\n👤 Selezione rappresentanti...")
        
        # Qui dovremmo intercettare la selezione rappresentanti
        # Modifichiamo temporaneamente la pipeline per catturare i risultati
        
        # Prima verifichiamo se la pipeline ha i metadati corretti
        if hasattr(pipeline, 'cluster_labels') and pipeline.cluster_labels is not None:
            print(f"✅ Pipeline ha cluster_labels: {len(pipeline.cluster_labels)}")
            print(f"   Outlier in pipeline: {sum(1 for x in pipeline.cluster_labels if x == -1)}")
        else:
            print(f"❌ Pipeline non ha cluster_labels!")
            
        if hasattr(pipeline, 'session_ids') and pipeline.session_ids is not None:
            print(f"✅ Pipeline ha session_ids: {len(pipeline.session_ids)}")
        else:
            print(f"❌ Pipeline non ha session_ids!")
        
        # Chiamiamo direttamente la funzione di selezione rappresentanti per debug
        print(f"\n🎯 DEBUG: Chiamata diretta _select_representatives_for_human_review")
        
        # Impostiamo manualmente i dati nella pipeline per il test
        pipeline.cluster_labels = cluster_labels
        pipeline.session_ids = session_ids
        pipeline.session_texts = session_texts
        
        # Chiamiamo la funzione con debug
        rappresentanti = pipeline._select_representatives_for_human_review(
            num_representatives=min(50, clustered),  # Max 50 per debug
            num_suggested_labels=min(50, clustered),
            max_sessions=100,
            confidence_threshold=0.95,
            force_review=True,
            disagreement_threshold=0.0
        )
        
        print(f"📊 RISULTATO SELEZIONE RAPPRESENTANTI:")
        print(f"   🎯 Rappresentanti selezionati: {len(rappresentanti) if rappresentanti else 0}")
        
        if rappresentanti:
            # Analizza i rappresentanti
            rep_clusters = set()
            for rep in rappresentanti:
                if hasattr(rep, 'cluster_id'):
                    rep_clusters.add(rep.cluster_id)
                elif isinstance(rep, dict) and 'cluster_id' in rep:
                    rep_clusters.add(rep['cluster_id'])
            
            print(f"   🎯 Cluster con rappresentanti: {len(rep_clusters)}")
            print(f"   📋 Cluster IDs: {sorted(rep_clusters)}")
            
            # Mostra alcuni esempi
            print(f"\n📄 PRIMI 5 RAPPRESENTANTI:")
            for i, rep in enumerate(rappresentanti[:5]):
                if isinstance(rep, dict):
                    cluster_id = rep.get('cluster_id', 'N/A')
                    session_id = rep.get('session_id', 'N/A')
                    print(f"   {i+1}. Session {session_id}, Cluster {cluster_id}")
                else:
                    print(f"   {i+1}. Tipo rappresentante: {type(rep)}")
        
        # Ora testiamo il salvataggio
        print(f"\n💾 TEST SALVATAGGIO...")
        
        # Qui dovremmo interceptare il salvataggio per vedere cosa succede
        print(f"⚠️  ATTENZIONE: Il test di salvataggio modificherebbe il database!")
        print(f"   Per sicurezza, salto il test di salvataggio effettivo.")
        print(f"   Controllare manualmente la funzione salva_classificazioni_puro")
        
    except Exception as e:
        print(f"❌ Errore durante il debug: {e}")
        import traceback
        traceback.print_exc()

def analyze_save_function():
    """Analizza la funzione di salvataggio per identificare il bug"""
    
    print(f"\n🔍 ANALISI FUNZIONE SALVATAGGIO")
    print("=" * 50)
    
    # Cerchiamo la funzione salva_classificazioni_puro
    from Pipeline.end_to_end_pipeline import EndToEndPipeline
    
    # Verifichiamo se la funzione esiste
    if hasattr(EndToEndPipeline, 'salva_classificazioni_puro'):
        print(f"✅ Funzione salva_classificazioni_puro trovata")
        
        # Otteniamo il codice sorgente per analisi
        import inspect
        try:
            source = inspect.getsource(EndToEndPipeline.salva_classificazioni_puro)
            print(f"📄 CODICE FUNZIONE salva_classificazioni_puro:")
            print("-" * 60)
            # Mostriamo solo le prime 50 righe per non sovraccaricare
            lines = source.split('\n')[:50]
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}: {line}")
            if len(source.split('\n')) > 50:
                print(f"... (funzione continua per altre {len(source.split('\n')) - 50} righe)")
        except Exception as e:
            print(f"❌ Impossibile ottenere codice sorgente: {e}")
    else:
        print(f"❌ Funzione salva_classificazioni_puro non trovata!")
    
    # Cerchiamo anche altre funzioni di salvataggio
    save_functions = [attr for attr in dir(EndToEndPipeline) 
                     if 'salva' in attr.lower() or 'save' in attr.lower()]
    
    print(f"\n🔍 FUNZIONI DI SALVATAGGIO DISPONIBILI:")
    for func_name in save_functions:
        print(f"   - {func_name}")

if __name__ == "__main__":
    print("🐛 DEBUG PERDITA METADATI CLUSTERING - HUMANITAS")
    print("=" * 70)
    
    debug_clustering_save_process()
    analyze_save_function()
    
    print(f"\n✅ DEBUG COMPLETATO")
    print(f"💡 Controlla i risultati sopra per identificare dove si perdono i metadati!")
