#!/usr/bin/env python3
"""
Debug Rappresentanti Vuoti - Identificazione Root Cause
Autore: Valerio Bignardi  
Data: 2025-09-09

Scopo: Tracciare perché i rappresentanti sono vuoti nel log tracing.log
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from Models.tenant import Tenant
from Utils.trace import trace_all

def debug_rappresentanti_step_by_step():
    """Debug passo passo della selezione rappresentanti"""
    
    print("🔍 DEBUG RAPPRESENTANTI VUOTI")
    print("=" * 60)
    
    # Simula stesso tenant del log
    tenant = Tenant.get_by_slug("16c222a9-f293-11ef-9315-96000228e7fe")
    
    # Crea pipeline
    pipeline = EndToEndPipeline(
        tenant=tenant,
        confidence_threshold=0.7,
        auto_mode=True
    )
    
    print(f"📊 Pipeline creata per tenant: {tenant.tenant_id}")
    
    # 1. ESTRAZIONE SESSIONI (simuliamo stesso contesto)
    print(f"\n1️⃣ ESTRAZIONE SESSIONI...")
    try:
        sessioni = pipeline.estrai_sessioni(giorni_indietro=7, limit=None, force_full_extraction=True)
        print(f"   ✅ Estratte {len(sessioni)} sessioni")
    except Exception as e:
        print(f"   ❌ Errore estrazione: {e}")
        return
    
    # 2. CLUSTERING 
    print(f"\n2️⃣ CLUSTERING...")
    try:
        documenti = pipeline.esegui_clustering(sessioni, force_reprocess=False)
        print(f"   ✅ Clustering completato: {len(documenti)} documenti")
        
        # Conta rappresentanti SUBITO dopo clustering
        rappresentanti_post_clustering = [doc for doc in documenti if doc.is_representative]
        print(f"   🎯 Rappresentanti POST-CLUSTERING: {len(rappresentanti_post_clustering)}")
        
        # Debug cluster info
        cluster_counts = {}
        rappresentanti_per_cluster = {}
        
        for doc in documenti:
            cluster_id = doc.cluster_id
            if cluster_id is not None and cluster_id != -1:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                
                if doc.is_representative:
                    if cluster_id not in rappresentanti_per_cluster:
                        rappresentanti_per_cluster[cluster_id] = 0
                    rappresentanti_per_cluster[cluster_id] += 1
        
        print(f"   📊 Cluster totali: {len(cluster_counts)}")
        print(f"   📊 Cluster con rappresentanti: {len(rappresentanti_per_cluster)}")
        
        if len(rappresentanti_per_cluster) > 0:
            print(f"   🎯 Dettaglio rappresentanti per cluster:")
            for cluster_id in sorted(rappresentanti_per_cluster.keys())[:5]:  # primi 5
                total = cluster_counts.get(cluster_id, 0)
                reps = rappresentanti_per_cluster[cluster_id]
                print(f"      Cluster {cluster_id}: {reps} rappresentanti su {total} totali")
        
    except Exception as e:
        print(f"   ❌ Errore clustering: {e}")
        return
    
    # 3. SELEZIONE RAPPRESENTANTI DIRETTA
    print(f"\n3️⃣ SELEZIONE RAPPRESENTANTI...")
    try:
        rappresentanti_selezionati = pipeline.select_representatives_from_documents(
            documenti=documenti,
            max_sessions=300
        )
        
        print(f"   🎯 RISULTATO: {len(rappresentanti_selezionati)} rappresentanti selezionati")
        
        if len(rappresentanti_selezionati) == 0:
            print(f"   ❌ PROBLEMA CONFERMATO: Nessun rappresentante selezionato!")
            
            # Debug più dettagliato
            print(f"\n🔍 ANALISI DETTAGLIATA:")
            print(f"   📊 Documenti totali: {len(documenti)}")
            
            rappresentanti_diretti = [doc for doc in documenti if doc.is_representative]
            print(f"   👥 Documenti con is_representative=True: {len(rappresentanti_diretti)}")
            
            if len(rappresentanti_diretti) == 0:
                print(f"   ❌ ROOT CAUSE: Nessun documento marcato come rappresentante!")
                print(f"      🔍 Verifica se set_as_representative() viene chiamato...")
                
                # Cerca documenti con selection_reason
                docs_con_reason = [doc for doc in documenti if hasattr(doc, 'selection_reason') and doc.selection_reason]
                print(f"   📋 Documenti con selection_reason: {len(docs_con_reason)}")
                
            else:
                print(f"   ✅ Rappresentanti trovati, problema in select_representatives_from_documents()")
        else:
            print(f"   ✅ Rappresentanti selezionati correttamente!")
            
    except Exception as e:
        print(f"   ❌ Errore selezione: {e}")
        return

if __name__ == "__main__":
    debug_rappresentanti_step_by_step()
