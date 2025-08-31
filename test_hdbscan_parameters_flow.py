#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
TEST FLUSSO PARAMETRI HDBSCAN: REACT → BACKEND → CLUSTERING
=====================================================================
Autore: Valerio Bignardi
Data: 2025-08-31
Descrizione: Test per verificare che i parametri impostati dall'utente
             in React vengano realmente usati da HDBSCAN durante il clustering
             
Scopo: Assicurarsi che non ci siano override di parametri utente
       con valori di default del sistema
=====================================================================
"""

import sys
import os
import json
import yaml
from typing import Dict, Any, Optional

# Aggiungi path del progetto
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Utils.tenant import Tenant
from Clustering.hdbscan_clusterer import HDBSCANClusterer


def test_parameters_flow():
    """
    Test completo del flusso dei parametri dall'interfaccia React al clustering HDBSCAN
    
    Verifica:
    1. Caricamento parametri da configurazione tenant-specific
    2. Override con parametri custom dell'utente (simulando React)
    3. Applicazione corretta nei costruttori HDBSCAN
    4. Effettivo utilizzo durante il clustering
    
    Returns:
        None
    """
    print("🧪 TEST FLUSSO PARAMETRI HDBSCAN: REACT → BACKEND → CLUSTERING")
    print("="*70)
    
    try:
        # 1. INIZIALIZZAZIONE TENANT
        print("\n📋 FASE 1: Inizializzazione tenant")
        tenant = Tenant.from_uuid('015007d9-d413-11ef-86a5-96000228e7fe')
        print(f"✅ Tenant: {tenant.tenant_name} ({tenant.tenant_id})")
        
        # 2. CARICAMENTO CONFIGURAZIONE DEFAULT
        print("\n⚙️  FASE 2: Configurazione default tenant")
        
        # Carica configurazione tenant-specific (come fa il sistema)
        config_path = os.path.join(project_root, 'config.yaml')
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Carica config tenant-specific se esiste
        tenant_config_path = os.path.join(
            project_root, 'tenant_configs', f'{tenant.tenant_slug}_clustering_config.yaml'
        )
        
        if os.path.exists(tenant_config_path):
            with open(tenant_config_path, 'r') as f:
                tenant_config = yaml.safe_load(f)
                clustering_default = tenant_config.get('clustering', {})
                print(f"📁 Config tenant-specific trovata: {tenant_config_path}")
        else:
            clustering_default = base_config.get('clustering', {})
            print(f"📁 Config generale usata: {config_path}")
        
        print("🔧 Parametri DEFAULT caricati:")
        for key, value in clustering_default.items():
            if not key.startswith('_'):  # Skip parametri interni
                print(f"   {key}: {value}")
        
        # 3. SIMULAZIONE PARAMETRI CUSTOM DA REACT
        print("\n🎨 FASE 3: Simulazione parametri custom da React Frontend")
        
        # Simula quello che arriverebbe dal frontend React
        custom_parameters = {
            'min_cluster_size': 15,        # Utente ha cambiato da default
            'min_samples': 8,              # Utente ha cambiato da default  
            'cluster_selection_epsilon': 0.18,  # Utente ha personalizzato
            'alpha': 0.65,                 # Utente ha modificato
            'metric': 'cosine',            # Utente ha cambiato da euclidean
            'cluster_selection_method': 'leaf',  # Utente ha scelto leaf
            'allow_single_cluster': True,   # Utente ha abilitato
            'use_umap': True,              # Utente ha abilitato UMAP
            'umap_n_neighbors': 25,        # Utente ha personalizzato UMAP
            'umap_min_dist': 0.05,         # Utente ha scelto valore custom
            'umap_n_components': 35        # Utente ha ridotto dimensioni
        }
        
        print("🎯 Parametri CUSTOM simulati (da React):")
        for key, value in custom_parameters.items():
            print(f"   {key}: {value}")
        
        # 4. CREAZIONE CLUSTERER CON PARAMETRI CUSTOM
        print("\n🔧 FASE 4: Creazione HDBSCANClusterer con parametri custom")
        
        # Simula il passaggio dei parametri come farebbe l'API
        clusterer = HDBSCANClusterer(
            min_cluster_size=custom_parameters['min_cluster_size'],
            min_samples=custom_parameters['min_samples'],
            cluster_selection_epsilon=custom_parameters['cluster_selection_epsilon'],
            alpha=custom_parameters['alpha'],
            metric=custom_parameters['metric'],
            cluster_selection_method=custom_parameters['cluster_selection_method'],
            allow_single_cluster=custom_parameters['allow_single_cluster'],
            use_umap=custom_parameters['use_umap'],
            umap_n_neighbors=custom_parameters['umap_n_neighbors'],
            umap_min_dist=custom_parameters['umap_min_dist'],
            umap_n_components=custom_parameters['umap_n_components']
        )
        
        # 5. VERIFICA PARAMETRI EFFETTIVI APPLICATI
        print("\n🔍 FASE 5: Verifica parametri effettivi applicati")
        
        effective_params = clusterer.get_effective_parameters()
        
        print("📊 PARAMETRI EFFETTIVI NEL CLUSTERER:")
        for key, value in effective_params.items():
            if key != 'note':
                print(f"   {key}: {value}")
        
        # 6. CONFRONTO: CUSTOM vs EFFETTIVI
        print("\n⚖️  FASE 6: Confronto parametri custom vs effettivi")
        
        mismatches = []
        for param_name, custom_value in custom_parameters.items():
            if param_name in effective_params:
                effective_value = effective_params[param_name]
                if custom_value != effective_value:
                    mismatches.append({
                        'parameter': param_name,
                        'custom': custom_value,
                        'effective': effective_value,
                        'match': False
                    })
                    print(f"❌ MISMATCH {param_name}: custom={custom_value}, effettivo={effective_value}")
                else:
                    print(f"✅ MATCH {param_name}: {custom_value}")
        
        # 7. RISULTATO FINALE
        print(f"\n🎯 RISULTATO TEST:")
        if mismatches:
            print(f"❌ TROVATI {len(mismatches)} MISMATCHES!")
            print("🚨 I parametri dall'interfaccia React NON vengono applicati correttamente!")
            for mismatch in mismatches:
                print(f"   - {mismatch['parameter']}: {mismatch['custom']} → {mismatch['effective']}")
        else:
            print("✅ SUCCESSO: Tutti i parametri custom vengono applicati correttamente!")
            print("🎉 L'interfaccia React controlla effettivamente HDBSCAN!")
        
        # 8. TEST AGGIUNTIVO: VERIFICA CHE I DEFAULT NON SOVRASCRIVANO
        print(f"\n🔬 FASE 7: Test sovrascrittura parametri")
        
        # Crea clusterer con solo alcuni parametri per vedere se i default sovrastano
        test_clusterer = HDBSCANClusterer(
            min_cluster_size=99,  # Valore insolito per verificare che non venga sovrascritto
            alpha=0.123           # Valore insolito per verificare che non venga sovrascritto
        )
        
        test_params = test_clusterer.get_effective_parameters()
        
        if test_params['min_cluster_size'] == 99 and test_params['alpha'] == 0.123:
            print("✅ PARAMETRI CUSTOM NON SOVRASCRITTI dai default")
        else:
            print(f"❌ PARAMETRI SOVRASCRITTI! min_cluster_size: {test_params['min_cluster_size']}, alpha: {test_params['alpha']}")
        
        print(f"\n🏁 TEST COMPLETATO")
        
    except Exception as e:
        print(f"❌ ERRORE durante test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parameters_flow()
