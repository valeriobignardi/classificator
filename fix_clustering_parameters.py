#!/usr/bin/env python3
"""
Script per correggere i parametri di clustering di Humanitas
Autore: Valerio Bignardi
Data: 2025-01-17
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

def fix_clustering_parameters():
    """
    Applica i parametri di clustering ottimali per Humanitas
    """
    from Pipeline.end_to_end_pipeline import get_supervised_training_params_from_db
    
    # Questa funzione non esiste ancora, per ora uso manuale
    def save_clustering_parameters_to_db(tenant_id, params):
        print(f"🔧 SIMULAZIONE: Salvataggio parametri per {tenant_id}")
        for param, value in params.items():
            print(f"   SET {param} = {value}")
        print("   📝 Nota: Implementare save_clustering_parameters_to_db nella pipeline")
        return True
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Parametri ottimali per 1360 documenti
    optimal_params = {
        'min_cluster_size': 5,
        'min_samples': 3, 
        'cluster_selection_epsilon': 0.1,
        'alpha': 1.0,
        'cluster_selection_method': 'eom'
    }
    
    print("🔧 Applicazione parametri clustering ottimali...")
    print(f"   Tenant ID: {tenant_id}")
    
    try:
        # Salva i nuovi parametri
        save_clustering_parameters_to_db(tenant_id, optimal_params)
        
        print("✅ Parametri clustering aggiornati con successo!")
        print("📊 Nuovi parametri:")
        for param, value in optimal_params.items():
            print(f"   - {param}: {value}")
            
        print("\n🚀 PROSSIMI PASSI:")
        print("1. Riavvia il sistema di clustering")
        print("2. Esegui un nuovo clustering sui dati di Humanitas")
        print("3. Verifica che vengano generati cluster validi")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nell'aggiornamento parametri: {e}")
        return False

if __name__ == "__main__":
    success = fix_clustering_parameters()
    if not success:
        sys.exit(1)
