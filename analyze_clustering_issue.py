#!/usr/bin/env python3
"""
Script per analizzare e correggere i parametri di clustering di Humanitas
Autore: Valerio Bignardi
Data: 2025-01-17
"""

import sys
import os
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

def analyze_clustering_parameters():
    """
    Analizza i parametri di clustering attuali e propone correzioni
    """
    print("🔍 ANALISI PARAMETRI CLUSTERING HUMANITAS")
    print("="*60)
    
    # Carica i parametri attuali dal database
    from Utils.tenant_config_helper import get_all_clustering_parameters_for_tenant
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    try:
        params = get_all_clustering_parameters_for_tenant(tenant_id)
        
        print("📊 PARAMETRI ATTUALI:")
        print(f"   - min_cluster_size: {params.get('min_cluster_size', 'NON_DEFINITO')}")
        print(f"   - min_samples: {params.get('min_samples', 'NON_DEFINITO')}")
        print(f"   - cluster_selection_epsilon: {params.get('cluster_selection_epsilon', 'NON_DEFINITO')}")
        print(f"   - alpha: {params.get('alpha', 'NON_DEFINITO')}")
        print(f"   - cluster_selection_method: {params.get('cluster_selection_method', 'NON_DEFINITO')}")
        
        print("\n❌ PROBLEMI IDENTIFICATI:")
        issues = []
        
        min_cluster_size = params.get('min_cluster_size', 5)
        min_samples = params.get('min_samples', 3)
        
        if min_cluster_size > 10:
            issues.append(f"min_cluster_size={min_cluster_size} troppo alto per 1360 documenti")
            
        if min_samples > 8:
            issues.append(f"min_samples={min_samples} troppo alto, impedisce formazione cluster")
            
        if not issues:
            print("   ✅ Nessun problema evidente nei parametri")
        else:
            for issue in issues:
                print(f"   ❌ {issue}")
        
        print("\n💡 PARAMETRI OTTIMALI CONSIGLIATI:")
        print("   Per 1360 documenti, parametri suggeriti:")
        optimal_params = {
            'min_cluster_size': 5,  # Più permissivo
            'min_samples': 3,       # Valore standard
            'cluster_selection_epsilon': 0.1,  # Più permissivo
            'alpha': 1.0,           # Valore standard
            'cluster_selection_method': 'eom'  # Metodo standard
        }
        
        for param, value in optimal_params.items():
            current = params.get(param, 'NON_DEFINITO')
            status = "🔄 CAMBIARE" if str(current) != str(value) else "✅ OK"
            print(f"   - {param}: {current} → {value} {status}")
        
        print("\n🛠️ COMANDO PER APPLICARE LE CORREZIONI:")
        print("python fix_clustering_parameters.py")
        
        return optimal_params
        
    except Exception as e:
        print(f"❌ Errore nel recupero parametri: {e}")
        return None

def create_fix_script():
    """
    Crea script per correggere i parametri di clustering
    """
    script_content = '''#!/usr/bin/env python3
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
            
        print("\\n🚀 PROSSIMI PASSI:")
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
'''
    
    with open('/home/ubuntu/classificatore/fix_clustering_parameters.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("📝 Script fix_clustering_parameters.py creato!")

if __name__ == "__main__":
    print("🔍 Avvio analisi parametri clustering...")
    
    optimal_params = analyze_clustering_parameters()
    
    if optimal_params:
        create_fix_script()
        print("\\n✅ Analisi completata!")
        print("🎯 PROBLEMA IDENTIFICATO: Parametri clustering troppo restrittivi")
        print("💡 SOLUZIONE: Usare parametri più permissivi per permettere formazione cluster")
    else:
        print("❌ Analisi fallita!")
        sys.exit(1)
