#!/usr/bin/env python3
"""
Script per applicare configurazioni clustering ottimizzate per Humanitas
Autore: Valerio Bignardi
Data: 2025-08-28
Obiettivo: Applicare le configurazioni ottimali derivate dall'analisi
"""

import json
import sys
import os
from datetime import datetime

def load_analysis_config():
    """
    Carica la configurazione di analisi dal file JSON
    """
    config_path = "/home/ubuntu/classificatore/humanitas_clustering_analysis.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Errore nel caricamento configurazione: {e}")
        return None

def display_configuration_options(config):
    """
    Mostra le opzioni di configurazione disponibili
    """
    print("🎯 CONFIGURAZIONI CLUSTERING OTTIMIZZATE DISPONIBILI")
    print("=" * 80)
    
    for i, opt_config in enumerate(config['optimized_configurations'], 1):
        print(f"\n{i}. {opt_config['name']}")
        print(f"   📋 {opt_config['description']}")
        print(f"   🎯 Obiettivi:")
        for key, value in opt_config['target_objectives'].items():
            print(f"      • {key}: {value}")
        print(f"   💡 Caso d'uso: {opt_config['use_case']}")
    
    print(f"\n4. TOP_ATTUALE")
    print(f"   📋 Usa la migliore configurazione già testata (Versione {config['top_configurations'][0]['version']})")
    print(f"   🎯 Quality Score: {config['top_configurations'][0]['quality_score']:.4f}")
    print(f"   📊 Silhouette: {config['top_configurations'][0]['silhouette_score']:.4f}")
    
    print(f"\n5. MOSTRA_DETTAGLI")
    print(f"   📋 Visualizza i dettagli completi di tutte le configurazioni")

def show_detailed_config(opt_config):
    """
    Mostra i dettagli completi di una configurazione
    """
    print(f"\n🔧 CONFIGURAZIONE: {opt_config['name']}")
    print("=" * 60)
    print(f"📋 Descrizione: {opt_config['description']}")
    print(f"💡 Caso d'uso: {opt_config['use_case']}")
    
    print(f"\n🎯 Obiettivi:")
    for key, value in opt_config['target_objectives'].items():
        print(f"   • {key}: {value}")
    
    print(f"\n⚙️  Parametri:")
    for key, value in opt_config['parameters'].items():
        print(f"   • {key}: {value}")

def generate_config_file(parameters, config_name):
    """
    Genera un file di configurazione YAML per il clustering
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clustering_config_{config_name.lower()}_{timestamp}.yaml"
    filepath = f"/home/ubuntu/classificatore/tenant_configs/{filename}"
    
    # Assicurati che la directory esista
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    yaml_content = f"""# Configurazione Clustering Ottimizzata - {config_name}
# Generata il: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Tenant: Humanitas (015007d9-d413-11ef-86a5-96000228e7fe)

clustering:
  # Parametri HDBSCAN
  alpha: {parameters['alpha']}
  metric: "{parameters['metric']}"
  min_samples: {parameters['min_samples']}
  min_cluster_size: {parameters['min_cluster_size']}
  max_cluster_size: {parameters['max_cluster_size']}
  cluster_selection_method: "{parameters['cluster_selection_method']}"
  cluster_selection_epsilon: {parameters['cluster_selection_epsilon']}
  allow_single_cluster: {str(parameters['allow_single_cluster']).lower()}
  
  # Parametri UMAP (se abilitato)
  use_umap: {str(parameters['use_umap']).lower()}
  umap_metric: "{parameters['umap_metric']}"
  umap_min_dist: {parameters['umap_min_dist']}
  umap_n_neighbors: {parameters['umap_n_neighbors']}
  umap_n_components: {parameters['umap_n_components']}
  umap_random_state: {parameters['umap_random_state']}
  
  # Filtri
  only_user: {str(parameters['only_user']).lower()}
  
# Metadati configurazione
metadata:
  config_name: "{config_name}"
  generated_at: "{datetime.now().isoformat()}"
  tenant_id: "015007d9-d413-11ef-86a5-96000228e7fe"
  tenant_name: "Humanitas"
  version: "optimized_v1.0"
"""
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"✅ Configurazione salvata: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Errore nel salvataggio: {e}")
        return None

def main():
    """
    Funzione principale per la selezione e applicazione della configurazione
    """
    print("🔬 APPLICAZIONE CONFIGURAZIONI CLUSTERING OTTIMIZZATE")
    print("=" * 80)
    print("Basato sull'analisi di 22 configurazioni testate per Humanitas")
    
    # Carica configurazione
    config = load_analysis_config()
    if not config:
        return
    
    # Mostra raccomandazione iniziale
    print(f"\n💡 RACCOMANDAZIONE INIZIALE:")
    print(f"   {config['analysis_summary']['recommendations']}")
    
    while True:
        print(f"\n" + "="*60)
        display_configuration_options(config)
        
        print(f"\n❓ Seleziona una configurazione (1-5) o 'q' per uscire:")
        choice = input("   Scelta: ").strip().lower()
        
        if choice == 'q':
            print("👋 Arrivederci!")
            break
        
        if choice == '5':
            print("\n📋 DETTAGLI CONFIGURAZIONI:")
            print("=" * 60)
            for opt_config in config['optimized_configurations']:
                show_detailed_config(opt_config)
            
            print(f"\n🏆 CONFIGURAZIONE TOP ATTUALE (Versione {config['top_configurations'][0]['version']}):")
            print("=" * 60)
            top_config = config['top_configurations'][0]
            print(f"📊 Metriche:")
            print(f"   • Quality Score: {top_config['quality_score']:.4f}")
            print(f"   • Silhouette Score: {top_config['silhouette_score']:.4f}")
            print(f"   • Cluster: {top_config['n_clusters']}")
            print(f"   • Outliers: {top_config['n_outliers']} ({top_config['outlier_ratio']:.1%})")
            
            print(f"⚙️  Parametri:")
            for key, value in top_config['parameters'].items():
                print(f"   • {key}: {value}")
            continue
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= 3:
                # Configurazione ottimizzata
                selected_config = config['optimized_configurations'][choice_num - 1]
                config_name = selected_config['name']
                parameters = selected_config['parameters']
                
            elif choice_num == 4:
                # Configurazione top attuale
                top_config = config['top_configurations'][0]
                config_name = f"TOP_ATTUALE_V{top_config['version']}"
                parameters = top_config['parameters']
                selected_config = {
                    'name': config_name,
                    'description': f"Migliore configurazione testata (Versione {top_config['version']})"
                }
                
            else:
                print("❌ Scelta non valida!")
                continue
            
            # Mostra dettagli della configurazione selezionata
            print(f"\n✅ CONFIGURAZIONE SELEZIONATA: {config_name}")
            print("=" * 60)
            print(f"📋 {selected_config['description']}")
            
            print(f"\n⚙️  Parametri che verranno applicati:")
            for key, value in parameters.items():
                print(f"   • {key}: {value}")
            
            # Conferma
            confirm = input(f"\n❓ Vuoi generare il file di configurazione? (s/n): ").strip().lower()
            
            if confirm == 's':
                config_path = generate_config_file(parameters, config_name)
                if config_path:
                    print(f"\n🎉 CONFIGURAZIONE PRONTA!")
                    print(f"📁 File: {config_path}")
                    print(f"\n📋 PROSSIMI PASSI:")
                    print(f"1. Copia il file nella directory appropriata del sistema")
                    print(f"2. Esegui il clustering con la nuova configurazione")
                    print(f"3. Monitora le metriche: silhouette_score, outlier_ratio, n_clusters")
                    print(f"4. Confronta i risultati con la configurazione precedente")
                    
                    # Mostra comando per test
                    print(f"\n🔧 COMANDO TEST SUGGERITO:")
                    print(f"python run_clustering.py --config {config_path} --tenant Humanitas")
                
            else:
                print("❌ Operazione annullata")
                
        except ValueError:
            print("❌ Inserisci un numero valido!")
        except Exception as e:
            print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()
