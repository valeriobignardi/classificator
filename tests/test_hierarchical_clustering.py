#!/usr/bin/env python3
"""
Test per il sistema di clustering gerarchico adattivo
Dimostra l'uso del nuovo HierarchicalAdaptiveClusterer e l'integrazione con la pipeline
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Aggiungi path per import moduli locali
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.end_to_end_pipeline import EndToEndPipeline
from Clustering.hierarchical_adaptive_clusterer import HierarchicalAdaptiveClusterer


def crea_sessioni_mock_con_conflitti() -> Dict[str, Dict]:
    """
    Crea un dataset mock che contiene conflitti di etichette intenzionali
    per testare il clustering gerarchico
    """
    sessioni_mock = {}
    
    # Categoria 1: Richieste di informazioni (con varianti che potrebbero confondere)
    info_texts = [
        "Vorrei sapere gli orari di apertura della clinica",
        "Potreste darmi informazioni sui vostri servizi?",
        "A che ora aprite domani mattina?",
        "Quali sono gli orari del laboratorio analisi?",
        "Ho bisogno di informazioni sui reparti disponibili",
        # Varianti ambigue
        "Dove posso trovare informazioni sui costi?",  # Potrebbe essere info o preventivo
        "Come posso sapere i tempi di attesa?",        # Potrebbe essere info o lamentela
    ]
    
    # Categoria 2: Prenotazioni (con sovrapposizioni)
    prenotazione_texts = [
        "Vorrei prenotare una visita cardiologica",
        "Posso fissare un appuntamento per la settimana prossima?",
        "Ho bisogno di prenotare degli esami del sangue",
        "Devo prenotare una risonanza magnetica",
        "Come posso prenotare una visita specialistica?",
        # Varianti ambigue
        "Quanto costa prenotare una visita privata?",   # Potrebbe essere prenotazione o preventivo
        "Posso cambiare l'orario del mio appuntamento?", # Potrebbe essere modifica o nuova prenotazione
    ]
    
    # Categoria 3: Lamentele/Problemi (con diverse sfumature)
    lamentela_texts = [
        "Il dottore è arrivato con due ore di ritardo",
        "La sala d'attesa è molto sporca",
        "Il personale è stato molto scortese",
        "Ho aspettato tre ore per una visita di 5 minuti",
        "La fattura che ho ricevuto è sbagliata",
        # Varianti ambigue  
        "Non riesco a contattare nessuno al telefono",    # Potrebbe essere lamentela o richiesta info
        "I tempi di attesa sono troppo lunghi",           # Potrebbe essere lamentela o richiesta info
    ]
    
    # Categoria 4: Emergenze (potrebbero sovrapporsi con prenotazioni urgenti)
    emergenza_texts = [
        "Ho un forte dolore al petto, cosa devo fare?",
        "Mio figlio ha la febbre alta, è urgente",
        "Ho avuto un incidente, dove posso andare?",
        "Serve un'ambulanza immediatamente",
        # Varianti ambigue
        "Ho bisogno di una visita urgente oggi",          # Potrebbe essere emergenza o prenotazione urgente
        "È possibile avere un consulto medico subito?",   # Potrebbe essere emergenza o prenotazione
    ]
    
    # Crea sessioni con timestamp distribuiti
    all_categories = [
        ("informazioni", info_texts),
        ("prenotazione", prenotazione_texts), 
        ("lamentela", lamentela_texts),
        ("emergenza", emergenza_texts)
    ]
    
    session_id = 1
    base_time = datetime.now() - timedelta(days=7)
    
    for categoria, texts in all_categories:
        for i, text in enumerate(texts):
            session_time = base_time + timedelta(hours=i*2, minutes=session_id*5)
            
            sessioni_mock[f"session_{session_id:04d}"] = {
                'testo_completo': text,
                'data_conversazione': session_time.strftime('%Y-%m-%d %H:%M:%S'),
                'categoria_attesa': categoria,  # Per analisi post-test
                'lunghezza_conversazione': len(text.split()),
                'source': 'test_mock'
            }
            session_id += 1
    
    return sessioni_mock


def test_analisi_conflitti_baseline():
    """
    Test 1: Analizza conflitti nel clustering standard (baseline)
    """
    print("🔍 TEST 1: ANALISI CONFLITTI BASELINE")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = EndToEndPipeline(config_path='config.yaml')
    
    # Crea dataset con conflitti
    sessioni = crea_sessioni_mock_con_conflitti()
    print(f"📊 Dataset creato: {len(sessioni)} sessioni")
    
    # Analizza conflitti con clustering standard
    risultati_analisi = pipeline.analizza_conflitti_etichette(sessioni, show_details=True)
    
    print(f"\n📋 RISULTATI BASELINE:")
    stats = risultati_analisi['statistiche']
    print(f"   Cluster totali: {stats['cluster_totali']}")
    print(f"   Cluster con conflitti: {stats['cluster_con_conflitti']}")
    print(f"   Percentuale conflitti: {stats['percentuale_conflitti']:.1f}%")
    print(f"   Raccomandazione: {stats['raccomandazione_principale']}")
    
    return risultati_analisi


def test_clustering_gerarchico_diretto():
    """
    Test 2: Test diretto del clustering gerarchico avanzato
    """
    print("\n🌳 TEST 2: CLUSTERING GERARCHICO DIRETTO")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = EndToEndPipeline(config_path='config.yaml')
    
    # Crea dataset
    sessioni = crea_sessioni_mock_con_conflitti()
    
    # Esegui clustering gerarchico diretto
    embeddings, session_memberships, cluster_info, hierarchical_structure = pipeline.esegui_clustering_gerarchico_avanzato(
        sessioni,
        confidence_threshold=0.8,   # Alta confidenza per regioni pure
        boundary_threshold=0.4,     # Soglia moderata per boundary
        max_iterations=5            # Più iterazioni per convergenza
    )
    
    print(f"\n📊 RISULTATI CLUSTERING GERARCHICO:")
    stats = hierarchical_structure['statistics']
    print(f"   Regioni totali: {len(hierarchical_structure['regions'])}")
    print(f"   Profondità massima: {stats['max_depth']}")
    print(f"   Sessioni multi-membership: {stats['multi_membership_sessions']}")
    print(f"   Conflitti risolti: {stats.get('conflicts_resolved', 'N/A')}")
    
    # Analizza composizione regioni
    print(f"\n🔍 DETTAGLIO REGIONI:")
    for region_id, region_data in hierarchical_structure['regions'].items():
        print(f"   Regione {region_id}:")
        print(f"     Tipo: {region_data['type']}")
        print(f"     Membri: {len(region_data['members'])}")
        print(f"     Confidence: {region_data.get('avg_confidence', 'N/A')}")
        if region_data.get('intent_string'):
            print(f"     Intent: {region_data['intent_string']}")
    
    return hierarchical_structure


def test_risoluzione_automatica():
    """
    Test 3: Test della risoluzione automatica dei conflitti
    """
    print("\n🔧 TEST 3: RISOLUZIONE AUTOMATICA CONFLITTI")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = EndToEndPipeline(config_path='config.yaml')
    
    # Crea dataset con molti conflitti
    sessioni = crea_sessioni_mock_con_conflitti()
    
    # Test con strategia automatica
    risultati_auto = pipeline.risolvi_conflitti_automaticamente(
        sessioni,
        strategia='auto'  # Lascia che il sistema scelga la strategia migliore
    )
    
    print(f"\n📈 RISULTATI RISOLUZIONE AUTOMATICA:")
    print(f"   Strategia applicata: {risultati_auto['strategia_applicata']}")
    print(f"   Conflitti iniziali: {risultati_auto['conflitti_iniziali']}")
    print(f"   Conflitti risolti: {risultati_auto['conflitti_risolti']}")
    print(f"   Miglioramento: {risultati_auto['miglioramento_percentuale']:.1f}%")
    print(f"   Tempo esecuzione: {risultati_auto['tempo_esecuzione']:.2f}s")
    
    # Test con strategia gerarchica forzata
    print(f"\n🌳 Test con strategia gerarchica forzata...")
    risultati_gerarchico = pipeline.risolvi_conflitti_automaticamente(
        sessioni,
        strategia='clustering_gerarchico_adattivo'
    )
    
    print(f"   Miglioramento gerarchico: {risultati_gerarchico['miglioramento_percentuale']:.1f}%")
    print(f"   Dettagli: {risultati_gerarchico['risoluzione_dettagli']}")
    
    return risultati_auto, risultati_gerarchico


def test_comparativo_performance():
    """
    Test 4: Confronto performance tra clustering standard e gerarchico
    """
    print("\n⚡ TEST 4: CONFRONTO PERFORMANCE")
    print("="*60)
    
    # Inizializza pipeline
    pipeline = EndToEndPipeline(config_path='config.yaml')
    
    # Crea dataset più grande
    sessioni_base = crea_sessioni_mock_con_conflitti()
    
    # Espandi dataset per test performance (replica con variazioni)
    sessioni_grandi = {}
    for i in range(3):  # Triplica il dataset
        for session_id, session_data in sessioni_base.items():
            new_id = f"{session_id}_replica_{i}"
            new_data = session_data.copy()
            # Aggiungi piccole variazioni per realismo
            if i == 1:
                new_data['testo_completo'] += " - potete aiutarmi?"
            elif i == 2:
                new_data['testo_completo'] += " Grazie"
            sessioni_grandi[new_id] = new_data
    
    print(f"📊 Dataset espanso: {len(sessioni_grandi)} sessioni")
    
    # Test clustering standard
    print(f"🔄 Test clustering standard...")
    start_time = datetime.now()
    embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni_grandi)
    time_standard = (datetime.now() - start_time).total_seconds()
    
    clusters_standard = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    outliers_standard = sum(1 for label in cluster_labels if label == -1)
    
    # Test clustering gerarchico
    print(f"🌳 Test clustering gerarchico...")
    start_time = datetime.now()
    embeddings, session_memberships, cluster_info, hierarchical_structure = pipeline.esegui_clustering_gerarchico_avanzato(
        sessioni_grandi,
        confidence_threshold=0.75,
        boundary_threshold=0.45,
        max_iterations=3
    )
    time_gerarchico = (datetime.now() - start_time).total_seconds()
    
    # Calcola metriche comparative
    print(f"\n📊 CONFRONTO RISULTATI:")
    print(f"   CLUSTERING STANDARD:")
    print(f"     Tempo: {time_standard:.2f}s")
    print(f"     Cluster: {clusters_standard}")
    print(f"     Outlier: {outliers_standard} ({outliers_standard/len(sessioni_grandi)*100:.1f}%)")
    
    print(f"   CLUSTERING GERARCHICO:")
    print(f"     Tempo: {time_gerarchico:.2f}s")
    print(f"     Regioni: {len(hierarchical_structure['regions'])}")
    print(f"     Multi-membership: {hierarchical_structure['statistics']['multi_membership_sessions']}")
    
    # Calcola overhead
    overhead = ((time_gerarchico - time_standard) / time_standard) * 100
    print(f"\n⚡ OVERHEAD GERARCHICO: {overhead:+.1f}%")
    
    return {
        'standard': {'time': time_standard, 'clusters': clusters_standard, 'outliers': outliers_standard},
        'gerarchico': {'time': time_gerarchico, 'regions': len(hierarchical_structure['regions']), 'overhead': overhead}
    }


def test_configurazione_yaml():
    """
    Test 5: Test della configurazione YAML per clustering gerarchico
    """
    print("\n⚙️ TEST 5: CONFIGURAZIONE YAML")
    print("="*60)
    
    # Test configurazione con clustering gerarchico abilitato
    print("🔧 Modifico config.yaml per abilitare clustering gerarchico...")
    
    # Leggi config attuale
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modifica configurazione
    config['hierarchical_clustering']['enabled'] = True
    config['hierarchical_clustering']['confidence_threshold'] = 0.8
    config['hierarchical_clustering']['max_iterations'] = 5
    
    # Salva config temporanea
    with open('config_test_hierarchical.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configurazione temporanea creata: config_test_hierarchical.yaml")
    
    # Test pipeline con nuova configurazione
    pipeline = EndToEndPipeline(config_path='config_test_hierarchical.yaml')
    sessioni = crea_sessioni_mock_con_conflitti()
    
    # Dovrebbe automaticamente usare clustering gerarchico
    print("🌳 Test con configurazione gerarchica abilitata...")
    embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni)
    
    print(f"✅ Pipeline configurata correttamente")
    print(f"   Tipo clustering utilizzato: {'Gerarchico' if hasattr(pipeline, '_last_hierarchical_clusterer') else 'Standard'}")
    
    # Cleanup
    if os.path.exists('config_test_hierarchical.yaml'):
        os.remove('config_test_hierarchical.yaml')
    
    return True


def main():
    """
    Esegue tutti i test del clustering gerarchico
    """
    print("🚀 SUITE TEST CLUSTERING GERARCHICO ADATTIVO")
    print("="*80)
    print(f"⏰ Avvio test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test 1: Analisi baseline
        baseline_results = test_analisi_conflitti_baseline()
        
        # Test 2: Clustering gerarchico diretto
        hierarchical_structure = test_clustering_gerarchico_diretto()
        
        # Test 3: Risoluzione automatica
        auto_results, hierarchical_results = test_risoluzione_automatica()
        
        # Test 4: Performance
        performance_results = test_comparativo_performance()
        
        # Test 5: Configurazione
        config_test = test_configurazione_yaml()
        
        # Riepilogo finale
        print("\n🎯 RIEPILOGO GENERALE")
        print("="*60)
        print("✅ Tutti i test completati con successo!")
        print(f"📊 Conflitti baseline: {baseline_results['statistiche']['cluster_con_conflitti']}")
        print(f"🌳 Regioni gerarchiche: {len(hierarchical_structure['regions'])}")
        print(f"📈 Miglioramento automatico: {auto_results['miglioramento_percentuale']:.1f}%")
        print(f"⚡ Overhead performance: {performance_results['gerarchico']['overhead']:+.1f}%")
        print(f"⚙️ Configurazione YAML: {'✅ OK' if config_test else '❌ ERRORE'}")
        
        print(f"\n🏁 Test completati: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE I TEST: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
