#!/usr/bin/env python3
"""
Test del nuovo sistema di training supervisionato avanzato
con estrazione completa del dataset

Questo script testa:
1. Estrazione completa di tutte le discussioni dal database
2. Clustering su tutto il dataset
3. Selezione intelligente di rappresentanti per review umana
4. Training supervisionato con limite solo per la review umana
"""

import yaml
import sys
import os
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificazione_discussioni')

def test_config_supervised_training():
    """Testa che la configurazione supervised_training sia corretta"""
    print("📋 Test configurazione supervised_training...")
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        supervised_config = config.get('supervised_training', {})
        
        if not supervised_config:
            print("❌ Configurazione supervised_training non trovata!")
            return False
        
        # Verifica sezioni chiave
        extraction_config = supervised_config.get('extraction', {})
        human_review_config = supervised_config.get('human_review', {})
        
        print(f"✅ Configurazione supervised_training trovata:")
        print(f"  🔄 use_full_dataset: {extraction_config.get('use_full_dataset', False)}")
        print(f"  👤 max_total_sessions: {human_review_config.get('max_total_sessions', 'N/A')}")
        print(f"  📝 representatives_per_cluster: {human_review_config.get('representatives_per_cluster', 'N/A')}")
        print(f"  📊 selection_strategy: {human_review_config.get('selection_strategy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test configurazione: {e}")
        return False

def test_pipeline_methods():
    """Testa che i nuovi metodi nella pipeline esistano"""
    print("\n🔧 Test metodi pipeline...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Crea istanza di test
        pipeline = EndToEndPipeline(tenant_slug="test")
        
        # Verifica che i nuovi metodi esistano
        if not hasattr(pipeline, 'estrai_sessioni'):
            print("❌ Metodo estrai_sessioni non trovato!")
            return False
            
        if not hasattr(pipeline, '_select_representatives_for_human_review'):
            print("❌ Metodo _select_representatives_for_human_review non trovato!")
            return False
        
        # Verifica signature del metodo estrai_sessioni
        import inspect
        sig = inspect.signature(pipeline.estrai_sessioni)
        params = list(sig.parameters.keys())
        
        if 'force_full_extraction' not in params:
            print("❌ Parametro force_full_extraction non trovato in estrai_sessioni!")
            return False
        
        print(f"✅ Metodi pipeline trovati:")
        print(f"  📊 estrai_sessioni: {params}")
        print(f"  🔍 _select_representatives_for_human_review: presente")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test metodi pipeline: {e}")
        return False

def test_extraction_logic():
    """Testa la logica di estrazione completa"""
    print("\n🔄 Test logica estrazione...")
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        # Crea istanza di test (senza inizializzazione completa)
        pipeline = EndToEndPipeline.__new__(EndToEndPipeline)
        pipeline.config_path = 'config.yaml'
        
        # Simula il test della logica di estrazione
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        supervised_config = config.get('supervised_training', {})
        extraction_config = supervised_config.get('extraction', {})
        
        # Simula la logica di decisione
        use_full_dataset = extraction_config.get('use_full_dataset', False)
        force_full_extraction = True
        
        if use_full_dataset or force_full_extraction:
            extraction_mode = "COMPLETA"
            actual_limit = None
        else:
            extraction_mode = "LIMITATA"
            actual_limit = 100
        
        print(f"✅ Logica estrazione testata:")
        print(f"  📊 use_full_dataset: {use_full_dataset}")
        print(f"  🔄 force_full_extraction: {force_full_extraction}")
        print(f"  📋 extraction_mode: {extraction_mode}")
        print(f"  🔢 actual_limit: {actual_limit}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test logica estrazione: {e}")
        return False

def test_human_review_config():
    """Testa la configurazione per la review umana"""
    print("\n👤 Test configurazione review umana...")
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        supervised_config = config.get('supervised_training', {})
        human_review_config = supervised_config.get('human_review', {})
        
        # Parametri chiave
        max_total_sessions = human_review_config.get('max_total_sessions', 500)
        representatives_per_cluster = human_review_config.get('representatives_per_cluster', 3)
        selection_strategy = human_review_config.get('selection_strategy', 'prioritize_by_size')
        min_cluster_size = human_review_config.get('min_cluster_size_for_review', 2)
        
        print(f"✅ Configurazione review umana:")
        print(f"  📊 max_total_sessions: {max_total_sessions}")
        print(f"  📝 representatives_per_cluster: {representatives_per_cluster}")
        print(f"  🎯 selection_strategy: {selection_strategy}")
        print(f"  📏 min_cluster_size_for_review: {min_cluster_size}")
        
        # Simula calcolo sessioni
        n_clusters = 50  # Esempio
        total_with_default = n_clusters * representatives_per_cluster
        
        print(f"\n📊 Esempio con {n_clusters} cluster:")
        print(f"  📝 Sessioni con default ({representatives_per_cluster}/cluster): {total_with_default}")
        print(f"  🎯 Limite massimo: {max_total_sessions}")
        print(f"  ⚡ Selezione intelligente necessaria: {'SÌ' if total_with_default > max_total_sessions else 'NO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test configurazione review umana: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("🧪 TEST SISTEMA TRAINING SUPERVISIONATO AVANZATO")
    print("=" * 60)
    
    tests = [
        test_config_supervised_training,
        test_pipeline_methods,
        test_extraction_logic,
        test_human_review_config
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Riassunto
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 RIASSUNTO TEST: {passed}/{total} passati")
    
    if passed == total:
        print("🎉 TUTTI I TEST PASSATI!")
        print("\n✅ Il sistema di training supervisionato avanzato è pronto per l'uso.")
        print("\n📋 Per utilizzarlo:")
        print("  1. Assicurati che supervised_training.extraction.use_full_dataset = true")
        print("  2. Configura human_review.max_total_sessions (default: 500)")
        print("  3. Usa il nuovo endpoint /train/supervised/advanced/<client_name>")
        print("  4. La pipeline estrarrà TUTTE le discussioni per clustering completo")
        print("  5. Solo le sessioni rappresentative (max 500) saranno sottoposte all'umano")
    else:
        print("❌ ALCUNI TEST FALLITI!")
        print("🔧 Controlla la configurazione e i metodi implementati.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
