#!/usr/bin/env python3
"""
Script per pipeline completa da zero con supervisione umana
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Pipeline.end_to_end_pipeline import EndToEndPipeline

def main():
    """Esecuzione pipeline completa"""
    
    print("🚀 PIPELINE COMPLETA DA ZERO - HUMANITAS")
    print("="*70)
    
    # Inizializza pipeline
    print("🔌 Inizializzazione pipeline...")
    pipeline = EndToEndPipeline(
        tenant_slug='humanitas'
        # auto_retrain ora viene letto da config.yaml
    )
    
    # Step 1: Estrazione dati
    print("\n📊 STEP 1: Estrazione dati freschi...")
    try:
        sessioni = pipeline.estrai_sessioni(limit=200, giorni_indietro=30)
        print(f"✅ Estratte {len(sessioni)} sessioni fresche")
        
        if not sessioni:
            print("❌ Nessuna sessione estratta - impossibile continuare")
            return False
            
        # Mostra preview delle prime sessioni
        print(f"\n📈 Preview prime 3 sessioni:")
        for i, (sid, data) in enumerate(list(sessioni.items())[:3]):
            testo_preview = data['testo_completo'][:100] + "..." if len(data['testo_completo']) > 100 else data['testo_completo']
            print(f"  {i+1}. {sid}: {testo_preview}")
            
    except Exception as e:
        print(f"❌ Errore nell'estrazione: {e}")
        return False
    
    # Step 2: Clustering
    print(f"\n🎯 STEP 2: Clustering intelligente...")
    try:
        embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        outliers = sum(1 for label in cluster_labels if label == -1)
        
        print(f"✅ Clustering completato:")
        print(f"   📊 Cluster trovati: {n_clusters}")
        print(f"   🔍 Outlier: {outliers}")
        print(f"   🏷️  Etichette suggerite: {len(suggested_labels)}")
        print(f"   📈 Rappresentanti totali: {sum(len(reps) for reps in representatives.values())}")
        
    except Exception as e:
        print(f"❌ Errore nel clustering: {e}")
        return False
    
    # Step 3: Verifica etichette suggerite dal clustering intelligente
    print(f"\n🧠 STEP 3: Verifica etichette suggerite...")
    try:
        print(f"✅ Etichette suggerite generate dal clustering intelligente:")
        print(f"   🏷️  Cluster con etichette: {len(suggested_labels)}")
        
        # Mostra le etichette suggerite
        print(f"   📋 Etichette suggerite:")
        for cluster_id, label in list(suggested_labels.items())[:5]:
            print(f"      Cluster {cluster_id}: {label}")
        if len(suggested_labels) > 5:
            print(f"      ... e altre {len(suggested_labels) - 5}")
            
    except Exception as e:
        print(f"❌ Errore nella gestione etichette: {e}")
        return False
    
    # Step 4: Training ensemble ML
    print(f"\n🎓 STEP 4: Training ensemble ML...")
    try:
        # Training senza review interattiva per ora
        metrics = pipeline.allena_classificatore(
            sessioni=sessioni,
            cluster_labels=cluster_labels,
            representatives=representatives,
            suggested_labels=suggested_labels,
            interactive_mode=False  # Disabilitiamo review interattiva per ora
        )
        
        print(f"✅ Training ML completato:")
        print(f"   📊 Training accuracy: {metrics.get('training_accuracy', 'N/A'):.3f}")
        print(f"   🔢 Samples: {metrics.get('n_samples', 'N/A')}")
        print(f"   🏷️  Classes: {metrics.get('n_classes', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Errore nel training ML: {e}")
        return False
    
    # Step 5: Test classificazione ensemble
    print(f"\n🧪 STEP 5: Test classificazione ensemble...")
    try:
        # Test con alcuni esempi
        test_texts = [
            "Vorrei prenotare una visita cardiologica",
            "Ho problemi con l'accesso al portale",
            "Quando posso ritirare i risultati delle analisi?"
        ]
        
        print(f"   Test con {len(test_texts)} esempi...")
        
        results = []
        for text in test_texts:
            result = pipeline.ensemble_classifier.predict_with_ensemble(
                text, return_details=True
            )
            results.append(result)
            print(f"   '{text[:50]}...' → {result['predicted_label']} ({result['confidence']:.3f})")
        
        print(f"✅ Test ensemble completato")
        
    except Exception as e:
        print(f"❌ Errore nel test ensemble: {e}")
        return False
    
    print(f"\n🎉 PIPELINE COMPLETATA CON SUCCESSO!")
    print(f"   ✅ Sistema pronto per supervisione umana")
    print(f"   ✅ Modello ML allenato e funzionante")
    print(f"   ✅ Ensemble LLM+ML operativo")
    
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESSO' if success else '❌ ERRORE'}")
    sys.exit(0 if success else 1)
