#!/usr/bin/env python3
"""
Pipeline veloce per test supervisione umana - estrae solo poche sessioni
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 PIPELINE VELOCE PER TEST SUPERVISIONE UMANA")
    print("="*70)
    
    try:
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        
        print("🔌 Inizializzazione pipeline...")
        
        # Inizializza pipeline
        pipeline = EndToEndPipeline(
            tenant_slug='humanitas'
            # auto_retrain ora viene letto da config.yaml
        )
        
        print("📊 Estrazione 50 sessioni per test veloce...")
        
        # Estrai solo 50 sessioni per test veloce
        sessioni = pipeline.estrai_sessioni(giorni_indietro=30, limit=50)
        print(f"✅ Estratte {len(sessioni)} sessioni")
        
        if len(sessioni) < 10:
            print("⚠️ Poche sessioni estratte, aumentando il range temporale...")
            sessioni = pipeline.estrai_sessioni(giorni_indietro=90, limit=50)
            print(f"✅ Estratte {len(sessioni)} sessioni (90 giorni)")
        
        if len(sessioni) == 0:
            print("❌ Nessuna sessione estratta")
            return False
        
        print("🔍 Clustering e analisi...")
        
        # Esegui clustering
        embeddings, cluster_labels, representatives, suggested_labels = pipeline.esegui_clustering(sessioni)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"✅ Trovati {n_clusters} cluster")
        
        if n_clusters == 0:
            print("⚠️ Nessun cluster trovato, uso classificazione diretta")
            
            # Classifica direttamente alcune sessioni
            print("🎓 Training classificatore con poche sessioni...")
            metrics = pipeline._allena_classificatore_fallback(sessioni)
            print(f"✅ Classificatore allenato: {metrics}")
        else:
            print(f"✅ Etichette generate per {len(suggested_labels)} cluster")
            
            print("🎓 Training classificatore ensemble...")
            
            # Allena il classificatore con i dati del clustering
            metrics = pipeline.allena_classificatore(
                sessioni=sessioni,
                cluster_labels=cluster_labels,
                representatives=representatives,
                suggested_labels=suggested_labels,
                interactive_mode=False  # Disabilita modalità interattiva per test veloce
            )
            
            print(f"✅ Training completato: {metrics}")
        
        print("🧪 Test del sistema ensemble con supervisione umana...")
        
        # Test alcune classificazioni
        test_texts = [
            "Vorrei prenotare una visita cardiologica urgente",
            "Ho problemi con l'accesso al portale online",
            "Quando posso ritirare i risultati delle analisi?"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📄 Test {i}: '{text}'")
            
            try:
                result = pipeline.ensemble_classifier.predict_with_ensemble(
                    text, 
                    return_details=True,
                    embedder=pipeline.embedder
                )
                
                print(f"   Risultato: {result['predicted_label']} (conf: {result['confidence']:.3f})")
                print(f"   Metodo: {result['method']}")
                
                if 'agreement' in result:
                    print(f"   Accordo LLM-ML: {'SÌ' if result['agreement'] else 'NO'}")
                
                if 'human_intervention' in result:
                    print(f"   Intervento umano: {'SÌ' if result['human_intervention'] else 'NO'}")
                    
            except Exception as e:
                print(f"   ❌ Errore: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ PIPELINE VELOCE COMPLETATA!")
        print(f"📊 Sistema pronto per test supervisione umana interattivi")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nella pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
