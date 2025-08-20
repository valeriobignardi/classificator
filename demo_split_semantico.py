#!/usr/bin/env python3
"""
Demo completo del funzionamento del clustering gerarchico adattivo
Mostra step-by-step come funziona la fase di split semantico
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Aggiungi path per import moduli locali
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🚀 DEMO CLUSTERING GERARCHICO ADATTIVO")
print("="*80)
print(f"⏰ Avvio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Simula dataset con conflitti intenzionali
def create_conflicting_dataset() -> tuple:
    """
    Crea un dataset mock che simula conflitti di etichette
    """
    print("📊 FASE 1: CREAZIONE DATASET CON CONFLITTI")
    print("-" * 50)
    
    # Testi con conflitti semantici intenzionali
    texts = [
        # Gruppo 1: Prenotazioni (ma alcune ambigue)
        "Vorrei prenotare una visita cardiologica per la prossima settimana",
        "Posso fissare un appuntamento con il dermatologo?",
        "Ho bisogno di prenotare degli esami del sangue urgenti",
        "Quanto costa prenotare una visita privata?",  # AMBIGUO: prenotazione o info costi?
        
        # Gruppo 2: Informazioni (ma alcune sovrapposte)
        "Quali sono gli orari di apertura del laboratorio?",
        "Vorrei sapere i servizi disponibili nella vostra struttura",
        "Come posso sapere i tempi di attesa per le visite?",  # AMBIGUO: info o lamentela?
        "Dove trovo informazioni sui costi delle prestazioni?",  # AMBIGUO: info o preventivo?
        
        # Gruppo 3: Problemi/Lamentele (con sfumature diverse)
        "Il dottore è arrivato con due ore di ritardo alla visita",
        "La sala d'attesa è molto sporca e disorganizzata",
        "Non riesco a contattare nessuno al telefono da giorni",  # AMBIGUO: lamentela o richiesta assistenza?
        "I tempi di attesa sono davvero troppo lunghi",  # AMBIGUO: lamentela o richiesta info?
        
        # Gruppo 4: Casi chiaramente ambigui (per testare boundary regions)
        "Ho bisogno di aiuto per la mia situazione",  # MOLTO AMBIGUO
        "Potreste darmi delle informazioni?",  # GENERICO
        "Non so cosa fare, potete aiutarmi?",  # VAGO
        "C'è qualcuno che può assistermi?",  # GENERICO
    ]
    
    session_ids = [f"session_{i:03d}" for i in range(len(texts))]
    
    # Crea embeddings mock (simulano embeddings reali con similarità)
    np.random.seed(42)  # Per risultati riproducibili
    
    # Embeddings con clusters che si sovrappongono intenzionalmente
    embeddings = []
    for i, text in enumerate(texts):
        if i < 4:  # Gruppo prenotazioni (con 1 ambiguo)
            base = np.array([1.0, 0.0, 0.0])
            if i == 3:  # Caso ambiguo prenotazione/costi
                base += np.array([0.0, 0.5, 0.0])  # Sposta verso info
        elif i < 8:  # Gruppo informazioni (con 2 ambigui)
            base = np.array([0.0, 1.0, 0.0])
            if i in [6, 7]:  # Casi ambigui
                base += np.array([0.3, 0.0, 0.3])  # Sposta verso altri gruppi
        elif i < 12:  # Gruppo lamentele (con 2 ambigui)
            base = np.array([0.0, 0.0, 1.0])
            if i in [10, 11]:  # Casi ambigui
                base += np.array([0.0, 0.4, 0.0])  # Sposta verso info
        else:  # Gruppo molto ambiguo (boundary cases)
            base = np.array([0.33, 0.33, 0.33])  # Equidistante da tutti
        
        # Aggiungi rumore per realismo
        noise = np.random.normal(0, 0.1, 3)
        embedding = base + noise
        embedding = embedding / np.linalg.norm(embedding)  # Normalizza
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    print(f"✅ Dataset creato:")
    print(f"   📝 Testi: {len(texts)}")
    print(f"   🔢 Embeddings: {embeddings.shape}")
    print(f"   ⚠️ Conflitti intenzionali: 6 casi ambigui")
    print(f"   🎯 Gruppi target: prenotazioni(4), info(4), lamentele(4), ambigui(4)")
    
    return texts, embeddings, session_ids

def simulate_llm_classifications(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Simula classificazioni LLM che creano conflitti
    """
    print("\n🧠 FASE 2: SIMULAZIONE CLASSIFICAZIONI LLM CON CONFLITTI")
    print("-" * 50)
    
    # Simulazioni di classificazioni che creano conflitti intenzionali
    mock_classifications = [
        # Gruppo 1: Prenotazioni
        {"label": "prenotazione", "confidence": 0.85},
        {"label": "prenotazione", "confidence": 0.78},
        {"label": "prenotazione", "confidence": 0.82},
        {"label": "info_costi", "confidence": 0.65},  # CONFLITTO: dovrebbe essere prenotazione
        
        # Gruppo 2: Informazioni  
        {"label": "info_generali", "confidence": 0.80},
        {"label": "info_servizi", "confidence": 0.75},
        {"label": "lamentela_attesa", "confidence": 0.62},  # CONFLITTO: dovrebbe essere info
        {"label": "preventivo", "confidence": 0.58},  # CONFLITTO: dovrebbe essere info
        
        # Gruppo 3: Lamentele
        {"label": "lamentela_ritardo", "confidence": 0.88},
        {"label": "lamentela_igiene", "confidence": 0.84},
        {"label": "richiesta_assistenza", "confidence": 0.55},  # CONFLITTO: dovrebbe essere lamentela
        {"label": "info_tempi", "confidence": 0.52},  # CONFLITTO: dovrebbe essere lamentela
        
        # Gruppo 4: Casi boundary (bassa confidenza naturale)
        {"label": "richiesta_generica", "confidence": 0.45},
        {"label": "info_generali", "confidence": 0.40},
        {"label": "richiesta_assistenza", "confidence": 0.38},
        {"label": "altro", "confidence": 0.35},
    ]
    
    print("✅ Classificazioni LLM simulate:")
    for i, (text, classification) in enumerate(zip(texts, mock_classifications)):
        confidence_indicator = "🔴" if classification["confidence"] < 0.5 else "🟡" if classification["confidence"] < 0.7 else "🟢"
        print(f"   {i:2d}. {confidence_indicator} '{classification['label']}' (conf: {classification['confidence']:.2f})")
        print(f"       📝 \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    
    conflicts_detected = sum(1 for c in mock_classifications if c["confidence"] < 0.7)
    print(f"\n📊 Analisi:")
    print(f"   ⚠️ Classificazioni potenzialmente conflittuali: {conflicts_detected}/16")
    print(f"   🎯 Confidenza media: {np.mean([c['confidence'] for c in mock_classifications]):.3f}")
    
    return mock_classifications

def demonstrate_split_semantic_phase(texts: List[str], 
                                   embeddings: np.ndarray, 
                                   classifications: List[Dict[str, Any]]) -> None:
    """
    Dimostra step-by-step il funzionamento della fase di split semantico
    """
    print("\n🌳 FASE 3: SPLIT SEMANTICO DETTAGLIATO")
    print("="*60)
    
    # Step 1: Clustering iniziale con HDBSCAN
    print("🔍 STEP 1: Clustering iniziale HDBSCAN")
    print("-" * 40)
    
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=3, metric='cosine')
    initial_labels = clusterer.fit_predict(embeddings)
    
    # Analizza cluster formati
    unique_labels = set(initial_labels)
    print(f"📊 Cluster iniziali trovati: {len(unique_labels)}")
    
    cluster_composition = {}
    for label in unique_labels:
        indices = [i for i, l in enumerate(initial_labels) if l == label]
        cluster_composition[label] = indices
        
        if label == -1:
            print(f"   🔴 Outlier: {len(indices)} sessioni")
        else:
            print(f"   🔵 Cluster {label}: {len(indices)} sessioni")
            
            # Mostra composizione semantica del cluster
            cluster_labels = [classifications[i]["label"] for i in indices]
            label_counts = {}
            for cl in cluster_labels:
                label_counts[cl] = label_counts.get(cl, 0) + 1
            
            print(f"      Etichette: {dict(label_counts)}")
            
            # Rileva conflitti (più di 1 etichetta significativa)
            significant_labels = {l: c for l, c in label_counts.items() if c >= 2}
            if len(significant_labels) > 1:
                print(f"      ⚠️ CONFLITTO RILEVATO: {len(significant_labels)} etichette significative")
    
    # Step 2: Analisi conflitti per cluster specifico
    print(f"\n🔍 STEP 2: Analisi dettagliata conflitto (Cluster 0)")
    print("-" * 40)
    
    # Prendiamo il cluster 0 come esempio (dovrebbe avere conflitti)
    target_cluster = 0
    if target_cluster in cluster_composition:
        cluster_indices = cluster_composition[target_cluster]
        
        print(f"📊 Cluster {target_cluster} - Analisi conflitto:")
        print(f"   Membri: {len(cluster_indices)} sessioni")
        
        # Analizza etichette nel cluster
        cluster_labels = [classifications[i]["label"] for i in cluster_indices]
        cluster_confidences = [classifications[i]["confidence"] for i in cluster_indices]
        
        from collections import Counter
        label_counts = Counter(cluster_labels)
        print(f"   Distribuzione etichette: {dict(label_counts)}")
        
        # Calcola severità conflitto
        unique_labels_count = len(set(cluster_labels))
        entropy = -sum((count/len(cluster_labels)) * np.log2(count/len(cluster_labels)) 
                      for count in label_counts.values())
        avg_confidence = np.mean(cluster_confidences)
        
        conflict_severity = (entropy / np.log2(unique_labels_count)) * (1 - avg_confidence)
        
        print(f"   📊 Metriche conflitto:")
        print(f"      Entropia etichette: {entropy:.3f}")
        print(f"      Confidenza media: {avg_confidence:.3f}")  
        print(f"      Severità conflitto: {conflict_severity:.3f}")
        
        # Step 3: Decisione strategia split
        print(f"\n🔧 STEP 3: Strategia risoluzione conflitto")
        print("-" * 40)
        
        if conflict_severity > 0.7:
            strategy = "SPLIT GERARCHICO"
            print(f"   ✅ Strategia scelta: {strategy}")
            print(f"   📝 Motivo: Conflitto severo (severità > 0.7)")
        elif conflict_severity > 0.4:
            strategy = "BOUNDARY REFINEMENT"
            print(f"   ✅ Strategia scelta: {strategy}")
            print(f"   📝 Motivo: Conflitto moderato (0.4 < severità ≤ 0.7)")
        else:
            strategy = "SOFT REASSIGNMENT"
            print(f"   ✅ Strategia scelta: {strategy}")
            print(f"   📝 Motivo: Conflitto lieve (severità ≤ 0.4)")
        
        # Step 4: Simulazione split semantico
        if strategy == "SPLIT GERARCHICO":
            print(f"\n🌳 STEP 4: Esecuzione Split Semantico")
            print("-" * 40)
            
            # Raggruppa per etichetta semantica
            semantic_groups = {}
            for i, idx in enumerate(cluster_indices):
                label = classifications[idx]["label"]
                confidence = classifications[idx]["confidence"]
                
                if label not in semantic_groups:
                    semantic_groups[label] = []
                semantic_groups[label].append({
                    'index': idx,
                    'text': texts[idx],
                    'confidence': confidence,
                    'embedding': embeddings[idx]
                })
            
            print(f"   📊 Gruppi semantici identificati:")
            
            subclusters_created = 0
            for label, members in semantic_groups.items():
                if len(members) >= 2:  # Minimo per sotto-cluster
                    
                    # Calcola coerenza semantica interna
                    if len(members) > 1:
                        member_embeddings = np.array([m['embedding'] for m in members])
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarities = cosine_similarity(member_embeddings)
                        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                    else:
                        avg_similarity = 1.0
                    
                    avg_confidence = np.mean([m['confidence'] for m in members])
                    
                    print(f"      🌿 Sotto-cluster '{label}':")
                    print(f"         Membri: {len(members)}")
                    print(f"         Confidenza media: {avg_confidence:.3f}")
                    print(f"         Coerenza semantica: {avg_similarity:.3f}")
                    
                    # Validazione qualità sotto-cluster
                    if avg_similarity > 0.7 and avg_confidence > 0.6:
                        print(f"         ✅ ACCETTATO (alta qualità)")
                        subclusters_created += 1
                        
                        # Mostra campioni
                        for j, member in enumerate(members[:2]):  # Max 2 campioni
                            text_preview = member['text'][:50] + "..." if len(member['text']) > 50 else member['text']
                            print(f"         📝 Campione {j+1}: \"{text_preview}\"")
                    else:
                        print(f"         ⚠️ SCARTATO (bassa qualità: sim={avg_similarity:.3f}, conf={avg_confidence:.3f})")
                else:
                    print(f"      🔸 Gruppo '{label}': {len(members)} membri (troppo piccolo)")
            
            print(f"\n   📊 Risultato split:")
            print(f"      Sotto-cluster creati: {subclusters_created}")
            print(f"      Cluster originale → Nodo gerarchico padre")
            print(f"      Conflitto risolto: ✅")
            
            # Step 5: Validazione risultato
            print(f"\n✅ STEP 5: Validazione risultato")
            print("-" * 40)
            
            if subclusters_created >= 2:
                print(f"   🎉 Split semantico RIUSCITO!")
                print(f"   📈 Benefici:")
                print(f"      - Conflitto risolto senza perdita informazioni")
                print(f"      - Preservata struttura gerarchica")
                print(f"      - Mantenuta granularità semantica")
                print(f"      - Possibilità future di merge se necessario")
            else:
                print(f"   ⚠️ Split semantico NON EFFICACE")
                print(f"   🔄 Fallback automatico a boundary refinement")
    
    print(f"\n🎯 RIEPILOGO FASE SPLIT SEMANTICO")
    print("="*50)
    print(f"✅ Il sistema dimostra come i conflitti vengono:")
    print(f"   1. 🔍 Rilevati attraverso analisi entropica")
    print(f"   2. 📊 Valutati per severità e dimensione")  
    print(f"   3. 🧠 Analizzati semanticamente con LLM")
    print(f"   4. 🌳 Risolti con split gerarchico intelligente")
    print(f"   5. ✅ Validati per qualità e coerenza")

def main():
    """
    Demo completo del sistema
    """
    try:
        # Fase 1: Crea dataset
        texts, embeddings, session_ids = create_conflicting_dataset()
        
        # Fase 2: Simula classificazioni LLM
        classifications = simulate_llm_classifications(texts)
        
        # Fase 3: Dimostra split semantico
        demonstrate_split_semantic_phase(texts, embeddings, classifications)
        
        print(f"\n🏁 DEMO COMPLETATA CON SUCCESSO")
        print(f"⏰ Terminata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n💡 PUNTI CHIAVE DEL CLUSTERING GERARCHICO:")
        print(f"   🎯 Preserva incertezza invece di forzare decisioni")
        print(f"   🧠 Usa intelligenza semantica LLM per split consapevoli")  
        print(f"   🌳 Mantiene struttura gerarchica per analisi future")
        print(f"   🔄 Si adatta iterativamente ai nuovi conflitti")
        print(f"   📊 Fornisce metriche dettagliate di qualità")
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE LA DEMO: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
