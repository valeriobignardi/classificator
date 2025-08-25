#!/usr/bin/env python3
"""
Test per verificare che BERTopic funzioni con dataset piccoli
"""

import sys
sys.path.append('/home/ubuntu/classificazione_discussioni')

from TopicModeling.bertopic_feature_provider import BERTopicFeatureProvider
import numpy as np

print("🧪 TEST: BERTopic con dataset piccoli")

# Crea dataset di test piccolo (tipico caso problematico)
test_texts = [
    "Vorrei prenotare una visita cardiologica",
    "Come posso accedere al portale paziente",  
    "Dove ritiro i referti degli esami",
    "Informazioni sui costi delle prestazioni",
    "Indicazioni stradali per l'ospedale",
    "Orari di apertura del centro prelievi",
    "Prenotazione TAC urgente",
    "Problema con la ricetta elettronica",
    "Info parcheggio per pazienti",
    "Contatti reparto cardiologia"
]

print(f"📊 Dataset test: {len(test_texts)} campioni")

try:
    # Test 1: Inizializzazione BERTopic
    print(f"\n📋 Test 1: Inizializzazione BERTopic")
    bertopic_provider = BERTopicFeatureProvider(
        random_state=42,
        calculate_probabilities=True,  # Questo causava l'errore
        n_neighbors=5,  # Ridotto per dataset piccolo
        hdbscan_min_cluster_size=2,  # Ridotto per dataset piccolo
        hdbscan_min_samples=1,  # Ridotto per dataset piccolo
        metric="euclidean"  # Usa metrica robusta
    )
    
    if not bertopic_provider.is_available():
        print("❌ BERTopic non disponibile")
        exit(1)
    
    print("✅ BERTopic provider inizializzato")
    
    # Test 2: Creazione embeddings fake (per test veloce)
    print(f"\n📋 Test 2: Generazione embeddings di test")
    embeddings = np.random.rand(len(test_texts), 768)  # Simulazione embeddings
    print(f"✅ Embeddings generati: shape {embeddings.shape}")
    
    # Test 3: Training BERTopic (il punto che falliva)  
    print(f"\n📋 Test 3: Training BERTopic su dataset piccolo")
    bertopic_provider.fit(test_texts, embeddings=embeddings)
    print("✅ BERTopic fit completato senza errori!")
    
    # Test 4: Transform per verificare funzionalità
    print(f"\n📋 Test 4: Transform e feature extraction")
    features = bertopic_provider.transform(
        test_texts[:3], 
        embeddings=embeddings[:3],
        top_k=3,
        return_one_hot=True
    )
    
    print(f"✅ Transform completato")
    print(f"   topic_probas shape: {features.get('topic_probas', np.array([])).shape}")
    print(f"   one_hot shape: {features.get('one_hot', np.array([])).shape}")
    print(f"   topic_ids: {features.get('topic_ids', [])}")
    
    print(f"\n🎉 SUCCESSO: BERTopic funziona correttamente con dataset piccoli!")
    
except Exception as e:
    print(f"❌ ERRORE: {e}")
    import traceback
    traceback.print_exc()
    
print(f"\n🏁 Test completato!")
