#!/usr/bin/env python3
"""
Test debug per verificare problemi nel training BERTopic

Autore: Valerio Bignardi
Data: 2025-08-28
"""

import sys
import os
import yaml
import traceback
import numpy as np
from typing import Dict, Any

# Aggiungi path
sys.path.append('TopicModeling')
sys.path.append('EmbeddingEngine')
sys.path.append('MongoDB')

def test_bertopic_training():
    """
    Test completo del training BERTopic per identificare problemi
    
    Scopo della funzione: Debug training BERTopic in isolamento
    Parametri di input: None
    Parametri di output: None (stampe di debug)
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-08-28 - Creato per debug
    
    Autore: Valerio Bignardi
    Data: 2025-08-28
    """
    print("🔍 DEBUG BERTOPIC TRAINING")
    print("=" * 50)
    
    try:
        # 1. Test import
        print("📦 Test import...")
        from bertopic_feature_provider import BERTopicFeatureProvider
        print("✅ BERTopicFeatureProvider importato")
        
        # 2. Carica configurazione
        print("\n📋 Caricamento configurazione...")
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        bertopic_config = config.get('bertopic', {})
        print(f"✅ Config caricata: enabled={bertopic_config.get('enabled')}")
        print(f"   use_svd: {bertopic_config.get('use_svd')}")
        print(f"   svd_components: {bertopic_config.get('svd_components')}")
        
        # 3. Test dati mock
        print("\n🧪 Preparazione dati mock...")
        
        # Creo testi di test sufficienti per BERTopic
        test_texts = [
            "Vorrei prenotare una visita cardiologica urgente",
            "Ho problemi con l'app non riesco a vedere i referti",
            "Quando posso prenotare un esame del sangue",
            "Il dottore mi ha prescritto degli esami",
            "Non riesco ad accedere al portale online",
            "Vorrei disdire un appuntamento già prenotato",
            "Ho bisogno di una visita specialistica",
            "I risultati degli esami sono pronti",
            "Problema con la prenotazione online",
            "Vorrei cambiare la data dell'appuntamento",
            "Non funziona la app sul telefono",
            "Dove posso ritirare i referti medici",
            "Vorrei prenotare una visita dermatologica",
            "Ho dimenticato la password dell'account",
            "Quando esce il prossimo appuntamento disponibile",
            "Ho problemi con il pagamento online",
            "Vorrei parlare con un operatore",
            "Non riesco a fare il login",
            "Devo modificare i miei dati personali",
            "Vorrei prenotare una visita ginecologica",
            "L'app si blocca continuamente",
            "Ho bisogno di assistenza tecnica",
            "Vorrei prenotare un controllo generale",
            "Non riesco a scaricare i documenti",
            "Problema con la carta di credito",
            "Vorrei spostare un appuntamento",
            "Ho difficoltà con la registrazione",
            "Quando posso fare gli esami prescritti",
            "L'applicazione non funziona correttamente",
            "Vorrei annullare la prenotazione"
        ]
        
        print(f"✅ Creati {len(test_texts)} testi di test")
        
        # 4. Test embeddings mock
        print("\n🧠 Generazione embeddings mock...")
        # Creo embeddings mock (normalmente verrebbero da LaBSE)
        embedding_dim = 768
        embeddings = np.random.rand(len(test_texts), embedding_dim).astype(np.float32)
        # Normalizza gli embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        print(f"✅ Embeddings mock generati: shape {embeddings.shape}")
        
        # 5. Test inizializzazione BERTopic
        print("\n🚀 Test inizializzazione BERTopicFeatureProvider...")
        provider = BERTopicFeatureProvider(
            use_svd=bertopic_config.get('use_svd', False),
            svd_components=bertopic_config.get('svd_components', 32)
        )
        print("✅ BERTopicFeatureProvider inizializzato")
        
        # 6. Test fit
        print("\n🔥 Test training (fit)...")
        provider.fit(test_texts, embeddings=embeddings)
        print("✅ Fit completato con successo")
        
        # 7. Test transform
        print("\n🔄 Test transform...")
        result = provider.transform(
            test_texts,
            embeddings=embeddings,
            return_one_hot=bertopic_config.get('return_one_hot', False),
            top_k=bertopic_config.get('top_k', None)
        )
        print("✅ Transform completato con successo")
        
        # 8. Verifica risultati
        print("\n📊 Verifica risultati...")
        topic_probas = result.get('topic_probas')
        one_hot = result.get('one_hot')
        
        print(f"   Topic probabilities shape: {topic_probas.shape if topic_probas is not None else 'None'}")
        print(f"   One-hot shape: {one_hot.shape if one_hot is not None else 'None'}")
        
        # 9. Test modello BERTopic interno
        if hasattr(provider, 'model') and provider.model is not None:
            print("\n🤖 Test modello BERTopic interno...")
            bertopic_model = provider.model
            
            # Test transform del modello BERTopic
            test_single = ["Vorrei prenotare una visita medica"]
            topics, probs = bertopic_model.transform(test_single)
            print(f"✅ Transform modello interno: topic={topics[0]}, prob={probs[0]}")
            
            # Test get_topic_info
            if hasattr(bertopic_model, 'get_topic_info'):
                topic_info = bertopic_model.get_topic_info()
                print(f"✅ Topic info shape: {topic_info.shape if topic_info is not None else 'None'}")
                if topic_info is not None and not topic_info.empty:
                    print(f"   Primi 3 topic: {topic_info.head(3)['Name'].tolist()}")
            else:
                print("⚠️ get_topic_info non disponibile")
        else:
            print("❌ Modello BERTopic interno non disponibile")
        
        print("\n✅ TUTTI I TEST SUPERATI!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE durante test: {e}")
        print(f"🔍 Traceback completo:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bertopic_training()
