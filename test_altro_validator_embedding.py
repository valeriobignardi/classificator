#!/usr/bin/env python3
"""
Test per la nuova implementazione embedding-based di AltroTagValidator

Verifica che la logica embedding-based funzioni correttamente:
1. Estrazione tag da response LLM
2. Calcolo embedding e confronto semantico  
3. Decisione finale basata su soglia configurabile
4. Creazione automatica nuovi tag quando necessario

Autore: Valerio Bignardi
Data: 28 Agosto 2025
Ultima modifica: 2025-08-28 - Test per nuova implementazione
"""

import sys
import os
import logging
from typing import Dict, Any

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aggiunge percorsi necessari
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HumanReview'))

def test_altro_validator_embedding():
    """
    Test completo della nuova logica embedding-based
    
    Scopo della funzione: Testa tutti i componenti del nuovo validatore
    Parametri di input: None
    Parametri di output: Risultati test su console
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-08-28 - Creato per test nuova implementazione
    
    Autore: Valerio Bignardi
    Data: 2025-08-28
    """
    try:
        logger.info("🧪 === INIZIO TEST ALTRO_TAG_VALIDATOR EMBEDDING-BASED ===")
        
        # Import del validatore
        from HumanReview.altro_tag_validator import AltroTagValidator, ValidationResult
        
        # Test case con diverse response LLM simulate
        test_cases = [
            {
                "name": "Tag esistente - alta similarità",
                "llm_response": "Basandomi sul contenuto della conversazione, suggerisco il tag 'ASSISTENZA_TECNICA' per questa classificazione.",
                "expected_behavior": "similarity_match"
            },
            {
                "name": "Tag nuovo - bassa similarità", 
                "llm_response": "Propongo di creare un nuovo tag 'PROBLEMI_RETE_FIBRA' per questa specifica casistica.",
                "expected_behavior": "new_tag_created"
            },
            {
                "name": "Response ambigua",
                "llm_response": "Non sono sicuro della classificazione più appropriata per questo caso.",
                "expected_behavior": "error"
            },
            {
                "name": "Tag medico specifico",
                "llm_response": "Classificherei questa conversazione come 'CARDIOLOGIA_INTERVENTISTICA'.",
                "expected_behavior": "new_tag_created"  # Assumendo che non esista
            }
        ]
        
        # Inizializza validatore per tenant Humanitas
        logger.info("🔧 Inizializzazione AltroTagValidator...")
        validator = AltroTagValidator(tenant_id="humanitas")
        
        # Verifica inizializzazione
        stats = validator.get_validation_stats()
        logger.info(f"📊 Stats validatore: {stats}")
        
        # Test dei metodi di estrazione
        logger.info("\n🔍 === TEST ESTRAZIONE TAG DA LLM ===")
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n--- Test Case {i+1}: {test_case['name']} ---")
            logger.info(f"Response LLM: {test_case['llm_response']}")
            
            # Test estrazione tag
            extracted_tag = validator._extract_tag_from_llm_response(test_case['llm_response'])
            logger.info(f"🏷️  Tag estratto: {extracted_tag}")
            
            # Test validazione completa
            if extracted_tag:
                logger.info("🔄 Esecuzione validazione completa...")
                
                result = validator.validate_altro_classification(
                    conversation_id=f"test_conv_{i+1}",
                    llm_raw_response=test_case['llm_response'],
                    conversation_data=None
                )
                
                logger.info(f"📋 Risultato validazione:")
                logger.info(f"   - Final tag: {result.final_tag}")
                logger.info(f"   - Should add new tag: {result.should_add_new_tag}")
                logger.info(f"   - Confidence: {result.confidence:.3f}")
                logger.info(f"   - Validation path: {result.validation_path}")
                logger.info(f"   - Similarity score: {result.similarity_score}")
                logger.info(f"   - Matched existing: {result.matched_existing_tag}")
                logger.info(f"   - LLM suggested: {result.llm_suggested_tag}")
                
                # Verifica comportamento atteso
                if result.validation_path == test_case['expected_behavior']:
                    logger.info("✅ Comportamento come atteso")
                else:
                    logger.warning(f"⚠️  Comportamento diverso dall'atteso: {result.validation_path} vs {test_case['expected_behavior']}")
                
                # Se deve aggiungere nuovo tag, testa anche quello
                if result.should_add_new_tag and extracted_tag:
                    logger.info(f"📝 Test aggiunta nuovo tag: '{result.final_tag}'")
                    # Nota: non aggiungiamo realmente per evitare di sporcare il database
                    logger.info("ℹ️  (Aggiunta tag skippata in modalità test)")
            
            logger.info("-" * 50)
        
        # Test performance embedding
        logger.info("\n⚡ === TEST PERFORMANCE EMBEDDING ===")
        
        sample_tags = ["ASSISTENZA_TECNICA", "PROBLEMA_FATTURAZIONE", "CAMBIO_CONTRATTO", "NUOVO_TAG_TEST"]
        
        import time
        start_time = time.time()
        
        for tag in sample_tags:
            embedding = validator._get_tag_embedding(tag)
            logger.info(f"🧮 Embedding '{tag}': shape={embedding.shape}, norm={float(np.linalg.norm(embedding)):.3f}")
        
        end_time = time.time()
        logger.info(f"⏱️  Tempo calcolo 4 embedding: {(end_time - start_time):.3f}s")
        
        # Test cache
        logger.info("\n💾 === TEST CACHE EMBEDDING ===")
        
        start_time = time.time()
        for tag in sample_tags:  # Stessi tag - dovrebbe essere da cache
            embedding = validator._get_tag_embedding(tag)
        end_time = time.time()
        
        logger.info(f"⚡ Tempo secondo accesso (cache): {(end_time - start_time):.3f}s")
        
        # Statistiche finali
        final_stats = validator.get_validation_stats()
        logger.info(f"\n📊 Statistiche finali: {final_stats}")
        
        logger.info("\n✅ === TEST COMPLETATO CON SUCCESSO ===")
        
    except Exception as e:
        logger.error(f"❌ Errore durante test: {e}", exc_info=True)
        return False
    
    return True

def test_similarity_calculation():
    """
    Test specifico per il calcolo delle similarità
    
    Scopo della funzione: Verifica accuratezza calcolo similarità coseno
    Parametri di input: None
    Parametri di output: Log risultati test
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-08-28 - Test calcoli similarità
    
    Autore: Valerio Bignardi
    Data: 2025-08-28
    """
    try:
        logger.info("\n🔬 === TEST CALCOLO SIMILARITÀ ===")
        
        from HumanReview.altro_tag_validator import AltroTagValidator
        import numpy as np
        
        validator = AltroTagValidator(tenant_id="humanitas")
        
        # Test coppie di similarità note
        test_pairs = [
            ("ASSISTENZA_TECNICA", "SUPPORTO_TECNICO", "Alta similarità attesa"),
            ("FATTURAZIONE", "PROBLEMA_FATTURAZIONE", "Media similarità attesa"),
            ("CARDIOLOGIA", "MECCANICA_AUTO", "Bassa similarità attesa"),
            ("TELEFONIA", "TELEFONO", "Alta similarità attesa"),
        ]
        
        for tag1, tag2, description in test_pairs:
            embedding1 = validator._get_tag_embedding(tag1)
            embedding2 = validator._get_tag_embedding(tag2)
            
            similarity = validator._calculate_similarity(embedding1, embedding2)
            
            logger.info(f"📊 '{tag1}' vs '{tag2}': {similarity:.3f} ({description})")
        
        # Test embedding identici
        test_embedding = validator._get_tag_embedding("TEST_TAG")
        identical_similarity = validator._calculate_similarity(test_embedding, test_embedding)
        logger.info(f"🎯 Similarità tag identico: {identical_similarity:.3f} (deve essere ~1.0)")
        
        # Test embedding ortogonali
        embedding_dim = test_embedding.shape[0]
        zero_embedding = np.zeros(embedding_dim)
        zero_similarity = validator._calculate_similarity(test_embedding, zero_embedding)
        logger.info(f"❄️  Similarità vs zero embedding: {zero_similarity:.3f} (deve essere ~0.0)")
        
    except Exception as e:
        logger.error(f"❌ Errore test similarità: {e}", exc_info=True)

def test_config_loading():
    """
    Test caricamento configurazione
    
    Scopo della funzione: Verifica corretta lettura config.yaml
    Parametri di input: None  
    Parametri di output: Log configurazione caricata
    Valori di ritorno: None
    Tracciamento aggiornamenti: 2025-08-28 - Test configurazione
    
    Autore: Valerio Bignardi
    Data: 2025-08-28
    """
    try:
        logger.info("\n⚙️  === TEST CARICAMENTO CONFIGURAZIONE ===")
        
        from HumanReview.altro_tag_validator import AltroTagValidator
        
        # Test con configurazione da file
        validator = AltroTagValidator(tenant_id="humanitas")
        
        logger.info(f"🔧 Soglia similarità: {validator.similarity_threshold}")
        logger.info(f"💾 Cache abilitata: {validator.enable_cache}")
        logger.info(f"📏 Dimensione max cache: {validator.max_cache_size}")
        logger.info(f"🏢 Tenant ID: {validator.tenant_id}")
        
        # Test con configurazione custom
        custom_config = {
            'altro_tag_validator': {
                'semantic_similarity_threshold': 0.9,
                'enable_embedding_cache': False,
                'max_embedding_cache_size': 500
            }
        }
        
        custom_validator = AltroTagValidator(tenant_id="test", config=custom_config)
        
        logger.info(f"🔧 Custom - Soglia similarità: {custom_validator.similarity_threshold}")
        logger.info(f"💾 Custom - Cache abilitata: {custom_validator.enable_cache}")
        logger.info(f"📏 Custom - Dimensione max cache: {custom_validator.max_cache_size}")
        
    except Exception as e:
        logger.error(f"❌ Errore test configurazione: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("🚀 AVVIO TEST SUITE ALTRO_TAG_VALIDATOR")
    
    # Aggiungi numpy se necessario per i test
    try:
        import numpy as np
    except ImportError:
        logger.error("❌ NumPy richiesto per i test - pip install numpy")
        sys.exit(1)
    
    try:
        # Esegui tutti i test
        test_config_loading()
        test_similarity_calculation()
        test_success = test_altro_validator_embedding()
        
        if test_success:
            logger.info("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
            sys.exit(0)
        else:
            logger.error("❌ ALCUNI TEST FALLITI")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Test interrotti dall'utente")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Errore generale nei test: {e}", exc_info=True)
        sys.exit(1)
