"""
Test integrazione servizio embedding dockerizzato
Autore: Valerio Bignardi  
Data: 2025-08-29

Scopo: Test completo funzionamento client remoto e servizio embedding,
       verifica compatibilità con codice esistente
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any

# Aggiungi path per importare moduli
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/EmbeddingEngine')

def test_remote_client():
    """Test base client remoto LaBSE"""
    print("🧪 TEST: Client Remoto LaBSE")
    print("=" * 50)
    
    try:
        from EmbeddingEngine.labse_remote_client import LaBSERemoteClient
        
        # Test inizializzazione
        print("📦 Inizializzazione client...")
        client = LaBSERemoteClient(
            service_url="http://localhost:8080",
            timeout=60,
            max_retries=2,
            fallback_local=True
        )
        
        # Test connessione
        print("🔌 Test connessione...")
        service_info = client.get_service_info()
        if 'error' not in service_info:
            print(f"✅ Servizio info: {service_info}")
        else:
            print(f"⚠️ Servizio info: {service_info['error']}")
        
        # Test embedding singolo
        print("📝 Test embedding singolo...")
        start_time = time.time()
        single_text = "Questo è un test di embedding"
        embeddings = client.encode(single_text)
        elapsed = time.time() - start_time
        
        print(f"✅ Embedding shape: {embeddings.shape}")
        print(f"⏱️ Tempo: {elapsed:.3f}s")
        
        # Test embedding multipli
        print("📝 Test embedding multipli...")
        start_time = time.time()
        texts = [
            "Primo testo di test",
            "Secondo testo per verifica",
            "Terzo messaggio di prova", 
            "Quarto esempio di embedding"
        ]
        embeddings = client.encode(texts)
        elapsed = time.time() - start_time
        
        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"⏱️ Tempo: {elapsed:.3f}s")
        print(f"📊 Velocità: {len(texts)/elapsed:.1f} testi/sec")
        
        # Test normalizzazione
        print("🔢 Test normalizzazione...")
        embeddings_norm = client.encode(texts, normalize_embeddings=True)
        norms = np.linalg.norm(embeddings_norm, axis=1)
        print(f"✅ Norme vettori normalizzati: {norms}")
        
        # Statistiche client
        print("📈 Statistiche client:")
        stats = client.get_client_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test client remoto: {e}")
        return False

def test_factory():
    """Test factory embedders"""
    print("\n🧪 TEST: Factory Embedders")  
    print("=" * 50)
    
    try:
        from EmbeddingEngine.embedder_factory import EmbedderFactory, EmbedderType
        
        # Test tipi disponibili
        print("📋 Tipi embedder disponibili:")
        for etype in EmbedderFactory.get_available_types():
            print(f"   - {etype.value}")
        
        # Test creazione auto
        print("🤖 Test creazione automatica...")
        embedder = EmbedderFactory.create_embedder(
            embedder_type=EmbedderType.LABSE_AUTO,
            shared_instance=True
        )
        
        # Test funzionamento
        test_texts = ["Test factory", "Secondo test"]
        embeddings = embedder.encode(test_texts)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Test cache
        print("💾 Test cache istanze...")
        embedder2 = EmbedderFactory.create_embedder(
            embedder_type=EmbedderType.LABSE_AUTO,
            shared_instance=True
        )
        
        print(f"🔍 Stessa istanza? {embedder is embedder2}")
        
        # Statistiche cache
        cached = EmbedderFactory.get_cached_instances()
        print(f"📦 Istanze in cache: {len(cached)}")
        for key, instance_type in cached.items():
            print(f"   {key}: {instance_type}")
        
        # Health check
        print("🏥 Health check istanze...")
        health = EmbedderFactory.health_check()
        print(f"✅ Istanze healthy: {health['healthy_count']}")
        print(f"❌ Istanze unhealthy: {health['unhealthy_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test factory: {e}")
        return False

def test_compatibility():
    """Test compatibilità con codice esistente"""
    print("\n🧪 TEST: Compatibilità Codice Esistente")
    print("=" * 50)
    
    try:
        # Test import esistenti
        print("📦 Test import moduli esistenti...")
        
        # Simula uso in pipeline
        print("🔄 Test simulato pipeline...")
        from EmbeddingEngine.embedder_factory import get_embedder
        
        embedder = get_embedder(
            config={'service_url': 'http://localhost:8080'},
            embedder_type='labse_auto'
        )
        
        # Test method signature compatibilità
        test_texts = [
            "Messaggio di test per pipeline",
            "Secondo messaggio per verifica compatibilità"
        ]
        
        # Test parametri standard
        embeddings1 = embedder.encode(test_texts)
        embeddings2 = embedder.encode(test_texts, normalize_embeddings=True)
        embeddings3 = embedder.encode(test_texts, batch_size=1)
        
        print(f"✅ Test 1 - Shape: {embeddings1.shape}")
        print(f"✅ Test 2 - Shape: {embeddings2.shape}")  
        print(f"✅ Test 3 - Shape: {embeddings3.shape}")
        
        # Verifica normalizzazione
        norms1 = np.linalg.norm(embeddings1, axis=1)
        norms2 = np.linalg.norm(embeddings2, axis=1)
        print(f"📊 Norme non normalizzate: {norms1}")
        print(f"📊 Norme normalizzate: {norms2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test compatibilità: {e}")
        return False

def test_performance():
    """Test performance embedding"""
    print("\n🧪 TEST: Performance")
    print("=" * 50)
    
    try:
        from EmbeddingEngine.labse_remote_client import LaBSERemoteClient
        
        client = LaBSERemoteClient(service_url="http://localhost:8080")
        
        # Genera testi test
        test_sizes = [1, 5, 10, 20, 50]
        results = {}
        
        for size in test_sizes:
            texts = [f"Testo di test numero {i} per performance" for i in range(size)]
            
            print(f"⚡ Test {size} testi...")
            start_time = time.time()
            embeddings = client.encode(texts)
            elapsed = time.time() - start_time
            
            results[size] = {
                'time': elapsed,
                'speed': size / elapsed,
                'shape': embeddings.shape
            }
            
            print(f"   ⏱️ Tempo: {elapsed:.3f}s")
            print(f"   🚀 Velocità: {results[size]['speed']:.1f} testi/sec")
        
        # Report finale
        print("\n📊 REPORT PERFORMANCE:")
        print("-" * 30)
        for size, data in results.items():
            print(f"{size:2d} testi: {data['time']:.3f}s ({data['speed']:.1f} t/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test performance: {e}")
        return False

def test_error_handling():
    """Test gestione errori"""
    print("\n🧪 TEST: Gestione Errori")
    print("=" * 50)
    
    try:
        from EmbeddingEngine.labse_remote_client import LaBSERemoteClient
        
        # Test servizio non raggiungibile
        print("🔌 Test servizio non raggiungibile...")
        client = LaBSERemoteClient(
            service_url="http://localhost:9999",  # Porta inesistente
            timeout=5,
            max_retries=1,
            fallback_local=False
        )
        
        try:
            embeddings = client.encode("Test errore")
            print("❌ Dovrebbe aver fallito!")
            return False
        except Exception as e:
            print(f"✅ Errore gestito correttamente: {type(e).__name__}")
        
        # Test con fallback locale
        print("🔄 Test con fallback locale...")
        client_fallback = LaBSERemoteClient(
            service_url="http://localhost:9999",
            timeout=5,
            max_retries=1,
            fallback_local=True
        )
        
        try:
            embeddings = client_fallback.encode("Test fallback")
            print(f"✅ Fallback funziona: shape {embeddings.shape}")
        except Exception as e:
            print(f"⚠️ Anche fallback fallito: {e}")
        
        # Statistiche errori
        stats = client_fallback.get_client_stats()
        print(f"📈 Statistiche errori: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test gestione errori: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("🚀 AVVIO TEST INTEGRAZIONE SERVIZIO EMBEDDING")
    print("=" * 60)
    
    # Lista test da eseguire
    tests = [
        ("Client Remoto", test_remote_client),
        ("Factory", test_factory), 
        ("Compatibilità", test_compatibility),
        ("Performance", test_performance),
        ("Gestione Errori", test_error_handling)
    ]
    
    results = {}
    total_start = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"💥 ERRORE CRITICO in {test_name}: {e}")
            results[test_name] = False
        
        time.sleep(1)  # Pausa tra test
    
    total_elapsed = time.time() - total_start
    
    # Report finale
    print("\n" + "=" * 60)
    print("📋 REPORT FINALE TEST")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL" 
        print(f"{test_name:20s}: {status}")
    
    print("-" * 40)
    print(f"Test superati: {passed}/{total}")
    print(f"Tasso successo: {(passed/total)*100:.1f}%")
    print(f"Tempo totale: {total_elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST SUPERATI!")
        return True
    else:
        print(f"\n⚠️ {total-passed} TEST FALLITI")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
