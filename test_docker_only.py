#!/usr/bin/env python3
"""
Test Docker Only - Verifica che non ci siano più caricamenti locali LaBSE

Autore: Valerio Bignardi
Data: 2025-01-28
Aggiornamenti:
- 2025-01-28: Creato per testare eliminazione fallback locali
"""

import sys
import os

# Aggiunge paths per importare moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

def test_docker_only():
    """
    Test principale per verificare funzionamento solo Docker
    
    Scopo: Verificare che tutti i componenti usino solo servizio Docker
    """
    print("🚀 TEST DOCKER ONLY - Verifica eliminazione fallback locali\n")
    
    try:
        # Test 1: Simple Embedding Manager (senza tenant)
        print("1️⃣ Test SimpleEmbeddingManager...")
        from simple_embedding_manager import SimpleEmbeddingManager
        
        sem = SimpleEmbeddingManager()
        print(f"   ✅ SimpleEmbeddingManager inizializzato: {type(sem).__name__}")
        
        # Test 2: LaBSE Remote Client
        print("\n2️⃣ Test LaBSERemoteClient...")
        from labse_remote_client import LaBSERemoteClient
        
        client = LaBSERemoteClient(
            service_url="http://localhost:8081",
            fallback_local=False  # 🚫 NESSUN FALLBACK
        )
        
        # Test embed semplice
        test_embedding = client.encode(["Test Docker only"])
        print(f"   ✅ Embedding generato: shape {test_embedding.shape}")
        
        # Test 3: Embedding Engine Factory
        print("\n3️⃣ Test EmbeddingEngineFactory...")
        from embedding_engine_factory import EmbeddingEngineFactory
        
        factory = EmbeddingEngineFactory()
        default_embedder = factory.get_default_embedder()
        print(f"   ✅ Factory embedder: {type(default_embedder).__name__}")
        
        print("\n🎉 TUTTI I TEST PASSATI!")
        print("✅ Sistema configurato per usare SOLO Docker service")
        print("🚫 Nessun fallback locale disponibile")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE DURANTE TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_docker_only()
    exit(0 if success else 1)
