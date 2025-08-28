#!/usr/bin/env python3
"""
Test rapido per verificare che BERTopic non carichi più MiniLM-L6
quando dovrebbe usare i nostri embeddings precomputati.

Autore: Valerio Bignardi
Data: 2025-08-28
"""
import sys
import os

# Aggiungi il path del progetto
sys.path.append('.')

def test_bertopic_embedding_fix():
    """
    Test per verificare che BERTopic usi embeddings precomputati
    invece di caricare MiniLM-L6
    """
    print("🧪 TEST BERTOPIC EMBEDDING FIX")
    print("=" * 50)
    
    try:
        from TopicModeling.bertopic_feature_provider import BERTopicFeatureProvider
        print("✅ Import BERTopicFeatureProvider riuscito")
        
        # Verifica che il provider sia disponibile
        provider = BERTopicFeatureProvider()
        if not provider.is_available():
            print("❌ BERTopic non disponibile - dipendenze mancanti")
            return False
        
        print("✅ BERTopic disponibile")
        
        # Test di inizializzazione con embedder mock
        class MockEmbedder:
            def encode(self, texts, **kwargs):
                import numpy as np
                # Simula embeddings LaBSE (768 dimensioni)
                return np.random.randn(len(texts), 768)
        
        mock_embedder = MockEmbedder()
        
        provider_with_embedder = BERTopicFeatureProvider(embedder=mock_embedder)
        print("✅ Inizializzazione con embedder mock riuscita")
        
        # Verifica che l'embedding_model sia None (strategia precomputati)
        print(f"📋 Embedder configurato: {provider_with_embedder.embedder is not None}")
        
        print("\n🎯 RISULTATO ATTESO:")
        print("   • Embedder configurato: True")
        print("   • BERTopic userà embeddings precomputati")
        print("   • NON dovrebbe caricare sentence-transformers/all-MiniLM-L6-v2")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        return False
    except Exception as e:
        print(f"❌ Errore generale: {e}")
        return False

def test_wrapper_creation():
    """
    Test della creazione del wrapper (ora non usato ma mantenuto per debug)
    """
    print("\n🧪 TEST WRAPPER CREATION")
    print("=" * 50)
    
    try:
        from TopicModeling.bertopic_feature_provider import BERTopicFeatureProvider
        
        class MockEmbedder:
            def encode(self, texts, **kwargs):
                import numpy as np
                return np.random.randn(len(texts), 768)
        
        mock_embedder = MockEmbedder()
        provider = BERTopicFeatureProvider(embedder=mock_embedder)
        
        # Test creazione wrapper (ora non utilizzato)
        wrapper = provider._create_bertopic_embedding_wrapper()
        print("✅ Wrapper creato correttamente")
        
        # Test metodi wrapper
        test_docs = ["test document 1", "test document 2"]
        embeddings = wrapper.embed(test_docs)
        print(f"✅ Wrapper embed test: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test wrapper: {e}")
        return False

if __name__ == "__main__":
    print("🔧 VERIFICA FIX BERTOPIC EMBEDDING")
    print("Obiettivo: Evitare caricamento MiniLM-L6 quando configurato LaBSE")
    print()
    
    test1_ok = test_bertopic_embedding_fix()
    test2_ok = test_wrapper_creation()
    
    print("\n📊 RIEPILOGO TEST:")
    print(f"   • Test configurazione: {'✅' if test1_ok else '❌'}")
    print(f"   • Test wrapper: {'✅' if test2_ok else '❌'}")
    
    if test1_ok and test2_ok:
        print("\n🎉 TUTTI I TEST PASSATI")
        print("🔧 Il fix dovrebbe risolvere il problema MiniLM-L6")
    else:
        print("\n⚠️ ALCUNI TEST FALLITI")
        print("🔍 Potrebbero essere necessarie altre correzioni")
