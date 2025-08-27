#!/usr/bin/env python3
"""
Test del nuovo sistema embedding semplificato
"""

import os
import sys

# Aggiungi path dinamici relativi
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'EmbeddingEngine'))

def test_simple_embedding_manager():
    """Test del nuovo SimpleEmbeddingManager"""
    print("🧪 TESTING SimpleEmbeddingManager")
    print("=" * 50)
    
    try:
        from EmbeddingEngine.simple_embedding_manager import simple_embedding_manager
        
        # Test 1: Primo embedder
        print("\n📝 TEST 1: Primo embedder per wopta")
        embedder1 = simple_embedding_manager.get_embedder_for_tenant('wopta')
        print(f"✅ Embedder ottenuto: {type(embedder1).__name__}")
        
        # Test status
        status = simple_embedding_manager.get_status()
        print(f"📊 Status: {status}")
        
        # Test 2: Stesso tenant (dovrebbe riusare)
        print("\n📝 TEST 2: Stesso tenant (riuso)")
        embedder2 = simple_embedding_manager.get_embedder_for_tenant('wopta')
        print(f"✅ Embedder ottenuto: {type(embedder2).__name__}")
        print(f"🔍 Stesso oggetto? {embedder1 is embedder2}")
        
        # Test 3: Force reset
        print("\n📝 TEST 3: Force reset")
        simple_embedding_manager.force_reset("Test reset")
        
        # Test 4: Dopo reset
        print("\n📝 TEST 4: Dopo reset")
        embedder3 = simple_embedding_manager.get_embedder_for_tenant('wopta')
        print(f"✅ Embedder ottenuto: {type(embedder3).__name__}")
        print(f"🔍 Diverso dal precedente? {embedder1 is not embedder3}")
        
        status = simple_embedding_manager.get_status()
        print(f"📊 Status finale: {status}")
        
        print("\n✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
        
    except Exception as e:
        print(f"❌ ERRORE NEI TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_embedding_manager()
