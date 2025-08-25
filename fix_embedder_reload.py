#!/usr/bin/env python3
"""
Script per forzare il reload dell'embedder dopo cambio configurazione

Scopo: Risolvere il problema di cache embedder quando si cambia da LaBSE a OpenAI
"""

import sys
import os

# Aggiunta percorsi per imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'EmbeddingEngine'))

try:
    from embedding_engine_factory import embedding_factory
    
    def fix_embedder_for_tenant(tenant_id: str):
        """Forza reload embedder per tenant"""
        print(f"🔄 Forzando reload embedder per tenant: {tenant_id}")
        
        # Pulisce cache per il tenant specifico
        embedding_factory.clear_cache(tenant_id)
        
        # Ricarica embedder con configurazione aggiornata
        new_embedder = embedding_factory.reload_tenant_embedder(tenant_id)
        
        print(f"✅ Embedder ricaricato: {type(new_embedder).__name__}")
        
        # Test embedder
        try:
            if hasattr(new_embedder, 'test_model'):
                success = new_embedder.test_model()
                print(f"🧪 Test embedder: {'✅ OK' if success else '❌ FALLITO'}")
            
            # Test encoding semplice
            test_embedding = new_embedder.encode("Test embedding")
            print(f"📊 Test encoding: shape {test_embedding.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore test embedder: {e}")
            return False

    if __name__ == "__main__":
        # Tenant di test (Wopta)
        tenant_uuid = "16c222a9-f293-11ef-9315-96000228e7fe"
        
        print("🔧 Fix embedder reload per cambio da LaBSE a OpenAI")
        print(f"📋 Tenant: {tenant_uuid}")
        
        success = fix_embedder_for_tenant(tenant_uuid)
        
        if success:
            print("✅ Fix completato con successo!")
        else:
            print("❌ Fix fallito!")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Errore import: {e}")
    sys.exit(1)
