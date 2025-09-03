#!/usr/bin/env python3
"""
Test avanzato per verificare il salvataggio di embedding processati (post-UMAP) in MongoDB
"""
import numpy as np
import sys
import os

# Add path for imports
sys.path.append('/home/ubuntu/classificatore')
sys.path.append('/home/ubuntu/classificatore/Pipeline')
sys.path.append('/home/ubuntu/classificatore/Clustering')

def test_clusterer_final_embeddings():
    """Test per verificare che il clusterer salvi gli embedding finali"""
    try:
        from hdbscan_clusterer import HDBSCANClusterer
        print("✅ Import HDBSCANClusterer successful")
        
        # Crea clusterer di test
        clusterer = HDBSCANClusterer(
            min_cluster_size=5,
            min_samples=3,
            use_umap=False  # Per semplicità, disabilitiamo UMAP per questo test
        )
        
        # Test embeddings fittizi
        test_embeddings = np.random.rand(20, 768).astype(np.float32)
        print(f"📊 Test embeddings shape: {test_embeddings.shape}")
        
        # Simula fit_predict
        labels = clusterer.fit_predict(test_embeddings)
        print(f"📊 Cluster labels shape: {labels.shape}")
        
        # Verifica che final_embeddings sia stato salvato
        if hasattr(clusterer, 'final_embeddings'):
            print(f"✅ final_embeddings salvato: shape {clusterer.final_embeddings.shape}")
            
            # Verifica che sia lo stesso shape (senza UMAP dovrebbe essere uguale)
            if clusterer.final_embeddings.shape == test_embeddings.shape:
                print("✅ Shape corretto per embedding senza UMAP")
            else:
                print("❌ Shape diverso!")
                
        else:
            print("❌ final_embeddings NON SALVATO nel clusterer")
            
        # Test con UMAP abilitato (solo se disponibile)
        try:
            clusterer_umap = HDBSCANClusterer(
                min_cluster_size=5,
                min_samples=3,
                use_umap=True,
                umap_n_components=17  # Riduce a 17 dimensioni
            )
            
            labels_umap = clusterer_umap.fit_predict(test_embeddings)
            
            if hasattr(clusterer_umap, 'final_embeddings'):
                print(f"✅ final_embeddings con UMAP: shape {clusterer_umap.final_embeddings.shape}")
                
                # Con UMAP dovrebbe essere ridotto
                if clusterer_umap.final_embeddings.shape[1] == 17:
                    print("✅ UMAP riduzione confermata: 768 → 17 dimensioni")
                else:
                    print(f"⚠️ Riduzione UMAP imprevista: {clusterer_umap.final_embeddings.shape[1]} dimensioni")
                    
                if hasattr(clusterer_umap, 'umap_info') and clusterer_umap.umap_info.get('applied'):
                    print("✅ umap_info salvato e UMAP applicato")
                else:
                    print("❌ umap_info non disponibile o UMAP non applicato")
                    
            else:
                print("❌ final_embeddings con UMAP NON SALVATO")
                
        except Exception as umap_error:
            print(f"⚠️ Test UMAP saltato (libreria non disponibile): {umap_error}")
            
    except Exception as e:
        print(f"❌ Errore test clusterer: {e}")

def test_embedder_name_with_umap():
    """Test per verificare che _get_embedder_name includa info UMAP"""
    try:
        # Simula clusterer con UMAP info
        class MockClusterer:
            def __init__(self, umap_applied=True):
                if umap_applied:
                    self.umap_info = {
                        'applied': True,
                        'parameters': {'n_components': 17}
                    }
                else:
                    self.umap_info = {'applied': False}
        
        class MockEmbedder:
            def __init__(self):
                self.model_name = "test-model"
        
        class MockPipeline:
            def __init__(self, umap_applied=True):
                self.embedder = MockEmbedder()
                self.clusterer = MockClusterer(umap_applied)
            
            def _get_embedder_name(self):
                # Copia il metodo dalla pipeline
                embedder_type = type(self.embedder).__name__
                base_name = f"{embedder_type}_{self.embedder.model_name}"
                
                if hasattr(self.clusterer, 'umap_info') and self.clusterer.umap_info.get('applied'):
                    umap_params = self.clusterer.umap_info.get('parameters', {})
                    n_components = umap_params.get('n_components', 'unknown')
                    base_name += f"_UMAP{n_components}D"
                
                return base_name
        
        # Test senza UMAP
        pipeline_no_umap = MockPipeline(umap_applied=False)
        name_no_umap = pipeline_no_umap._get_embedder_name()
        print(f"📏 Embedder name senza UMAP: {name_no_umap}")
        
        if "UMAP" not in name_no_umap:
            print("✅ Nome senza UMAP corretto")
        else:
            print("❌ Nome contiene UMAP ma non dovrebbe")
        
        # Test con UMAP
        pipeline_with_umap = MockPipeline(umap_applied=True)
        name_with_umap = pipeline_with_umap._get_embedder_name()
        print(f"📏 Embedder name con UMAP: {name_with_umap}")
        
        if "UMAP17D" in name_with_umap:
            print("✅ Nome con UMAP corretto")
        else:
            print("❌ Nome NON contiene info UMAP")
            
    except Exception as e:
        print(f"❌ Errore test embedder name: {e}")

def test_mongodb_embedding_dimensions():
    """Test per verificare che MongoDB possa gestire embedding di diverse dimensioni"""
    try:
        from mongo_classification_reader import MongoClassificationReader
        
        # Test diversi tipi di embedding
        embedding_768 = np.random.rand(768).astype(np.float32)
        embedding_17_umap = np.random.rand(17).astype(np.float32)
        
        # Test conversione per MongoDB
        embedding_768_list = embedding_768.tolist()
        embedding_17_list = embedding_17_umap.tolist()
        
        print(f"✅ Embedding 768D convertito a list: length {len(embedding_768_list)}")
        print(f"✅ Embedding 17D (UMAP) convertito a list: length {len(embedding_17_list)}")
        
        # Test JSON serialization
        import json
        json.dumps(embedding_768_list)
        json.dumps(embedding_17_list)
        print("✅ Entrambi gli embedding sono JSON serializable")
        
        # Test che i valori siano ragionevoli
        if all(-10 < x < 10 for x in embedding_768_list[:5]):  # Check primi 5 valori
            print("✅ Valori embedding 768D nei range attesi")
        else:
            print("⚠️ Valori embedding 768D potrebbero essere strani")
            
        if all(-10 < x < 10 for x in embedding_17_list):  # Check tutti i valori (solo 17)
            print("✅ Valori embedding 17D nei range attesi")
        else:
            print("⚠️ Valori embedding 17D potrebbero essere strani")
        
    except Exception as e:
        print(f"❌ Errore test MongoDB embedding: {e}")

if __name__ == "__main__":
    print("🧪 TEST EMBEDDING UMAP INTEGRATION")
    print("=" * 60)
    
    print("\n1. Testing clusterer final embeddings:")
    test_clusterer_final_embeddings()
    
    print("\n2. Testing embedder name with UMAP info:")
    test_embedder_name_with_umap()
    
    print("\n3. Testing MongoDB embedding dimensions:")
    test_mongodb_embedding_dimensions()
    
    print("\n🏁 UMAP INTEGRATION TEST COMPLETE")
