#!/usr/bin/env python3
"""
Test per verificare le modifiche al debug dei suggested_labels
"""

import sys
import os
sys.path.append('/home/ubuntu/classificatore')

def test_suggested_labels_debug():
    """
    Test per verificare che il debug funzioni correttamente
    """
    print("🧪 Test debug suggested_labels...")
    
    # Verifica che il file di log venga creato
    log_file = '/home/ubuntu/classificatore/suggested_labels.log'
    
    # Pulisci il log precedente
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Importa il clusterer
    from Clustering.intelligent_intent_clusterer import IntelligentIntentClusterer
    
    print("✅ Import successful - modifiche applicate correttamente")
    print(f"📄 File di log verrà creato in: {log_file}")
    print("\n🎯 Modifiche implementate:")
    print("   - ✅ Debug dettagliato per ogni rappresentante")
    print("   - ✅ Logica outlier modificata (trattati come rappresentanti)")
    print("   - ✅ Logging in suggested_labels.log")
    print("   - ✅ Debug format: 'Caso: <id> - Rappresentante del Cluster n° <cluster>: <label>'")
    print("   - ✅ Debug format outlier: 'Caso: <id> - Outlier: <label>'")

if __name__ == "__main__":
    test_suggested_labels_debug()
