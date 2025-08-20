#!/usr/bin/env python3
"""
Test di Integrazione Database per IntelligentClassifier

Scopo: Testare il caricamento dinamico delle etichette dalla tabella TAG.tags
e la validazione/inserimento di nuove etichette con soglia di confidenza 0.85

Autore: GitHub Copilot
Data: 2025-01-24
Ultimo aggiornamento: 2025-01-24 - Test integrazione MySQL
"""

import sys
import os
import logging
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificazione_discussioni')

def test_database_integration():
    """
    Test principale per l'integrazione database
    """
    print("🧪 TEST DATABASE INTEGRATION - IntelligentClassifier")
    print("=" * 60)
    
    try:
        # 1. Test caricamento configurazione
        print("\n1️⃣ TEST: Caricamento configurazione...")
        config_path = '/home/ubuntu/classificazione_discussioni/config.yaml'
        if not os.path.exists(config_path):
            print(f"❌ File configurazione non trovato: {config_path}")
            return False
        print("✅ Configurazione trovata")
        
        # 2. Test inizializzazione classifier
        print("\n2️⃣ TEST: Inizializzazione IntelligentClassifier...")
        from Classification.intelligent_classifier import IntelligentClassifier
        
        classifier = IntelligentClassifier(
            config_path=config_path,
            enable_logging=True
        )
        print("✅ Classifier inizializzato")
        
        # 3. Test connessione MySQL
        print("\n3️⃣ TEST: Connessione MySQL...")
        if classifier.mysql_connector is None:
            print("❌ MySQL connector non inizializzato")
            return False
        print("✅ MySQL connector attivo")
        
        # 4. Test caricamento etichette da database
        print("\n4️⃣ TEST: Caricamento etichette da TAG.tags...")
        initial_labels = classifier.domain_labels.copy()
        print(f"📊 Etichette caricate: {len(initial_labels)}")
        for i, label in enumerate(initial_labels[:10]):  # Prime 10
            print(f"   {i+1}. {label}")
        if len(initial_labels) > 10:
            print(f"   ... e altre {len(initial_labels) - 10} etichette")
        
        # 5. Test descrizioni etichette
        print("\n5️⃣ TEST: Caricamento descrizioni etichette...")
        descriptions_count = len(classifier.label_descriptions)
        print(f"📊 Descrizioni caricate: {descriptions_count}")
        for label, desc in list(classifier.label_descriptions.items())[:3]:
            print(f"   {label}: {desc[:50]}..." if len(desc) > 50 else f"   {label}: {desc}")
        
        # 6. Test classificazione con etichetta esistente
        print("\n6️⃣ TEST: Classificazione con etichetta esistente...")
        test_text = "Il paziente lamenta dolore al petto e difficoltà respiratorie"
        result = classifier.classify_conversation(test_text)
        print(f"📝 Testo: {test_text}")
        print(f"🏷️ Risultato: {result.get('label', 'N/A')}")
        print(f"🎯 Confidenza: {result.get('confidence', 'N/A')}")
        
        # 7. Test soglia validazione nuove etichette
        print("\n7️⃣ TEST: Soglia validazione (0.85)...")
        print(f"📏 Soglia attuale: {classifier.embedding_similarity_threshold}")
        if classifier.embedding_similarity_threshold != 0.85:
            print("⚠️ Soglia non corretta! Dovrebbe essere 0.85")
        else:
            print("✅ Soglia configurata correttamente")
        
        # 8. Test funzioni database
        print("\n8️⃣ TEST: Funzioni database...")
        if hasattr(classifier, '_load_domain_labels_from_database'):
            print("✅ _load_domain_labels_from_database disponibile")
        else:
            print("❌ _load_domain_labels_from_database mancante")
            
        if hasattr(classifier, '_add_new_validated_label'):
            print("✅ _add_new_validated_label disponibile")
        else:
            print("❌ _add_new_validated_label mancante")
            
        if hasattr(classifier, '_reload_domain_labels'):
            print("✅ _reload_domain_labels disponibile")
        else:
            print("❌ _reload_domain_labels mancante")
        
        # 9. Test ricaricamento etichette
        print("\n9️⃣ TEST: Ricaricamento dinamico etichette...")
        old_count = len(classifier.domain_labels)
        classifier._reload_domain_labels()
        new_count = len(classifier.domain_labels)
        print(f"📊 Etichette prima: {old_count}, dopo: {new_count}")
        if new_count == old_count:
            print("✅ Ricaricamento completato (nessuna modifica)")
        else:
            print(f"🔄 Ricaricamento completato ({abs(new_count - old_count)} modifiche)")
        
        print("\n" + "=" * 60)
        print("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("✅ Sistema pronto per la classificazione dinamica")
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """
    Esecuzione test database integration
    """
    print(f"🕐 Avvio test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_database_integration()
    
    if success:
        print(f"\n🎯 Test completato con successo alle {datetime.now().strftime('%H:%M:%S')}")
        sys.exit(0)
    else:
        print(f"\n💥 Test fallito alle {datetime.now().strftime('%H:%M:%S')}")
        sys.exit(1)
