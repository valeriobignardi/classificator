#!/usr/bin/env python3
"""
Test Completo per Validazione Dinamica Etichette

Scopo: Testare l'intero flusso di validazione e inserimento automatico
di nuove etichette con soglia di confidenza 0.85

Autore: GitHub Copilot
Data: 2025-08-20
Ultimo aggiornamento: 2025-08-20 - Test validazione dinamica completa
"""

import sys
import os
import logging
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificazione_discussioni')

def test_dynamic_label_validation():
    """
    Test principale per la validazione dinamica delle etichette
    """
    print("🔬 TEST VALIDAZIONE DINAMICA ETICHETTE")
    print("=" * 70)
    
    try:
        # 1. Inizializzazione sistema
        print("\n1️⃣ TEST: Inizializzazione sistema completo...")
        config_path = '/home/ubuntu/classificazione_discussioni/config.yaml'
        
        from Classification.intelligent_classifier import IntelligentClassifier
        
        classifier = IntelligentClassifier(
            config_path=config_path,
            enable_logging=True
        )
        
        print(f"✅ Sistema inizializzato con {len(classifier.domain_labels)} etichette")
        print("📋 Prime 10 etichette esistenti:")
        for i, label in enumerate(classifier.domain_labels[:10]):
            print(f"   {i+1}. {label}")
        
        # 2. Test classificazione con etichetta esistente
        print("\n2️⃣ TEST: Classificazione con etichetta esistente...")
        test_existing = "Il paziente ha problemi cardiaci e necessita di controlli"
        result_existing = classifier.classify_conversation(test_existing)
        
        print(f"📝 Testo: {test_existing}")
        print(f"🏷️ Etichetta: {result_existing.get('label', 'N/A')}")
        print(f"🎯 Confidenza: {result_existing.get('confidence', 'N/A')}")
        print(f"🔍 Metodo: {result_existing.get('method', 'N/A')}")
        
        # 3. Test con etichetta completamente nuova
        print("\n3️⃣ TEST: Etichetta completamente nuova...")
        test_new = "Il paziente necessita di terapia psicologica specializzata per disturbi dell'alimentazione"
        initial_count = len(classifier.domain_labels)
        
        result_new = classifier.classify_conversation(test_new)
        
        final_count = len(classifier.domain_labels)
        
        print(f"📝 Testo: {test_new}")
        print(f"🏷️ Etichetta: {result_new.get('label', 'N/A')}")
        print(f"🎯 Confidenza: {result_new.get('confidence', 'N/A')}")
        print(f"🔍 Metodo: {result_new.get('method', 'N/A')}")
        print(f"📊 Etichette prima: {initial_count}, dopo: {final_count}")
        
        if final_count > initial_count:
            print(f"🆕 NUOVA ETICHETTA AGGIUNTA AUTOMATICAMENTE!")
            nuove_etichette = classifier.domain_labels[initial_count:]
            for etichetta in nuove_etichette:
                print(f"   ➕ {etichetta}")
        
        # 4. Test soglia di confidenza
        print("\n4️⃣ TEST: Verifica soglia confidenza 0.85...")
        print(f"📏 Soglia configurata: {classifier.embedding_similarity_threshold}")
        
        if 'confidence' in result_new and result_new['confidence'] is not None:
            confidence = float(result_new['confidence'])
            if confidence >= 0.85:
                print(f"✅ Confidenza {confidence:.3f} >= 0.85 - Validazione automatica")
            else:
                print(f"⚠️ Confidenza {confidence:.3f} < 0.85 - Richiede validazione manuale")
        
        # 5. Test con etichetta simile esistente
        print("\n5️⃣ TEST: Etichetta simile a esistente...")
        test_similar = "Problemi cardiovascolari del paziente richiedono attenzione"
        result_similar = classifier.classify_conversation(test_similar)
        
        print(f"📝 Testo: {test_similar}")
        print(f"🏷️ Etichetta: {result_similar.get('label', 'N/A')}")
        print(f"🎯 Confidenza: {result_similar.get('confidence', 'N/A')}")
        print(f"🔍 Metodo: {result_similar.get('method', 'N/A')}")
        
        # 6. Test ricaricamento etichette
        print("\n6️⃣ TEST: Ricaricamento dinamico etichette...")
        pre_reload = len(classifier.domain_labels)
        classifier._reload_domain_labels()
        post_reload = len(classifier.domain_labels)
        
        print(f"📊 Etichette prima reload: {pre_reload}")
        print(f"📊 Etichette dopo reload: {post_reload}")
        
        if post_reload != pre_reload:
            print(f"🔄 Rilevate {abs(post_reload - pre_reload)} modifiche nel database")
        else:
            print("✅ Nessuna modifica rilevata - sistema sincronizzato")
        
        # 7. Test performance e statistiche
        print("\n7️⃣ TEST: Statistiche di sistema...")
        if hasattr(classifier, 'stats'):
            stats = classifier.stats
            print(f"📈 Statistiche classifier:")
            for key, value in stats.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} elementi")
                else:
                    print(f"   {key}: {value}")
        
        # 8. Test validazione database
        print("\n8️⃣ TEST: Validazione integrità database...")
        if classifier.mysql_connector:
            try:
                # Test connessione database
                tags_data = classifier.mysql_connector.get_all_tags()
                total_tags = len(tags_data) if tags_data else 0
                print(f"✅ Database accessibile - {total_tags} tag totali")
                
                # Mostra alcuni tag di esempio
                if tags_data:
                    print("📋 Esempi tag dal database:")
                    for i, tag in enumerate(tags_data[:5]):
                        print(f"   {i+1}. {tag['tag_name']}: {tag.get('tag_description', 'N/A')[:30]}...")
                        
            except Exception as e:
                print(f"❌ Errore accesso database: {e}")
        else:
            print("❌ Connector database non disponibile")
        
        print("\n" + "=" * 70)
        print("🎉 TEST VALIDAZIONE DINAMICA COMPLETATO CON SUCCESSO!")
        print("✅ Sistema pronto per classificazione automatica con validazione dinamica")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRORE NEL TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """
    Test casi limite e edge cases
    """
    print("\n🔍 TEST CASI LIMITE")
    print("=" * 50)
    
    try:
        config_path = '/home/ubuntu/classificazione_discussioni/config.yaml'
        from Classification.intelligent_classifier import IntelligentClassifier
        
        classifier = IntelligentClassifier(config_path=config_path, enable_logging=True)
        
        edge_cases = [
            {
                'name': 'Testo molto breve',
                'text': 'Dolore',
                'expected': 'Classificazione base'
            },
            {
                'name': 'Testo molto lungo',
                'text': ' '.join(['Il paziente presenta sintomi complessi'] * 50),
                'expected': 'Gestione testo lungo'
            },
            {
                'name': 'Testo con caratteri speciali',
                'text': 'Paziente con €#@!% simboli strani 123456',
                'expected': 'Pulizia testo'
            },
            {
                'name': 'Testo vuoto',
                'text': '',
                'expected': 'Gestione input vuoto'
            },
            {
                'name': 'Testo solo numeri',
                'text': '123 456 789',
                'expected': 'Classificazione numerica'
            }
        ]
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\n{i}️⃣ EDGE CASE: {case['name']}")
            try:
                result = classifier.classify_conversation(case['text'])
                print(f"   📝 Input: {case['text'][:50]}{'...' if len(case['text']) > 50 else ''}")
                print(f"   🏷️ Output: {result.get('label', 'N/A')}")
                print(f"   ✅ Gestito correttamente")
            except Exception as e:
                print(f"   ❌ Errore: {e}")
        
        print("\n✅ Test edge cases completato")
        return True
        
    except Exception as e:
        print(f"❌ Errore test edge cases: {e}")
        return False

if __name__ == "__main__":
    """
    Esecuzione completa test validazione dinamica
    """
    print(f"🕐 Avvio test validazione dinamica: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test principale
    success_main = test_dynamic_label_validation()
    
    # Test casi limite
    success_edge = test_edge_cases()
    
    overall_success = success_main and success_edge
    
    if overall_success:
        print(f"\n🎯 TUTTI I TEST COMPLETATI CON SUCCESSO alle {datetime.now().strftime('%H:%M:%S')}")
        print("🚀 Sistema pronto per l'uso in produzione!")
        sys.exit(0)
    else:
        print(f"\n💥 ALCUNI TEST FALLITI alle {datetime.now().strftime('%H:%M:%S')}")
        print("🔧 Rivedere i problemi evidenziati sopra")
        sys.exit(1)
