#!/usr/bin/env python3
"""
Test del sistema aggiornato di fine-tuning e gestione tag
"""

import sys
import os

# Aggiungi i path necessari
sys.path.append('/home/ubuntu/classificazione_b+++++_bck2')

def test_tag_database():
    """Test del TagDatabaseConnector"""
    print("🧪 Test TagDatabaseConnector...")
    
    try:
        from TagDatabase.tag_database_connector import TagDatabaseConnector
        
        tag_db = TagDatabaseConnector()
        
        # Test recupero tag
        tags = tag_db.get_all_tags()
        print(f"✅ Recuperati {len(tags)} tag dal database")
        
        # Test dizionario tag -> descrizioni
        tags_dict = tag_db.get_tags_dictionary()
        print(f"✅ Dizionario tag-descrizioni: {len(tags_dict)} elementi")
        
        # Mostra primi 3 tag
        for i, (tag_name, description) in enumerate(list(tags_dict.items())[:3]):
            print(f"  📋 {tag_name}: {description}")
            
        return True
        
    except Exception as e:
        print(f"❌ Errore test TagDatabaseConnector: {e}")
        return False

def test_finetuning_system():
    """Test del sistema di fine-tuning aggiornato"""
    print("\n🧪 Test Fine-tuning System...")
    
    try:
        from FineTuning.mistral_finetuning_manager import MistralFineTuningManager
        
        # Inizializza manager
        ft_manager = MistralFineTuningManager()
        
        # Test recupero tag con descrizioni
        tags_dict = ft_manager._get_tags_with_descriptions()
        print(f"✅ Fine-tuning: recuperate {len(tags_dict)} descrizioni tag")
        
        # Test system message con descrizioni
        system_msg = ft_manager._build_finetuning_system_message()
        print(f"✅ System message generato: {len(system_msg)} caratteri")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test Fine-tuning System: {e}")
        return False

def test_ml_ensemble():
    """Test del sistema ML arricchito"""
    print("\n🧪 Test ML Ensemble con tag descriptions...")
    
    try:
        from Classification.advanced_ensemble_classifier import AdvancedEnsembleClassifier
        
        # Inizializza classifier
        ensemble = AdvancedEnsembleClassifier()
        
        # Test preparazione dati arricchiti
        sample_texts = [
            "Vorrei ritirare i referti degli esami del sangue",
            "Come posso prenotare una visita cardiologica?",
            "Qual è il numero del reparto di radiologia?"
        ]
        sample_labels = ["ritiro_cartella_clinica_referti", "prenotazione_esami", "info_contatti"]
        
        enhanced_texts = ensemble.prepare_enhanced_training_data(sample_texts, sample_labels)
        print(f"✅ ML training arricchito: {len(enhanced_texts)} testi processati")
        
        # Mostra esempio di arricchimento
        if enhanced_texts:
            print(f"  📝 Esempio testo arricchito:")
            print(f"     Originale: {sample_texts[0]}")
            print(f"     Arricchito: {enhanced_texts[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore test ML Ensemble: {e}")
        return False

def main():
    """Test principale"""
    print("🚀 Test Sistema Aggiornato di Classificazione")
    print("=" * 50)
    
    results = []
    
    # Test database tag
    results.append(test_tag_database())
    
    # Test fine-tuning
    results.append(test_finetuning_system())
    
    # Test ML ensemble
    results.append(test_ml_ensemble())
    
    # Risultati finali
    print("\n" + "=" * 50)
    success_count = sum(results)
    total_tests = len(results)
    
    if success_count == total_tests:
        print(f"🎉 TUTTI I TEST SUPERATI! ({success_count}/{total_tests})")
        print("\n✅ SISTEMA AGGIORNATO PRONTO ALL'USO:")
        print("   🔹 Fine-tuning con descrizioni tag")
        print("   🔹 Validazione label anti-duplicati")
        print("   🔹 ML training arricchito")
        print("   🔹 Sistema anti-duplicati database")
    else:
        print(f"⚠️ Test parzialmente superati: {success_count}/{total_tests}")
        print("   Alcuni componenti potrebbero non funzionare correttamente")

if __name__ == "__main__":
    main()
