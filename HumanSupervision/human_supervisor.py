#!/usr/bin/env python3
"""
⚠️ DEPRECATO: Modulo per gestire la supervisione umana nei casi di disaccordo tra LLM e ML

AVVISO DI DEPRECAZIONE:
Questa classe è stata sostituita da QualityGateEngine + React UI per gestione asincrona.
Migra al nuovo sistema per maggiore scalabilità e user experience.

Vedi: HumanSupervision/DEPRECATED.md per dettagli sulla migrazione.
"""

import yaml
import warnings
from typing import Dict, Any, Optional
from datetime import datetime

# Emetti warning di deprecazione
warnings.warn(
    "HumanSupervision è deprecato. Usa QualityGateEngine + React UI invece.",
    DeprecationWarning,
    stacklevel=2
)

class HumanSupervision:
    """
    Gestisce l'interazione umana per risolvere disaccordi tra classificatori
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inizializza il sistema di supervisione umana
        
        Args:
            config_path: Percorso del file di configurazione
        """
        self.config_path = config_path
        self.supervision_enabled = self._load_supervision_config()
        
    def _load_supervision_config(self) -> bool:
        """Carica la configurazione di supervisione dal file YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('pipeline', {}).get('supervisioneUmana', False)
        except Exception as e:
            print(f"⚠️ Errore nel caricamento configurazione: {e}")
            return False
    
    def is_supervision_enabled(self) -> bool:
        """Verifica se la supervisione umana è abilitata"""
        return self.supervision_enabled
    
    def handle_disagreement(self, 
                           text: str,
                           llm_prediction: Dict[str, Any],
                           ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gestisce il disaccordo tra LLM e ML chiedendo input all'umano
        
        Args:
            text: Testo da classificare
            llm_prediction: Predizione del LLM
            ml_prediction: Predizione del ML
            
        Returns:
            Decisione finale con feedback umano
        """
        if not self.supervision_enabled:
            # Se supervisione disabilitata, ritorna il classificatore con confidenza più alta
            if llm_prediction['confidence'] >= ml_prediction['confidence']:
                return {
                    'predicted_label': llm_prediction['predicted_label'],
                    'confidence': llm_prediction['confidence'],
                    'method': 'LLM_AUTO',
                    'human_intervention': False
                }
            else:
                return {
                    'predicted_label': ml_prediction['predicted_label'], 
                    'confidence': ml_prediction['confidence'],
                    'method': 'ML_AUTO',
                    'human_intervention': False
                }
        
        # Mostra il disaccordo all'umano
        print("\n" + "="*80)
        print("🤖 DISACCORDO TRA CLASSIFICATORI - SUPERVISIONE UMANA RICHIESTA")
        print("="*80)
        
        # Mostra il testo (troncato se troppo lungo)
        text_preview = text[:300] + "..." if len(text) > 300 else text
        print(f"\n📄 Testo da classificare:")
        print(f"'{text_preview}'")
        
        print(f"\n🧠 Predizione LLM:")
        print(f"   Etichetta: {llm_prediction['predicted_label']}")
        print(f"   Confidenza: {llm_prediction['confidence']:.3f}")
        if 'motivation' in llm_prediction:
            print(f"   Motivazione: {llm_prediction['motivation']}")
        
        print(f"\n🤖 Predizione ML:")
        print(f"   Etichetta: {ml_prediction['predicted_label']}")
        print(f"   Confidenza: {ml_prediction['confidence']:.3f}")
        
        # Menu di scelta per l'umano
        print(f"\n🎯 Opzioni disponibili:")
        print(f"   1. Usa etichetta LLM: '{llm_prediction['predicted_label']}'")
        print(f"   2. Usa etichetta ML: '{ml_prediction['predicted_label']}'")
        print(f"   3. Crea nuova etichetta personalizzata")
        print(f"   4. Salta questo elemento (usa LLM automaticamente)")
        
        while True:
            try:
                choice = input("\n👤 Inserisci la tua scelta (1-4): ").strip()
                
                if choice == "1":
                    return {
                        'predicted_label': llm_prediction['predicted_label'],
                        'confidence': 1.0,  # Confidenza massima per scelta umana
                        'method': 'HUMAN_LLM',
                        'human_intervention': True,
                        'human_choice': 'llm',
                        'timestamp': datetime.now().isoformat()
                    }
                
                elif choice == "2":
                    return {
                        'predicted_label': ml_prediction['predicted_label'],
                        'confidence': 1.0,  # Confidenza massima per scelta umana
                        'method': 'HUMAN_ML',
                        'human_intervention': True,
                        'human_choice': 'ml',
                        'timestamp': datetime.now().isoformat()
                    }
                
                elif choice == "3":
                    # Chiedi nuova etichetta
                    custom_label = self._get_custom_label()
                    if custom_label:
                        return {
                            'predicted_label': custom_label,
                            'confidence': 1.0,  # Confidenza massima per scelta umana
                            'method': 'HUMAN_CUSTOM',
                            'human_intervention': True,
                            'human_choice': 'custom',
                            'custom_label': custom_label,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        print("❌ Etichetta vuota, riprova...")
                        continue
                
                elif choice == "4":
                    return {
                        'predicted_label': llm_prediction['predicted_label'],
                        'confidence': llm_prediction['confidence'],
                        'method': 'HUMAN_SKIP_LLM',
                        'human_intervention': True,
                        'human_choice': 'skip',
                        'timestamp': datetime.now().isoformat()
                    }
                
                else:
                    print("❌ Scelta non valida. Inserisci un numero da 1 a 4.")
                    continue
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interruzione utente - uso automaticamente LLM")
                return {
                    'predicted_label': llm_prediction['predicted_label'],
                    'confidence': llm_prediction['confidence'],
                    'method': 'HUMAN_INTERRUPTED_LLM',
                    'human_intervention': True,
                    'human_choice': 'interrupted'
                }
            except Exception as e:
                print(f"❌ Errore nell'input: {e}")
                continue
    
    def _get_custom_label(self) -> Optional[str]:
        """
        Chiede all'umano di inserire una nuova etichetta personalizzata
        
        Returns:
            Nuova etichetta o None se vuota
        """
        print(f"\n📝 Etichette comuni esistenti:")
        common_labels = [
            'problema_accesso_portale',
            'prenotazione_esami', 
            'ritiro_referti',
            'problemi_prenotazione',
            'fatturazione_pagamenti',
            'orari_contatti',
            'servizi_supporto',
            'assistenza_medica_specializzata',
            'preparazione_esami',
            'altro'
        ]
        
        for i, label in enumerate(common_labels, 1):
            print(f"   {i:2}. {label}")
        
        while True:
            try:
                custom_input = input("\n👤 Inserisci nuova etichetta (o numero da lista sopra): ").strip()
                
                if not custom_input:
                    return None
                
                # Verifica se l'utente ha scelto un numero dalla lista
                if custom_input.isdigit():
                    choice_num = int(custom_input)
                    if 1 <= choice_num <= len(common_labels):
                        return common_labels[choice_num - 1]
                    else:
                        print(f"❌ Numero non valido. Scegli da 1 a {len(common_labels)}")
                        continue
                
                # Validazione etichetta personalizzata
                if len(custom_input) < 3:
                    print("❌ Etichetta troppo corta (minimo 3 caratteri)")
                    continue
                
                if len(custom_input) > 50:
                    print("❌ Etichetta troppo lunga (massimo 50 caratteri)")
                    continue
                
                # Normalizza l'etichetta (sostituisci spazi con underscore, lowercase)
                normalized_label = custom_input.lower().replace(' ', '_').replace('-', '_')
                
                # Conferma con l'utente
                confirm = input(f"\n✅ Confermi l'etichetta '{normalized_label}'? (s/n): ").strip().lower()
                if confirm in ['s', 'si', 'y', 'yes']:
                    return normalized_label
                elif confirm in ['n', 'no']:
                    continue
                else:
                    print("❌ Risposta non valida. Usa 's' per confermare o 'n' per annullare.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interruzione utente")
                return None
            except Exception as e:
                print(f"❌ Errore nell'input: {e}")
                continue
    
    def log_human_decision(self, 
                          text: str,
                          decision: Dict[str, Any],
                          llm_pred: Dict[str, Any],
                          ml_pred: Dict[str, Any]) -> None:
        """
        Registra la decisione umana per analisi future
        
        Args:
            text: Testo classificato
            decision: Decisione finale
            llm_pred: Predizione LLM originale
            ml_pred: Predizione ML originale
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'llm_prediction': llm_pred,
            'ml_prediction': ml_pred,
            'human_decision': decision,
            'supervision_enabled': self.supervision_enabled
        }
        
        try:
            # Salva in un file di log JSON per analisi future
            import json
            log_file = "human_supervision_log.json"
            
            # Carica log esistenti
            logs = []
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except FileNotFoundError:
                pass
            
            # Aggiungi nuovo log
            logs.append(log_entry)
            
            # Salva aggiornato
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
            print(f"📊 Decisione registrata in {log_file}")
            
        except Exception as e:
            print(f"⚠️ Errore nel logging della decisione: {e}")
