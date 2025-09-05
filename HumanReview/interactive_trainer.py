"""
Sistema di training interattivo con supervisione umana
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime

# Aggiunge i percorsi per importare gli altri moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LLMClassifier'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))

# Import Tenant per principio universale
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from tenant import Tenant

# Import per validatore ALTRO
from altro_tag_validator import AltroTagValidator

# Non importiamo più direttamente IntelligentClassifier
# Lo riceviamo come parametro per maggiore flessibilità

class InteractiveTrainer:
    """
    Gestisce il training supervisionato interattivo con feedback umano
    """
    
    def __init__(self, 
                 tenant: Optional[Tenant] = None,
                 llm_classifier: Optional[Any] = None, 
                 auto_mode: bool = False, 
                 bertopic_model: Optional[Any] = None):
        """
        Inizializza il trainer interattivo
        
        PRINCIPIO UNIVERSALE: Accetta oggetto Tenant completo
        
        Args:
            tenant: Oggetto Tenant completo (None per compatibilità)
            llm_classifier: Classificatore LLM per proposte automatiche
            auto_mode: Se True, utilizza solo proposte automatiche senza input umano
            bertopic_model: Modello BERTopic per validazione incrociata
        """
        self.tenant = tenant
        self.tenant_id = tenant.tenant_id if tenant else None  # Estrae tenant_id dall'oggetto
        self.llm_classifier = llm_classifier
        self.auto_mode = auto_mode
        self.bertopic_model = bertopic_model
        self.human_feedback = []  # Storico feedback umano
        self.approved_labels = {}  # Etichette approvate dall'umano
        
        # Inizializza validatore per tag "altro" se abbiamo tenant
        self.altro_validator = None
        if self.tenant:
            try:
                self.altro_validator = AltroTagValidator(tenant=self.tenant)
            except Exception as e:
                print(f"⚠️ Warning: Impossibile inizializzare AltroTagValidator: {e}")
                self.altro_validator = None
        
    def review_cluster_representatives(self, 
                                     cluster_id: int,
                                     representatives: List[Dict],
                                     suggested_label: str) -> Tuple[str, float]:
        """
        Presenta i rappresentanti di un cluster per review umano
        
        Args:
            cluster_id: ID del cluster
            representatives: Lista di sessioni rappresentative
            suggested_label: Etichetta suggerita dal clustering
            
        Returns:
            Tuple (etichetta_finale, confidenza_umana)
        """
        print(f"\n" + "="*80)
        print(f"📋 REVIEW CLUSTER {cluster_id}")
        print(f"🏷️  Etichetta suggerita: '{suggested_label}'")
        print("="*80)
        
        # Mostra rappresentanti del cluster
        print(f"\n📝 CONVERSAZIONI RAPPRESENTATIVE ({len(representatives)} esempi):")
        print("-"*60)
        
        for i, rep in enumerate(representatives[:3], 1):  # Max 3 esempi
            session_id = rep.get('session_id', f'unknown_{i}')
            conversation = rep.get('testo_completo', rep.get('conversation', ''))
            
            # Tronca se troppo lungo
            if len(conversation) > 500:
                conversation = conversation[:500] + "..."
            
            print(f"\n[{i}] Sessione: {session_id}")
            print(f"    Conversazione:")
            print(f"    {conversation}")
            print("-"*60)
        
        # Proposta LLM se disponibile
        llm_suggestion = None
        llm_confidence = 0.0
        llm_motivation = ""
        
        if self.llm_classifier and self.llm_classifier.is_available():
            print(f"\n🤖 PROPOSTA LLM:")
            try:
                # Usa la prima conversazione rappresentativa per la proposta LLM
                sample_text = representatives[0].get('testo_completo', representatives[0].get('conversation', ''))
                
                # Il metodo classify_with_motivation ritorna un oggetto ClassificationResult
                llm_result = self.llm_classifier.classify_with_motivation(sample_text)
                llm_suggestion = llm_result.predicted_label
                llm_confidence = llm_result.confidence
                llm_motivation = llm_result.motivation
                
                print(f"   Etichetta: '{llm_suggestion}' (confidenza: {llm_confidence:.3f})")
                print(f"   Motivazione: {llm_motivation}")
                
            except Exception as e:
                print(f"   ❌ Errore LLM: {e}")
        
        # Input umano
        print(f"\n👤 DECISIONE:")
        
        # Se modalità automatica, usa la migliore opzione disponibile
        if self.auto_mode:
            print(f"🤖 MODALITÀ AUTOMATICA ATTIVA")
            
            # Priorità: 1) LLM se disponibile e affidabile, 2) clustering label
            if llm_suggestion and llm_confidence > 0.7:
                final_label = llm_suggestion
                human_confidence = llm_confidence
                print(f"✅ Usata proposta LLM: '{final_label}' (confidenza: {llm_confidence:.3f})")
            else:
                final_label = suggested_label
                human_confidence = 0.8  # Confidenza di default per clustering
                print(f"✅ Usata etichetta clustering: '{final_label}' (confidenza di default: {human_confidence:.3f})")
        else:
            # Modalità interattiva originale
            print(f"   [1] ✅ Approva etichetta clustering: '{suggested_label}'")
            
            if llm_suggestion and llm_suggestion != suggested_label:
                print(f"   [2] 🤖 Usa proposta LLM: '{llm_suggestion}'")
            
            print(f"   [3] 📝 Inserisci nuova etichetta")
            print(f"   [4] ⏭️  Salta questo cluster (sarà etichettato automaticamente)")
            
            while True:
                try:
                    choice = input(f"\nScelta [1-4]: ").strip()
                    
                    if choice == "1":
                        # Approva etichetta clustering
                        final_label = suggested_label
                        human_confidence = self._get_confidence_input()
                        print(f"✅ Approvata etichetta: '{final_label}'")
                        break
                        
                    elif choice == "2" and llm_suggestion:
                        # Usa proposta LLM
                        final_label = llm_suggestion
                        human_confidence = self._get_confidence_input()
                        print(f"🤖 Usata proposta LLM: '{final_label}'")
                        break
                        
                    elif choice == "3":
                        # Nuova etichetta
                        while True:
                            new_label = input("Inserisci nuova etichetta: ").strip()
                            if new_label:
                                final_label = new_label
                                human_confidence = self._get_confidence_input()
                                print(f"📝 Nuova etichetta: '{final_label}'")
                                break
                            else:
                                print("❌ Etichetta non può essere vuota")
                        
                    elif choice == "4":
                        # Salta
                        final_label = suggested_label  # Usa quella originale
                        human_confidence = 0.5  # Confidenza neutra
                        print(f"⏭️ Saltato, usata etichetta automatica: '{final_label}'")
                        break
                        
                    else:
                        print("❌ Scelta non valida. Usa 1, 2, 3 o 4.")
                        
                except KeyboardInterrupt:
                    print(f"\n🛑 Interrotto dall'utente. Uso etichetta automatica: '{suggested_label}'")
                    final_label = suggested_label
                    human_confidence = 0.5
                    break
        
        # Registra feedback
        feedback_entry = {
            'cluster_id': cluster_id,
            'suggested_label': suggested_label,
            'llm_suggestion': llm_suggestion,
            'llm_confidence': llm_confidence,
            'final_label': final_label,
            'human_confidence': human_confidence,
            'timestamp': datetime.now().isoformat(),
            'auto_mode': self.auto_mode
        }
        self.human_feedback.append(feedback_entry)
        self.approved_labels[cluster_id] = final_label
        
        return final_label, human_confidence
    
    def handle_altro_classification(self, conversation_text: str, force_human_decision: bool = False) -> Tuple[str, float, Dict[str, Any]]:
        """
        Gestisce una classificazione "ALTRO" con validazione incrociata LLM + BERTopic
        
        Args:
            conversation_text: Testo della conversazione classificata come "altro"
            force_human_decision: Se True, forza la decisione umana
            
        Returns:
            Tuple (etichetta_finale, confidenza, info_validazione)
        """
        if not self.altro_validator or not self.llm_classifier:
            # Fallback: restituisci "altro" senza validazione
            return "altro", 0.3, {"validation_path": "no_validator", "needs_human_review": True}
        
        try:
            print(f"\n" + "🔍" * 80)
            print(f"🏷️  VALERIO - VALIDAZIONE CLASSIFICAZIONE 'ALTRO'")
            print("🔍" * 80)
            
            # Esegui validazione con LLM + BERTopic + Similarità
            validation_result = self.altro_validator.validate_altro_classification(
                conversation_text=conversation_text,
                llm_classifier=self.llm_classifier,
                bertopic_model=self.bertopic_model,
                force_human_decision=force_human_decision
            )
            
            print(f"\n📊 RISULTATO VALIDAZIONE:")
            print(f"   Path: {validation_result.validation_path}")
            print(f"   Tag finale: '{validation_result.final_tag}'")
            print(f"   Confidenza: {validation_result.confidence:.3f}")
            
            if validation_result.similarity_score is not None:
                print(f"   Similarità: {validation_result.similarity_score:.3f}")
                if validation_result.matched_existing_tag:
                    print(f"   Tag simile trovato: '{validation_result.matched_existing_tag}'")
            
            if validation_result.bertopic_suggestion:
                print(f"   Suggerimento BERTopic: '{validation_result.bertopic_suggestion}'")
            
            # Se dobbiamo aggiungere nuovo tag, fallo immediatamente
            if validation_result.should_add_new_tag:
                print(f"\n➕ AGGIUNGENDO NUOVO TAG: '{validation_result.final_tag}'")
                
                success = self.altro_validator.add_new_tag_immediately(
                    validation_result.final_tag,
                    validation_result.confidence
                )
                
                if success:
                    print(f"✅ Nuovo tag '{validation_result.final_tag}' aggiunto e disponibile")
                else:
                    print(f"❌ Errore nell'aggiunta del tag '{validation_result.final_tag}'")
                    # Fallback ad "altro" se non riusciamo ad aggiungere il tag
                    validation_result.final_tag = "altro"
                    validation_result.confidence = 0.3
            
            # Se necessita review umana, mostra dettagli
            if validation_result.needs_human_review and not self.auto_mode:
                print(f"\n👤 REVIEW UMANA NECESSARIA")
                print(f"   Motivo: {validation_result.validation_path}")
                
                if validation_result.bertopic_suggestion:
                    print(f"   BERTopic suggerisce: '{validation_result.bertopic_suggestion}'")
                
                if validation_result.llm_raw_response:
                    print(f"   LLM raw response (primi 200 char):")
                    print(f"   {validation_result.llm_raw_response[:200]}...")
                
                # Chiedi decisione umana
                final_tag = self._get_human_decision_for_altro(validation_result)
                validation_result.final_tag = final_tag
                validation_result.confidence = 0.8  # Alta confidenza per decisione umana
            
            # Restituisci risultato
            validation_info = {
                "validation_path": validation_result.validation_path,
                "similarity_score": validation_result.similarity_score,
                "matched_existing_tag": validation_result.matched_existing_tag,
                "bertopic_suggestion": validation_result.bertopic_suggestion,
                "should_add_new_tag": validation_result.should_add_new_tag,
                "needs_human_review": validation_result.needs_human_review
            }
            
            return validation_result.final_tag, validation_result.confidence, validation_info
            
        except Exception as e:
            print(f"❌ Errore durante validazione ALTRO: {e}")
            return "altro", 0.2, {"validation_path": "error", "error": str(e), "needs_human_review": True}
    
    def _get_human_decision_for_altro(self, validation_result) -> str:
        """
        Chiede decisione umana per un caso "altro" complesso
        
        Args:
            validation_result: Risultato della validazione
            
        Returns:
            Tag deciso dall'umano
        """
        print(f"\n👤 DECISIONE UMANA RICHIESTA:")
        print(f"   [1] 🏷️  Mantieni 'altro'")
        
        option_counter = 2
        
        if validation_result.bertopic_suggestion and validation_result.bertopic_suggestion != "sconosciuto":
            print(f"   [{option_counter}] 🤖 Usa suggerimento BERTopic: '{validation_result.bertopic_suggestion}'")
            bertopic_option = option_counter
            option_counter += 1
        else:
            bertopic_option = None
        
        if validation_result.matched_existing_tag:
            print(f"   [{option_counter}] 🎯 Usa tag simile esistente: '{validation_result.matched_existing_tag}'")
            similar_option = option_counter
            option_counter += 1
        else:
            similar_option = None
        
        print(f"   [{option_counter}] 📝 Inserisci nuovo tag personalizzato")
        custom_option = option_counter
        
        while True:
            try:
                choice = input(f"\nScelta [1-{option_counter}]: ").strip()
                choice_int = int(choice)
                
                if choice_int == 1:
                    return "altro"
                elif bertopic_option and choice_int == bertopic_option:
                    return validation_result.bertopic_suggestion
                elif similar_option and choice_int == similar_option:
                    return validation_result.matched_existing_tag
                elif choice_int == custom_option:
                    while True:
                        new_tag = input("Inserisci nuovo tag: ").strip().lower()
                        if new_tag and new_tag != "altro":
                            # Aggiungi immediatamente il nuovo tag
                            if self.altro_validator.add_new_tag_immediately(new_tag, 0.9):
                                print(f"✅ Nuovo tag '{new_tag}' aggiunto")
                                return new_tag
                            else:
                                print(f"❌ Errore nell'aggiunta. Uso 'altro'")
                                return "altro"
                        else:
                            print("❌ Tag non valido")
                else:
                    print(f"❌ Scelta non valida. Usa un numero da 1 a {option_counter}")
                    
            except ValueError:
                print("❌ Inserisci un numero valido")
            except KeyboardInterrupt:
                print(f"\n🛑 Interrotto. Uso 'altro'")
                return "altro"
    
    def _get_confidence_input(self) -> float:
        """
        Chiede all'utente il livello di confidenza (o usa default in modalità automatica)
        
        Returns:
            Confidenza da 0.0 a 1.0
        """
        if self.auto_mode:
            confidence = 0.8  # Confidenza di default in modalità automatica
            print(f"🤖 Confidenza automatica: {confidence:.1f}")
            return confidence
            
        while True:
            try:
                conf_input = input("Confidenza [1-5, dove 5=molto sicuro]: ").strip()
                confidence_level = int(conf_input)
                
                if 1 <= confidence_level <= 5:
                    # Converte 1-5 in 0.2-1.0
                    confidence = confidence_level * 0.2
                    return confidence
                else:
                    print("❌ Inserisci un numero da 1 a 5")
                    
            except ValueError:
                print("❌ Inserisci un numero valido da 1 a 5")
            except KeyboardInterrupt:
                print("\n🛑 Uso confidenza predefinita: 0.6")
                return 0.6
                return 0.6
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Restituisce un riassunto del feedback umano raccolto
        
        Returns:
            Dizionario con statistiche del feedback
        """
        if not self.human_feedback:
            return {'total_reviews': 0}
        
        total_reviews = len(self.human_feedback)
        choices_count = {}
        avg_confidence = 0.0
        label_changes = 0
        
        for feedback in self.human_feedback:
            choice = feedback.get('choice', 'unknown')
            choices_count[choice] = choices_count.get(choice, 0) + 1
            avg_confidence += feedback.get('human_confidence', 0.5)
            
            if feedback['suggested_label'] != feedback['final_label']:
                label_changes += 1
        
        avg_confidence /= total_reviews
        
        return {
            'total_reviews': total_reviews,
            'choices_distribution': choices_count,
            'average_confidence': avg_confidence,
            'label_changes': label_changes,
            'change_rate': label_changes / total_reviews if total_reviews > 0 else 0
        }
    
    def should_continue_review(self, clusters_remaining: int) -> bool:
        """
        Chiede se continuare con il review o passare al training automatico
        (In modalità automatica continua sempre fino alla fine)
        
        Args:
            clusters_remaining: Numero di cluster rimanenti
            
        Returns:
            True se continuare, False se interrompere
        """
        if clusters_remaining <= 0:
            return False
        
        print(f"\n" + "="*60)
        print(f"📊 PROGRESS: {len(self.human_feedback)} cluster reviewati")
        print(f"🔢 Rimangono {clusters_remaining} cluster da revieware")
        
        # Mostra statistiche attuali
        summary = self.get_feedback_summary()
        if summary['total_reviews'] > 0:
            print(f"📈 Tasso di cambiamento etichette: {summary['change_rate']:.1%}")
            print(f"🎯 Confidenza media: {summary['average_confidence']:.2f}")
        
        # In modalità automatica, continua sempre
        if self.auto_mode:
            print(f"🤖 MODALITÀ AUTOMATICA: continuando automaticamente...")
            return True
        
        print(f"\n👤 OPZIONI:")
        print(f"   [1] ✅ Continua review")
        print(f"   [2] 🚀 Stop e procedi con training automatico")
        print(f"   [3] 🛑 Interrompi completamente")
        
        while True:
            try:
                choice = input(f"\nScelta [1-3]: ").strip()
                
                if choice == "1":
                    return True
                elif choice == "2":
                    print("🚀 Procedendo con training automatico...")
                    return False
                elif choice == "3":
                    print("🛑 Processo interrotto dall'utente")
                    return False
                else:
                    print("❌ Scelta non valida. Usa 1, 2 o 3.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Processo interrotto dall'utente")
                return False
