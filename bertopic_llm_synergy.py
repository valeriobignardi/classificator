"""
Come BERTopic può migliorare le performance degli LLM
Author: Valerio Bignardi
Date: 2025-08-21
"""

# ===============================
# 1. CONTEXT ENRICHMENT
# ===============================

def llm_with_bertopic_context():
    """BERTopic fornisce contesto semantico all'LLM"""
    
    input_text = "devo fare una risonanza urgente"
    
    # BERTopic analizza e fornisce contesto
    bertopic_analysis = {
        "dominant_topic": "esami_diagnostici", 
        "topic_probability": 0.87,
        "related_topics": ["prenotazioni", "urgenze", "diagnostica"],
        "topic_keywords": ["risonanza", "tac", "ecografia", "esame", "imaging"]
    }
    
    # Prompt arricchito per LLM
    enriched_prompt = f"""
    <CONTESTO SEMANTICO>
    Questo testo appartiene principalmente al topic: {bertopic_analysis['dominant_topic']} 
    (confidenza: {bertopic_analysis['topic_probability']})
    Topic correlati: {bertopic_analysis['related_topics']}
    Keywords chiave: {bertopic_analysis['topic_keywords']}
    </CONTESTO SEMANTICO>
    
    Classifica: "{input_text}"
    
    ETICHETTE: prenotazione_esami | emergenze | info_esami | altro
    """
    
    # ✅ VANTAGGI:
    # - LLM ha contesto semantico esplicito
    # - Riduce ambiguità interpretativa  
    # - Migliora consistenza classificazione
    # - Guida l'attenzione dell'LLM su aspetti rilevanti

# ===============================
# 2. UNCERTAINTY REDUCTION  
# ===============================

def bertopic_confidence_boost():
    """BERTopic riduce incertezza LLM in casi ambigui"""
    
    ambiguous_text = "ho una domanda sul pagamento"
    
    # LLM da solo potrebbe essere incerto
    llm_alone_response = {
        "label": "amministrativo",
        "confidence": 0.65,  # BASSA confidenza - ambiguo
        "uncertainty": "potrebbe essere fatturazione o amministrativo"
    }
    
    # BERTopic fornisce disambiguazione semantica
    bertopic_insight = {
        "topic_fatturazione": 0.82,    # ALTA probabilità 
        "topic_amministrativo": 0.23,  # BASSA probabilità
        "topic_keywords": ["costo", "prezzo", "fattura", "pagare"]
    }
    
    # Decisione combinata LLM + BERTopic
    combined_decision = {
        "label": "fatturazione",  # BERTopic disambigua!
        "confidence": 0.85,       # ALTA confidenza combinata
        "reasoning": "BERTopic indica forte associazione con topic fatturazione"
    }
    
    return "BERTopic risolve ambiguità semantic che confondono LLM"

# ===============================
# 3. PROMPT OPTIMIZATION
# ===============================

def adaptive_prompting():
    """BERTopic ottimizza il prompt in base al topic dominante"""
    
    def get_topic_specific_prompt(text, dominant_topic):
        base_prompt = "Classifica questa conversazione Humanitas:"
        
        topic_specific_prompts = {
            "prenotazioni": f"""
            {base_prompt}
            
            FOCUS: Questo testo riguarda PRENOTAZIONI/APPUNTAMENTI.
            Considera: urgenza, tipo esame, disponibilità, modifiche.
            Distingui tra: prenotazione_esami | cancellazione_prenotazione | cambio_orario
            """,
            
            "esami_medici": f"""
            {base_prompt}
            
            FOCUS: Questo testo riguarda ESAMI/PRESTAZIONI MEDICHE.  
            Considera: tipo esame, preparazione, risultati, procedure.
            Distingui tra: info_esami_prestazioni | prenotazione_esami | referti
            """,
            
            "amministrativo": f"""
            {base_prompt}
            
            FOCUS: Questo testo riguarda aspetti AMMINISTRATIVI.
            Considera: documenti, procedure, autorizzazioni, burocrazia.
            Distingui tra: amministrativo | fatturazione | accesso_portale
            """
        }
        
        return topic_specific_prompts.get(dominant_topic, base_prompt)
    
    # ✅ VANTAGGI:
    # - Prompt ottimizzato per contesto specifico
    # - LLM più focalizzato su distinzioni rilevanti  
    # - Migliore accuratezza su edge cases
    # - Riduzione hallucinations

# ===============================
# 4. ENSEMBLE DECISION MAKING
# ===============================

def bertopic_llm_ensemble():
    """BERTopic e LLM si correggono a vicenda"""
    
    def make_ensemble_decision(text):
        # LLM classification
        llm_result = {
            "label": "altro",
            "confidence": 0.55,
            "reasoning": "testo poco chiaro"
        }
        
        # BERTopic analysis  
        bertopic_result = {
            "dominant_topic": "convenzioni_assicurative",
            "probability": 0.89,
            "keywords": ["assicurazione", "convenzione", "rimborso"]
        }
        
        # Decision logic
        if bertopic_result["probability"] > 0.8 and llm_result["confidence"] < 0.7:
            # BERTopic ha alta confidenza, LLM è incerto
            # Usa topic mapping per suggerire label corretta
            topic_to_label = {
                "convenzioni_assicurative": "convenzioni_viaggio",
                "prenotazioni_urgenti": "prenotazione_esami", 
                "fatturazione_costi": "fatturazione"
            }
            
            suggested_label = topic_to_label.get(bertopic_result["dominant_topic"])
            
            return {
                "final_label": suggested_label,
                "confidence": 0.85,
                "method": "BERTopic-guided correction",
                "reasoning": f"LLM incerto, ma BERTopic molto sicuro su topic {bertopic_result['dominant_topic']}"
            }
        
        return llm_result
    
    # ✅ VANTAGGI:
    # - Correzione reciproca LLM ↔ BERTopic
    # - Maggiore robustezza decisionale
    # - Exploitation di punti di forza complementari

# ===============================
# 5. ZERO-SHOT LEARNING ENHANCEMENT
# ===============================

def bertopic_zero_shot_boost():
    """BERTopic aiuta LLM su nuove etichette mai viste"""
    
    # Scenario: nuova etichetta "telemedicina" mai vista nel training
    new_text = "vorrei fare una visita online con il dottore"
    
    # BERTopic trova topic correlati nel dataset esistente
    related_patterns = {
        "topic_consultazioni": 0.76,
        "topic_visite_mediche": 0.68, 
        "topic_tecnologia_digitale": 0.45,
        "similar_texts": [
            "appuntamento con specialista",
            "visita cardiologica", 
            "consultazione medica"
        ]
    }
    
    # LLM con contesto BERTopic
    enhanced_prompt = f"""
    CONTESTO: BERTopic ha identificato pattern simili a "telemedicina":
    - Topic consultazioni (76% match)
    - Topic visite mediche (68% match) 
    - Testi simili nel dataset: {related_patterns['similar_texts']}
    
    Considerando questi pattern, classifica: "{new_text}"
    """
    
    # ✅ RISULTATO: LLM può inferire meglio la nuova categoria
    # basandosi su pattern semantici scoperti da BERTopic

# ===============================
# IMPLEMENTAZIONE PRATICA
# ===============================

class BERTopicLLMSynergy:
    """Classe che combina BERTopic e LLM per performance superiori"""
    
    def __init__(self, bertopic_model, llm_classifier):
        self.bertopic = bertopic_model
        self.llm = llm_classifier
        
    def enhanced_classify(self, text):
        """Classificazione potenziata BERTopic + LLM"""
        
        # 1. Analisi BERTopic
        topics = self.bertopic.transform([text])
        dominant_topic = self._get_dominant_topic(topics)
        
        # 2. Prompt enhancement  
        enhanced_prompt = self._create_topic_aware_prompt(text, dominant_topic)
        
        # 3. LLM classification con contesto
        llm_result = self.llm.classify(enhanced_prompt)
        
        # 4. Ensemble decision
        final_result = self._ensemble_decide(llm_result, topics)
        
        return final_result
        
    def _get_dominant_topic(self, topics):
        """Estrae topic dominante da risultati BERTopic"""
        # Logic per identificare topic principale
        pass
        
    def _create_topic_aware_prompt(self, text, topic_info):
        """Crea prompt arricchito con informazioni topic"""
        # Logic per prompt enhancement
        pass
        
    def _ensemble_decide(self, llm_result, bertopic_topics):  
        """Decisione finale combinando LLM e BERTopic"""
        # Logic per ensemble decision
        pass

print("BERTopic può significativamente migliorare performance LLM!")
print("Combinazione sinergica: LLM (comprensione globale) + BERTopic (pattern locali)")
