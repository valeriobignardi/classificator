#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
PROMPT TEMPLATE INITIALIZER - SETUP PROMPT HUMANITAS
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-24
Descrizione: Script per inizializzare i prompt template obbligatori 
             per il tenant Humanitas
=====================================================================
"""

import sys
import os

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from Utils.prompt_manager import PromptManager
import json
from datetime import datetime

def initialize_humanitas_prompts():
    """
    Inizializza i prompt template obbligatori per il tenant Humanitas
    """
    print("🔧 Inizializzazione prompt template per tenant Humanitas...")
    
    # Inizializza PromptManager
    pm = PromptManager(config_path='config.yaml')
    
    if not pm.connect():
        print("❌ Errore: Impossibile connettersi al database")
        return False
    
    try:
        # PROMPT SYSTEM per intelligent_classifier_system
        system_prompt_content = """Sei un classificatore esperto per l'ospedale Humanitas con profonda conoscenza del dominio sanitario.

MISSIONE: Classifica conversazioni con pazienti/utenti in base al loro intento principale.

APPROCCIO ANALITICO:
1. Identifica l'intento principale (non dettagli secondari)
2. Considera il contesto ospedaliero (prenotazioni, referti, informazioni)
3. Distingui richieste operative da richieste informative
4. Se incerto tra 2 etichette, scegli la più specifica

CONFIDENCE GUIDELINES:
- 0.9-1.0: Intento chiarissimo e inequivocabile
- 0.7-0.8: Intento probabile con piccole ambiguità
- 0.5-0.6: Intento possibile ma con dubbi ragionevoli  
- 0.3-0.4: Molto incerto, probabilmente "altro"
- 0.0-0.2: Impossibile classificare

ETICHETTE DISPONIBILI:
{available_labels}

{priority_labels}
{conversation_context}

OUTPUT FORMAT (SOLO JSON):
{{"predicted_label": "etichetta_precisa", "confidence": 0.X, "motivation": "ragionamento_breve"}}

CRITICAL: Genera ESCLUSIVAMENTE JSON valido. Zero testo aggiuntivo."""

        system_dynamic_vars = {
            "available_labels": {
                "type": "function",
                "source": "classification",
                "function": "get_available_labels",
                "description": "Lista etichette disponibili dal database"
            },
            "priority_labels": {
                "type": "function", 
                "source": "classification",
                "function": "get_priority_labels_hint",
                "description": "Hint per etichette prioritarie"
            },
            "conversation_context": {
                "type": "variable",
                "source": "runtime",
                "description": "Contesto specifico della conversazione"
            }
        }
        
        # USER TEMPLATE per intelligent_classifier_user  
        user_template_content = """Analizza questo testo seguendo l'approccio degli esempi:
{examples_text}
{context_section}

TESTO DA CLASSIFICARE:
"{processed_text}"

RAGIONA STEP-BY-STEP:
1. Identifica l'intento principale
2. Confronta con gli esempi
3. Scegli l'etichetta più appropriata
4. Valuta la tua certezza

OUTPUT (SOLO JSON):"""

        user_dynamic_vars = {
            "examples_text": {
                "type": "variable",
                "source": "runtime",
                "description": "Esempi dinamici per few-shot learning"
            },
            "context_section": {
                "type": "variable",
                "source": "runtime", 
                "description": "Sezione contesto aggiuntivo opzionale"
            },
            "processed_text": {
                "type": "variable",
                "source": "runtime",
                "description": "Testo della conversazione da classificare"
            }
        }
        
        # Lista prompt da creare
        prompts_to_create = [
            {
                'tenant_id': 'humanitas',
                'tenant_name': 'Humanitas',
                'engine': 'LLM',
                'prompt_type': 'SYSTEM',
                'prompt_name': 'intelligent_classifier_system',
                'prompt_content': system_prompt_content,
                'dynamic_variables': system_dynamic_vars,
                'description': 'Prompt di sistema per il classificatore intelligente Humanitas',
                'config_parameters': {
                    'model_compatibility': ['mistral', 'gpt', 'claude'],
                    'temperature': 0.1,
                    'max_tokens': 150
                }
            },
            {
                'tenant_id': 'humanitas',
                'tenant_name': 'Humanitas', 
                'engine': 'LLM',
                'prompt_type': 'TEMPLATE',
                'prompt_name': 'intelligent_classifier_user',
                'prompt_content': user_template_content,
                'dynamic_variables': user_dynamic_vars,
                'description': 'Template per messaggi utente nel classificatore intelligente Humanitas',
                'config_parameters': {
                    'model_compatibility': ['mistral', 'gpt', 'claude'],
                    'max_examples': 5,
                    'context_aware': True
                }
            }
        ]
        
        cursor = pm.connection.cursor()
        
        for prompt_data in prompts_to_create:
            print(f"📝 Creando prompt: {prompt_data['engine']}/{prompt_data['prompt_type']}/{prompt_data['prompt_name']}")
            
            # Controlla se prompt esiste già
            check_query = """
            SELECT COUNT(*) FROM prompts 
            WHERE tenant_id = %s AND engine = %s AND prompt_type = %s AND prompt_name = %s
            """
            cursor.execute(check_query, (
                prompt_data['tenant_id'],
                prompt_data['engine'],
                prompt_data['prompt_type'],
                prompt_data['prompt_name']
            ))
            
            if cursor.fetchone()[0] > 0:
                print(f"   ⚠️  Prompt già esistente, saltato")
                continue
            
            # Inserisci nuovo prompt
            insert_query = """
            INSERT INTO prompts (
                tenant_id, tenant_name, engine, prompt_type, prompt_name, 
                prompt_content, dynamic_variables, config_parameters,
                description, created_by, version, is_active
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            cursor.execute(insert_query, (
                prompt_data['tenant_id'],
                prompt_data['tenant_name'],
                prompt_data['engine'],
                prompt_data['prompt_type'],
                prompt_data['prompt_name'],
                prompt_data['prompt_content'],
                json.dumps(prompt_data['dynamic_variables']),
                json.dumps(prompt_data['config_parameters']),
                prompt_data['description'],
                'setup_script',
                1,
                1
            ))
            
            print(f"   ✅ Prompt creato con successo")
        
        pm.connection.commit()
        cursor.close()
        
        print(f"✅ Inizializzazione completata! {len(prompts_to_create)} prompt configurati per tenant Humanitas")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante inizializzazione prompt: {e}")
        if pm.connection:
            pm.connection.rollback()
        return False
    
    finally:
        pm.disconnect()

def verify_prompts_configuration():
    """
    Verifica che tutti i prompt obbligatori siano configurati
    """
    print("\n🔍 Verifica configurazione prompt...")
    
    pm = PromptManager(config_path='config.yaml')
    
    if not pm.connect():
        print("❌ Errore: Impossibile connettersi al database")
        return False
    
    try:
        cursor = pm.connection.cursor()
        
        # Verifica prompt configurati per Humanitas
        query = """
        SELECT engine, prompt_type, prompt_name, is_active, created_at 
        FROM prompts 
        WHERE tenant_id = 'humanitas' 
        ORDER BY engine, prompt_type, prompt_name
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("❌ Nessun prompt trovato per tenant Humanitas")
            return False
        
        print(f"📋 Prompt configurati per tenant Humanitas ({len(results)} totali):")
        print("-" * 80)
        
        for engine, prompt_type, prompt_name, is_active, created_at in results:
            status = "✅ ATTIVO" if is_active else "❌ INATTIVO"
            print(f"  {engine:12} | {prompt_type:12} | {prompt_name:30} | {status} | {created_at}")
        
        cursor.close()
        
        # Test di caricamento prompt
        print(f"\n🧪 Test caricamento prompt...")
        
        test_prompts = [
            ('LLM', 'SYSTEM', 'intelligent_classifier_system'),
            ('LLM', 'TEMPLATE', 'intelligent_classifier_user')
        ]
        
        for engine, prompt_type, prompt_name in test_prompts:
            try:
                prompt_content = pm.get_prompt(
                    tenant_id='humanitas',
                    engine=engine,
                    prompt_type=prompt_type,
                    prompt_name=prompt_name
                )
                
                if prompt_content:
                    content_length = len(prompt_content)
                    print(f"  ✅ {engine}/{prompt_type}/{prompt_name} - {content_length} caratteri")
                else:
                    print(f"  ❌ {engine}/{prompt_type}/{prompt_name} - NON TROVATO")
                    
            except Exception as e:
                print(f"  ❌ {engine}/{prompt_type}/{prompt_name} - ERRORE: {e}")
        
        print("\n✅ Verifica completata!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante verifica: {e}")
        return False
    
    finally:
        pm.disconnect()

if __name__ == "__main__":
    print("=" * 70)
    print("🏥 SETUP PROMPT TEMPLATE HUMANITAS")
    print("=" * 70)
    
    # Inizializza prompt
    success = initialize_humanitas_prompts()
    
    if success:
        # Verifica configurazione
        verify_prompts_configuration()
        print("\n🎉 Setup completato con successo!")
    else:
        print("\n💥 Setup fallito!")
        sys.exit(1)
