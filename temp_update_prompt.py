#!/usr/bin/env python3
"""
Script temporaneo per aggiornare il prompt system con le modifiche dell'utente
Autore: GitHub Copilot
Data: 2025-08-24
"""

from Utils.prompt_manager import PromptManager
from datetime import datetime

def update_prompt_with_user_changes():
    """
    Aggiorna il prompt system con le modifiche specificate dall'utente
    """
    
    # Il nuovo prompt content con le modifiche dell'utente
    new_prompt_content = '''Sei un classificatore esperto per l'ospedale {{tenant_name}} con profonda conoscenza del dominio sanitario.

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

LABELING GUIDELINES:
la logica con cui devi creare le etichette è: <tipologia di richiesta>_<ambito a cui si riferisce> 

#LABELING EXAMPLE 
- info_prenotazione
- info_esame
- info_ricovero
- info_prericovero
- parere_clinico
- info_fertility_center 
- problema_accesso_portale
- problema_registrazione_portale
- info_ritiro_referto_cartella_clinica

{{context_guidance}}

OUTPUT FORMAT (SOLO JSON):
{{"predicted_label": "etichetta_precisa", "confidence": 0.X, "motivation": "ragionamento_breve"}}

CRITICAL: Genera ESCLUSIVAMENTE JSON valido. Zero testo aggiuntivo.'''
    
    print("🔄 Aggiornamento prompt system con le modifiche dell'utente...")
    
    pm = PromptManager()
    success = pm.update_prompt(
        tenant_id="humanitas",
        engine="LLM", 
        prompt_type="SYSTEM",
        prompt_name="intelligent_classifier_system",
        new_content=new_prompt_content,
        updated_by="manual_script_user_request"
    )
    
    if success:
        print("✅ Prompt aggiornato con successo!")
        
        # Pulisci cache per forzare reload
        pm.clear_cache()
        print("🧹 Cache pulita")
        
        # Verifica che l'aggiornamento sia andato a buon fine
        pm.connect()
        cursor = pm.connection.cursor()
        cursor.execute('''
            SELECT version, updated_at, LENGTH(prompt_content) as content_length, prompt_content
            FROM prompts 
            WHERE tenant_id = "humanitas" 
            AND engine = "LLM" 
            AND prompt_type = "SYSTEM" 
            AND prompt_name = "intelligent_classifier_system"
            ORDER BY version DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        
        if result:
            version, updated_at, length, content = result
            print(f"📋 Prompt verificato: v{version}, {updated_at}, {length} chars")
            
            if "LABELING GUIDELINES" in content:
                print("✅ Le tue modifiche sono state salvate correttamente!")
            else:
                print("❌ Errore: le modifiche non sono visibili")
                
            if "{{tenant_name}}" in content:
                print("✅ Placeholder {{tenant_name}} presente")
            else:
                print("❌ Placeholder {{tenant_name}} mancante")
                
        cursor.close()
        pm.disconnect()
        
    else:
        print("❌ Errore nell'aggiornamento del prompt")

if __name__ == "__main__":
    update_prompt_with_user_changes()
