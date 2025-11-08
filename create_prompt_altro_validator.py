#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-08-28
Descrizione: Script per creare il prompt universale 'prompt_altro_validator' 
             nel database TAG.prompts per il tenant Humanitas
Ultima modifica: 2025-08-28 - Implementazione iniziale
"""

import sys
import os
import yaml
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from config_loader import load_config

def load_config():
    """
    Scopo: Carica configurazione da config.yaml
    Input: Nessuno
    Output: Dict con configurazione
    Ultimo aggiornamento: 2025-08-28
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    try:
    return load_config()
    except Exception as e:
        raise Exception(f"Errore caricamento config: {e}")

def get_tenant_id_for_humanitas(connection):
    """
    Scopo: Recupera tenant_id per il tenant 'humanitas'
    Input: connection (connessione MySQL)
    Output: tenant_id string o None se non trovato
    Ultimo aggiornamento: 2025-08-28
    """
    try:
        cursor = connection.cursor()
        
        # Cerca tenant 'humanitas' (case insensitive)
        query = """
            SELECT tenant_id, tenant_name, tenant_slug 
            FROM tenants 
            WHERE LOWER(tenant_name) = 'humanitas' OR LOWER(tenant_slug) = 'humanitas'
            AND is_active = 1
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        cursor.close()
        
        if result:
            tenant_id, tenant_name, tenant_slug = result
            print(f"✅ Trovato tenant: {tenant_name} (ID: {tenant_id}, slug: {tenant_slug})")
            return tenant_id
        else:
            print("❌ Tenant 'humanitas' non trovato nella tabella tenants")
            return None
            
    except Error as e:
        print(f"❌ Errore ricerca tenant: {e}")
        return None

def create_prompt_altro_validator():
    """
    Scopo: Crea il prompt universale 'prompt_altro_validator' nel database
    Input: Nessuno  
    Output: True se creato con successo
    Ultimo aggiornamento: 2025-08-28
    """
    try:
        # Carica configurazione
        config = load_config()
        tag_db_config = config['tag_database']
        
        print("🔧 Connessione al database TAG...")
        
        # Connessione al database TAG
        connection = mysql.connector.connect(
            host=tag_db_config['host'],
            port=tag_db_config['port'], 
            user=tag_db_config['user'],
            password=tag_db_config['password'],
            database=tag_db_config['database']
        )
        
        print("✅ Connesso al database TAG")
        
        # Recupera tenant_id per Humanitas
        tenant_id = get_tenant_id_for_humanitas(connection)
        if not tenant_id:
            print("❌ Impossibile procedere senza tenant_id")
            return False
        
        cursor = connection.cursor()
        
        # Definisce il prompt universale
        prompt_content = """Sei un esperto classificatore di conversazioni sanitarie. Il tuo compito è validare e decidere l'etichetta finale per conversazioni classificate come "ALTRO".

**CONTESTO**: 
- Durante il training supervisionato, alcune conversazioni vengono etichettate come "ALTRO"
- I sistemi LLM e BERTopic hanno proposto dei tag per questa conversazione
- Tu devi decidere l'etichetta finale tra quelle disponibili o confermare se è davvero un caso "ALTRO"

**TAG DISPONIBILI NEL SISTEMA**:
{{LISTA_TAG}}

**ISTRUZIONI**:
1. Se ricevi 1 tag: LLM e BERTopic concordano su questo tag → VALIDA se è appropriato
2. Se ricevi 2 tag: LLM e BERTopic sono in disaccordo → SCEGLI il migliore o proponi alternativa

**FORMATO RISPOSTA** (JSON rigoroso):
```json
{
  "etichetta_finale": "tag_scelto",
  "confidenza": 0.85,
  "motivazione": "Spiegazione dettagliata della scelta",
  "categoria": "validazione_concordanza" | "risoluzione_disaccordo" | "nuovo_tag",
  "tag_analizzati": ["tag1", "tag2"]
}
```

**REGOLE**:
- Usa SOLO tag dalla lista {{LISTA_TAG}} se appropriati
- Se nessun tag è adatto, usa "altro" e spiega perché
- Confidenza alta (>0.8) per scelte ovvie, media (0.5-0.8) per casi dubbi
- Spiega sempre il ragionamento nella motivazione
- Se proponi un nuovo tag, categoria deve essere "nuovo_tag"

**CONVERSAZIONE DA CLASSIFICARE**:"""
        
        # Query per inserire il prompt
        insert_query = """
            INSERT INTO prompts (
                tenant_id, tenant_name, engine, prompt_type, prompt_name, 
                prompt_content, dynamic_variables, description, is_active, 
                created_by, created_at, updated_by, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                prompt_content = VALUES(prompt_content),
                dynamic_variables = VALUES(dynamic_variables),
                description = VALUES(description),
                updated_by = VALUES(updated_by),
                updated_at = VALUES(updated_at)
        """
        
        # Variabili dinamiche del prompt
        dynamic_variables = '["LISTA_TAG"]'
        description = "Prompt universale per validazione tag ALTRO con LLM. Gestisce sia concordanza che disaccordo tra LLM e BERTopic."
        current_time = datetime.now()
        
        # Parametri per l'inserimento
        prompt_params = (
            tenant_id,                    # tenant_id
            'Humanitas',                  # tenant_name
            'LLM',                        # engine  
            'SYSTEM',                     # prompt_type
            'prompt_altro_validator',     # prompt_name
            prompt_content,               # prompt_content
            dynamic_variables,            # dynamic_variables
            description,                  # description
            1,                           # is_active
            'Valerio Bignardi',          # created_by
            current_time,                # created_at
            'Valerio Bignardi',          # updated_by
            current_time                 # updated_at
        )
        
        print("🔧 Inserimento prompt nel database...")
        cursor.execute(insert_query, prompt_params)
        connection.commit()
        
        print("✅ Prompt 'prompt_altro_validator' creato con successo!")
        print(f"   - Tenant ID: {tenant_id}")
        print(f"   - Engine: LLM")
        print(f"   - Type: SYSTEM")
        print(f"   - Variables: {dynamic_variables}")
        print(f"   - Description: {description}")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Error as e:
        print(f"❌ Errore database: {e}")
        return False
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("CREAZIONE PROMPT ALTRO VALIDATOR")
    print("=" * 80)
    
    success = create_prompt_altro_validator()
    
    if success:
        print("\n🎉 SUCCESSO! Il prompt è stato creato e salvato nel database.")
        print("📋 Ora il sistema può utilizzare 'prompt_altro_validator' per validazioni LLM.")
    else:
        print("\n❌ ERRORE! Impossibile creare il prompt. Verifica la configurazione del database.")
    
    print("=" * 80)
