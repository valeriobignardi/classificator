#!/usr/bin/env python3
"""
Script di migrazione tools da tabella separata a campo JSON dei prompts

Autore: Sistema Classificazione
Data: 22 Agosto 2025
"""

import mysql.connector
import json
import yaml
from typing import List, Dict, Any

def load_config() -> Dict[str, Any]:
    """Carica configurazione database"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_database_connection(config: Dict[str, Any]):
    """Connessione al database TAG"""
    tag_config = config['tag_database']
    return mysql.connector.connect(
        host=tag_config['host'],
        port=tag_config['port'],
        user=tag_config['user'],
        password=tag_config['password'],
        database=tag_config['database'],
        charset='utf8mb4'
    )

def migrate_tools_to_prompts():
    """Migra i tools dalla tabella tools al campo JSON dei prompts"""
    
    print("🔄 Inizio migrazione tools → prompts")
    
    config = load_config()
    connection = get_database_connection(config)
    cursor = connection.cursor(dictionary=True)
    
    try:
        # 1. Recupera tutti i tools di Humanitas
        print("📋 Recupero tools esistenti...")
        cursor.execute("""
            SELECT tool_name, display_name, description, function_schema 
            FROM tools 
            WHERE tenant_name = 'Humanitas'
            ORDER BY tool_name
        """)
        
        tools_data = cursor.fetchall()
        print(f"✅ Trovati {len(tools_data)} tools da migrare")
        
        # 2. Costruisci array JSON dei tools
        tools_json = []
        for tool in tools_data:
            tool_entry = {
                "tool_name": tool['tool_name'],
                "display_name": tool['display_name'], 
                "description": tool['description'],
                "function_schema": json.loads(tool['function_schema']) if isinstance(tool['function_schema'], str) else tool['function_schema'],
                "is_active": True
            }
            tools_json.append(tool_entry)
            print(f"  • {tool['tool_name']}: {tool['display_name']}")
        
        # 3. Aggiorna il prompt intelligent_classifier_system con i tools
        print("🔄 Aggiorno prompt con tools integrati...")
        cursor.execute("""
            UPDATE prompts 
            SET tools = %s,
                updated_at = CURRENT_TIMESTAMP,
                updated_by = 'migration_script'
            WHERE tenant_name = 'Humanitas' 
            AND prompt_name = 'intelligent_classifier_system'
        """, (json.dumps(tools_json, ensure_ascii=False),))
        
        if cursor.rowcount > 0:
            print("✅ Prompt aggiornato con successo!")
        else:
            print("⚠️ Nessun prompt trovato per l'aggiornamento")
        
        # 4. Verifica migrazione
        print("🔍 Verifica migrazione...")
        cursor.execute("""
            SELECT prompt_name, 
                   JSON_LENGTH(tools) as num_tools,
                   JSON_EXTRACT(tools, '$[*].tool_name') as tool_names
            FROM prompts 
            WHERE tenant_name = 'Humanitas' 
            AND tools IS NOT NULL
        """)
        
        verification = cursor.fetchall()
        for row in verification:
            print(f"  • {row['prompt_name']}: {row['num_tools']} tools → {row['tool_names']}")
        
        connection.commit()
        print("✅ Migrazione completata con successo!")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante migrazione: {e}")
        connection.rollback()
        return False
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    success = migrate_tools_to_prompts()
    if success:
        print("\n🎉 Migrazione tools→prompts completata!")
        print("📝 Prossimi passi:")
        print("   1. Eliminare tabella tools")
        print("   2. Aggiornare PromptManager")
        print("   3. Modificare frontend")
    else:
        print("\n❌ Migrazione fallita!")
