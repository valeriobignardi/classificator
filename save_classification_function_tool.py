#!/usr/bin/env python3
"""
Script per salvare il Function Tool di classificazione per tutti i tenant

Autore: Valerio Bignardi
Data: 2025-08-30

Scopo:
    Salva il nuovo Function Tool 'classify_conversation' nella tabella TAG.tools
    per tutti i tenant attivi presenti nel database

Funzionalità:
    - Recupera tutti i tenant attivi da TAG.tenants
    - Crea/aggiorna il Function Tool per ogni tenant
    - Valida lo schema JSON del Function Tool
    - Gestisce duplicati e aggiornamenti

Ultimo aggiornamento: 2025-08-30
"""

import os
import sys
import json
import yaml
import logging
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Any
from datetime import datetime
from config_loader import load_config

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """
    Carica la configurazione del database dal file config.yaml
    
    Returns:
        Dict con configurazione database
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        config = load_config()
        return config.get('tag_database', {})
    except Exception as e:
        logger.error(f"❌ Errore caricamento configurazione: {e}")
        raise

def get_database_connection(db_config: Dict[str, Any]):
    """
    Crea una connessione al database MySQL
    
    Args:
        db_config: Configurazione database
        
    Returns:
        Connessione MySQL
    """
    try:
        connection = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            user=db_config.get('user', 'root'),
            password=db_config.get('password'),
            database=db_config.get('database', 'TAG'),
            charset='utf8mb4',
            autocommit=True
        )
        logger.info("✅ Connessione database stabilita")
        return connection
    except Error as e:
        logger.error(f"❌ Errore connessione database: {e}")
        raise

def get_all_active_tenants(connection) -> List[Dict[str, str]]:
    """
    Recupera tutti i tenant attivi dal database
    
    Args:
        connection: Connessione MySQL
        
    Returns:
        Lista di dizionari con informazioni tenant
    """
    try:
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT tenant_id, tenant_name, tenant_slug
        FROM tenants 
        WHERE is_active = TRUE
        ORDER BY tenant_name ASC
        """
        
        cursor.execute(query)
        tenants = cursor.fetchall()
        cursor.close()
        
        logger.info(f"📊 Trovati {len(tenants)} tenant attivi")
        return tenants
        
    except Error as e:
        logger.error(f"❌ Errore recupero tenant: {e}")
        return []

def get_classification_function_schema() -> Dict[str, Any]:
    """
    Definisce lo schema del Function Tool per la classificazione delle conversazioni
    
    Returns:
        Schema JSON per il Function Tool di classificazione
        
    Ultimo aggiornamento: 2025-08-30 - Implementazione Function Tools Ollama
    """
    return {
        "type": "function",
        "function": {
            "name": "classify_conversation",
            "description": "Classifica una conversazione in base al suo contenuto e restituisce etichetta, confidenza e motivazione",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicted_label": {
                        "type": "string",
                        "enum": [
                            "accesso_problemi",
                            "prenotazione_esami", 
                            "info_esami_specifici",
                            "ritiro_referti",
                            "richiesta_operatore",
                            "fatturazione_problemi",
                            "modifica_appuntamenti",
                            "orari_contatti",
                            "problemi_tecnici",
                            "info_generali",
                            "cortesia",
                            "altro"
                        ],
                        "description": "Etichetta di classificazione predetta per la conversazione"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Livello di confidenza nella classificazione (da 0.0 a 1.0)"
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Breve spiegazione del ragionamento dietro la classificazione"
                    }
                },
                "required": [
                    "predicted_label",
                    "confidence", 
                    "motivation"
                ]
            }
        }
    }

def tool_exists_for_tenant(connection, tool_name: str, tenant_id: str) -> bool:
    """
    Verifica se un tool esiste già per un tenant specifico
    
    Args:
        connection: Connessione MySQL
        tool_name: Nome del tool
        tenant_id: ID del tenant
        
    Returns:
        True se il tool esiste, False altrimenti
    """
    try:
        cursor = connection.cursor()
        
        query = """
        SELECT id FROM tools 
        WHERE tool_name = %s AND tenant_id = %s
        """
        
        cursor.execute(query, (tool_name, tenant_id))
        result = cursor.fetchone()
        cursor.close()
        
        return result is not None
        
    except Error as e:
        logger.error(f"❌ Errore verifica esistenza tool: {e}")
        return False

def create_or_update_tool(connection, tenant: Dict[str, str], force_update: bool = False) -> bool:
    """
    Crea o aggiorna il Function Tool di classificazione per un tenant
    
    Args:
        connection: Connessione MySQL
        tenant: Informazioni del tenant
        force_update: Se True, aggiorna anche se esiste già
        
    Returns:
        True se operazione riuscita, False altrimenti
    """
    tool_name = "classify_conversation"
    display_name = "Classificazione Conversazioni"
    description = "Function Tool per la classificazione automatica delle conversazioni usando Ollama Function Calling"
    
    function_schema = get_classification_function_schema()
    function_schema_json = json.dumps(function_schema, ensure_ascii=False, indent=2)
    
    try:
        cursor = connection.cursor()
        
        # Verifica se il tool esiste già
        tool_exists = tool_exists_for_tenant(connection, tool_name, tenant['tenant_id'])
        
        if tool_exists and not force_update:
            logger.info(f"⏭️  Tool '{tool_name}' già esistente per tenant {tenant['tenant_name']} - skip")
            return True
        
        if tool_exists and force_update:
            # Aggiorna tool esistente
            update_query = """
            UPDATE tools 
            SET display_name = %s, description = %s, function_schema = %s, 
                updated_at = CURRENT_TIMESTAMP, is_active = TRUE
            WHERE tool_name = %s AND tenant_id = %s
            """
            
            cursor.execute(update_query, (
                display_name,
                description,
                function_schema_json,
                tool_name,
                tenant['tenant_id']
            ))
            
            logger.info(f"✅ Tool '{tool_name}' aggiornato per tenant {tenant['tenant_name']}")
            
        else:
            # Crea nuovo tool
            insert_query = """
            INSERT INTO tools (tool_name, display_name, description, function_schema, 
                              is_active, tenant_id, tenant_name, created_at, updated_at)
            VALUES (%s, %s, %s, %s, TRUE, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
            
            cursor.execute(insert_query, (
                tool_name,
                display_name,
                description,
                function_schema_json,
                tenant['tenant_id'],
                tenant['tenant_name']
            ))
            
            logger.info(f"✅ Tool '{tool_name}' creato per tenant {tenant['tenant_name']}")
        
        cursor.close()
        return True
        
    except Error as e:
        logger.error(f"❌ Errore creazione/aggiornamento tool per {tenant['tenant_name']}: {e}")
        return False

def main():
    """
    Funzione principale per salvare il Function Tool per tutti i tenant
    """
    print("🚀 AVVIO SALVATAGGIO FUNCTION TOOL CLASSIFICAZIONE")
    print("=" * 60)
    
    try:
        # Carica configurazione
        print("📋 Caricamento configurazione...")
        db_config = load_config()
        
        # Connessione database
        print("🔌 Connessione al database...")
        connection = get_database_connection(db_config)
        
        # Recupera tenant attivi
        print("👥 Recupero tenant attivi...")
        tenants = get_all_active_tenants(connection)
        
        if not tenants:
            print("⚠️  Nessun tenant attivo trovato")
            return
        
        print(f"📊 Trovati {len(tenants)} tenant attivi:")
        for tenant in tenants:
            print(f"  • {tenant['tenant_name']} ({tenant['tenant_id']})")
        print()
        
        # Conferma utente
        force_update = False
        user_input = input("Vuoi aggiornare i tool esistenti? (s/N): ").strip().lower()
        if user_input in ['s', 'si', 'sì', 'y', 'yes']:
            force_update = True
            print("🔄 Modalità aggiornamento attivata")
        
        # Salvataggio tools
        print("💾 Salvataggio Function Tools...")
        success_count = 0
        
        for tenant in tenants:
            print(f"\n🔧 Processando tenant: {tenant['tenant_name']}")
            
            if create_or_update_tool(connection, tenant, force_update):
                success_count += 1
            
        # Chiusura connessione
        connection.close()
        
        # Risultati finali
        print("\n" + "=" * 60)
        print("📋 RISULTATI OPERAZIONE:")
        print(f"   ✅ Tenant processati con successo: {success_count}")
        print(f"   ❌ Tenant con errori: {len(tenants) - success_count}")
        print(f"   📊 Totale tenant: {len(tenants)}")
        
        if success_count == len(tenants):
            print("\n🎉 OPERAZIONE COMPLETATA CON SUCCESSO!")
            print("   Function Tool 'classify_conversation' salvato per tutti i tenant")
        else:
            print("\n⚠️  OPERAZIONE COMPLETATA CON ALCUNI ERRORI")
            print("   Controlla i log per i dettagli degli errori")
            
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")
        print(f"\n💥 OPERAZIONE FALLITA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
