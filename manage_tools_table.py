#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data creazione: 2025-08-30
Script per gestire la tabella TAG.tools e salvare il Function Tool 
per la classificazione per tutti i tenant attivi

Storia aggiornamenti:
- 2025-08-30: Creazione iniziale con vincolo UNIQUE corretto su (tool_name, tenant_id)
"""

import mysql.connector
import yaml
import json
import sys
import os
from datetime import datetime

def load_config():
    """
    Carica la configurazione dal file config.yaml
    
    Returns:
        dict: Configurazione completa
    """
    try:
        config_path = '/home/ubuntu/classificatore/config.yaml'
    return load_config()
    except Exception as e:
        print(f"❌ Errore caricamento configurazione: {e}")
        sys.exit(1)

def get_database_connection(config):
    """
    Crea connessione al database TAG
    
    Args:
        config: Configurazione database
        
    Returns:
        Connessione MySQL
    """
    try:
        db_config = config['tag_database']
        connection = mysql.connector.connect(**db_config)
        return connection
    except Exception as e:
        print(f"❌ Errore connessione database: {e}")
        sys.exit(1)

def get_active_tenants(cursor):
    """
    Recupera tutti i tenant attivi
    
    Args:
        cursor: Cursore database
        
    Returns:
        List[tuple]: Lista di tenant (tenant_id, tenant_name, tenant_slug)
    """
    try:
        cursor.execute("""
            SELECT tenant_id, tenant_name, tenant_slug 
            FROM tenants 
            WHERE is_active = 1
            ORDER BY tenant_name
        """)
        return cursor.fetchall()
    except Exception as e:
        print(f"❌ Errore recupero tenant: {e}")
        return []

def create_classification_tool_schema():
    """
    Crea lo schema JSON del Function Tool per la classificazione
    
    Returns:
        dict: Schema completo del tool
    """
    return {
        "type": "function",
        "function": {
            "name": "classify_conversation",
            "description": "Classifica una conversazione in base al contenuto e restituisce etichetta, confidenza e motivazione",
            "parameters": {
                "type": "object", 
                "properties": {
                    "predicted_label": {
                        "type": "string",
                        "enum": [
                            "prenotazione_esami",
                            "richiesta_informazioni", 
                            "reclamo_servizio",
                            "prenotazione_visite",
                            "disdetta_appuntamento",
                            "emergenza_sanitaria",
                            "altro"
                        ],
                        "description": "Etichetta predetta per la classificazione della conversazione"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Livello di confidenza della predizione (0.0-1.0)"
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Breve spiegazione del ragionamento dietro la classificazione"
                    }
                },
                "required": ["predicted_label", "confidence", "motivation"]
            }
        }
    }

def save_classification_tool_for_tenant(cursor, tenant_info, schema):
    """
    Salva il Function Tool per un tenant specifico
    
    Args:
        cursor: Cursore database
        tenant_info: Tupla (tenant_id, tenant_name, tenant_slug)
        schema: Schema JSON del tool
    """
    tenant_id, tenant_name, tenant_slug = tenant_info
    
    try:
        # Usa sempre lo stesso tool_name per tutti i tenant (ora supportato dal vincolo composito)
        tool_name = "classify_conversation"
        
        cursor.execute("""
            INSERT INTO tools (
                tool_name, 
                display_name, 
                description, 
                function_schema, 
                is_active, 
                tenant_id, 
                tenant_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                display_name = VALUES(display_name),
                description = VALUES(description),
                function_schema = VALUES(function_schema),
                is_active = VALUES(is_active),
                tenant_name = VALUES(tenant_name),
                updated_at = CURRENT_TIMESTAMP
        """, (
            tool_name,
            f"Classificazione Conversazioni - {tenant_name}",
            f"Function Tool per la classificazione automatica delle conversazioni per il tenant {tenant_name}. Restituisce etichetta predetta, confidenza e motivazione.",
            json.dumps(schema),
            1,  # is_active = True
            tenant_id,
            tenant_name
        ))
        
        print(f"   ✅ Tool salvato per {tenant_name} ({tenant_slug}) - ID: {tenant_id}")
        
    except Exception as e:
        print(f"   ❌ Errore salvataggio per {tenant_name}: {e}")

def main():
    """
    Funzione principale per gestire la tabella TAG.tools
    """
    print("🚀 Gestione tabella TAG.tools - Function Tool Classificazione")
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Vincolo UNIQUE aggiornato: (tool_name, tenant_id)")
    print()
    
    # Carica configurazione
    config = load_config()
    
    # Connessione database
    connection = get_database_connection(config)
    cursor = connection.cursor()
    
    try:
        # Recupera tenant attivi
        print("🔍 Recupero tenant attivi...")
        tenants = get_active_tenants(cursor)
        
        if not tenants:
            print("⚠️ Nessun tenant attivo trovato!")
            return
        
        print(f"📊 Trovati {len(tenants)} tenant attivi:")
        for tenant_id, tenant_name, tenant_slug in tenants:
            print(f"   • {tenant_name} ({tenant_slug}) - ID: {tenant_id}")
        print()
        
        # Crea schema del tool
        print("🛠️ Creazione schema Function Tool...")
        classification_schema = create_classification_tool_schema()
        print("   ✅ Schema creato con:")
        print("   📋 Funzione: classify_conversation")
        print("   🏷️ Parametri: predicted_label, confidence, motivation")
        print("   📝 Etichette supportate: 7 categorie principali")
        print()
        
        # Salva tool per ogni tenant
        print("💾 Salvataggio Function Tool per tutti i tenant...")
        for tenant_info in tenants:
            save_classification_tool_for_tenant(cursor, tenant_info, classification_schema)
        
        # Commit modifiche
        connection.commit()
        
        # Verifica risultati
        print()
        print("🔍 Verifica salvataggio...")
        cursor.execute("SELECT COUNT(*) FROM tools WHERE tool_name = 'classify_conversation'")
        count = cursor.fetchone()[0]
        print(f"✅ {count} tool salvati nella tabella TAG.tools")
        
        # Mostra dettagli
        cursor.execute("""
            SELECT tenant_name, tool_name, is_active, created_at, tenant_id 
            FROM tools 
            WHERE tool_name = 'classify_conversation' 
            ORDER BY tenant_name
        """)
        tools = cursor.fetchall()
        
        print("📋 Dettaglio tool salvati:")
        for tool in tools:
            tenant_name, tool_name, is_active, created_at, tenant_id = tool
            status = "✅ ATTIVO" if is_active else "❌ INATTIVO"
            print(f"   • {tenant_name}: {tool_name} - {status} - {created_at} - ID: {tenant_id}")
        
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        connection.rollback()
        
    finally:
        cursor.close()
        connection.close()
    
    print()
    print("🎉 Gestione tabella TAG.tools completata!")
    print("✨ Ogni tenant ora ha il proprio Function Tool per la classificazione!")

if __name__ == "__main__":
    main()

import sys
import os
from datetime import datetime
import json

# Import delle classi necessarie
sys.path.append(os.path.join(os.path.dirname(__file__), 'TagDatabase'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from tag_database_connector import TagDatabaseConnector
from tenant import Tenant

def main():
    """
    Gestione completa della tabella TAG.tools per Function Tools
    """
    print('🔧 Gestione tabella TAG.tools per Function Tools')
    print('=' * 60)
    
    # Connessione al database
    db = TagDatabaseConnector.create_for_tenant_resolution()
    db.connetti()
    
    try:
        # STEP 1: Svuota la tabella
        print('🗑️ Svuotamento tabella TAG.tools...')
        
        # Prima vediamo quanti record ci sono
        count_query = 'SELECT COUNT(*) FROM TAG.tools'
        count_result = db.esegui_query(count_query)
        record_count = count_result[0][0] if count_result else 0
        
        print(f'   📊 Record attuali: {record_count}')
        
        if record_count > 0:
            # Svuota la tabella
            delete_query = 'DELETE FROM TAG.tools'
            db.esegui_query(delete_query)
            print(f'   ✅ Eliminati {record_count} record')
        else:
            print('   ✅ Tabella già vuota')
        
        # STEP 2: Recupera tutti i tenant
        print('')
        print('🏢 Recupero tutti i tenant...')
        
        tenant_query = '''
            SELECT tenant_id, tenant_name, tenant_slug, is_active 
            FROM TAG.tenants 
            WHERE is_active = 1
            ORDER BY tenant_name
        '''
        
        tenants = db.esegui_query(tenant_query)
        print(f'   📊 Trovati {len(tenants)} tenant attivi')
        
        for tenant_data in tenants:
            tenant_id, tenant_name, tenant_slug, is_active = tenant_data
            print(f'   - {tenant_name} ({tenant_slug}) | ID: {tenant_id}')
        
        # STEP 3: Prepara il Function Tool per la classificazione
        print('')
        print('🛠️ Preparazione Function Tool classificazione...')
        
        classification_tool = {
            'type': 'function',
            'function': {
                'name': 'classify_conversation',
                'description': 'Classifica una conversazione sanitaria in una delle categorie predefinite',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'predicted_label': {
                            'type': 'string',
                            'enum': [
                                'prenotazione_esami',
                                'cancellazione_esami', 
                                'modifica_appuntamenti',
                                'richiesta_referti',
                                'informazioni_mediche',
                                'reclami',
                                'altro'
                            ],
                            'description': 'Etichetta di classificazione predetta'
                        },
                        'confidence': {
                            'type': 'number',
                            'minimum': 0.0,
                            'maximum': 1.0,
                            'description': 'Livello di confidenza della classificazione (0.0-1.0)'
                        },
                        'motivation': {
                            'type': 'string',
                            'description': 'Breve spiegazione del ragionamento per la classificazione'
                        }
                    },
                    'required': ['predicted_label', 'confidence', 'motivation']
                }
            }
        }
        
        tool_json = json.dumps(classification_tool, ensure_ascii=False, indent=2)
        
        print('   ✅ Function Tool preparato:')
        print(f'      📝 Nome: classify_conversation')
        print(f'      📝 Descrizione: Classifica conversazioni sanitarie')
        print(f'      📝 Parametri: predicted_label, confidence, motivation')
        print(f'      📝 Enum labels: 7 categorie sanitarie')
        
        # STEP 4: Inserisci il tool per ogni tenant
        print('')
        print('💾 Inserimento Function Tool per ogni tenant...')
        
        insert_query = '''
            INSERT INTO TAG.tools (
                tenant_id, tool_name, display_name, description, 
                function_schema, is_active, tenant_name, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        '''
        
        created_at = datetime.now()
        inserted_count = 0
        errors = []
        
        for tenant_data in tenants:
            tenant_id, tenant_name, tenant_slug, is_active = tenant_data
            
            try:
                # Inserisci il tool per questo tenant
                db.esegui_query(insert_query, (
                    tenant_id,
                    'classify_conversation',
                    'Classificazione Conversazioni',
                    'Function Tool per classificazione automatica conversazioni sanitarie tramite Ollama/Mistral',
                    tool_json,
                    1,  # is_active = 1 (True)
                    tenant_name,
                    created_at,
                    created_at
                ))
                
                inserted_count += 1
                print(f'   ✅ Tool inserito per {tenant_name} ({tenant_slug})')
                
            except Exception as e:
                error_msg = f'Errore inserimento per {tenant_name}: {e}'
                errors.append(error_msg)
                print(f'   ❌ {error_msg}')
        
        print('')
        print(f'🎉 OPERAZIONE COMPLETATA!')
        print(f'   📊 Tool inseriti: {inserted_count}/{len(tenants)}')
        print(f'   🛠️ Tool: classify_conversation')
        print(f'   📝 Tipo: function_tool')
        print(f'   🎯 Scopo: Classificazione conversazioni con Ollama Function Tools')
        
        if errors:
            print(f'   ⚠️ Errori: {len(errors)}')
            for error in errors:
                print(f'      - {error}')
        
        # STEP 5: Verifica finale
        print('')
        print('🔍 Verifica finale...')
        
        verify_query = '''
            SELECT t.tenant_name, t.tenant_slug, tool.tool_name, 
                   tool.display_name, tool.is_active, tool.created_at
            FROM TAG.tools tool
            JOIN TAG.tenants t ON tool.tenant_id = t.tenant_id
            WHERE tool.tool_name = 'classify_conversation'
            ORDER BY t.tenant_name
        '''
        
        results = db.esegui_query(verify_query)
        
        print(f'   📊 Verificati {len(results)} tool salvati:')
        for result in results:
            tenant_name, tenant_slug, tool_name, display_name, is_active, created_at = result
            status = "attivo" if is_active else "inattivo"
            print(f'   - {tenant_name} ({tenant_slug}): {tool_name} - {display_name} [{status}] - {created_at}')
        
        # STEP 6: Test di lettura del tool
        print('')
        print('🧪 Test lettura Function Tool...')
        
        if results:
            # Prendi il primo tenant per il test
            first_result = results[0]
            tenant_name, tenant_slug, tool_name, _, _, _ = first_result
            
            # Leggi il tool schema
            schema_query = '''
                SELECT function_schema 
                FROM TAG.tools tool
                JOIN TAG.tenants t ON tool.tenant_id = t.tenant_id
                WHERE t.tenant_slug = %s AND tool.tool_name = %s
            '''
            
            schema_result = db.esegui_query(schema_query, (tenant_slug, tool_name))
            if schema_result:
                function_schema = schema_result[0][0]
                # Se è già un dict (MySQL JSON), non fare il parsing
                if isinstance(function_schema, dict):
                    parsed_schema = function_schema
                else:
                    parsed_schema = json.loads(function_schema)
                
                print(f'   ✅ Tool letto per {tenant_name}:')
                print(f'      🔧 Funzione: {parsed_schema["function"]["name"]}')
                print(f'      📝 Enum labels: {len(parsed_schema["function"]["parameters"]["properties"]["predicted_label"]["enum"])} categorie')
                print(f'      🎯 Parametri richiesti: {", ".join(parsed_schema["function"]["parameters"]["required"])}')
            else:
                print(f'   ❌ Impossibile leggere tool per {tenant_name}')
        
        print('')
        print('✅ GESTIONE TABELLA TAG.tools COMPLETATA!')
        print('')
        print('🚀 PROSSIMI PASSI:')
        print('   1. ✅ Function Tool salvato per tutti i tenant')
        print('   2. 🔄 Aggiornare API REST nel server per gestire i tools')
        print('   3. 🎨 Verificare il frontend React per interfaccia tools')
        print('')
        
    except Exception as e:
        print(f'❌ Errore generale: {e}')
        import traceback
from config_loader import load_config
        traceback.print_exc()
    
    finally:
        db.disconnetti()
        print('✅ Connessione database chiusa')

if __name__ == '__main__':
    main()
