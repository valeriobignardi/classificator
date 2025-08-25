"""
File: engines_schema_manager.py
Autore: Sistema AI
Data creazione: 2025-08-25
Scopo: Gestione schema database per configurazioni AI (embedding e LLM engines)

Storico modifiche:
- 2025-08-25: Creazione iniziale per gestione engines AI per tenants
"""

import yaml
import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime


class EnginesSchemaManager:
    """
    Gestisce lo schema database per le configurazioni AI engines
    
    Questa classe si occupa di:
    - Creare/aggiornare tabella engines 
    - Gestire configurazioni embedding/LLM per tenant
    - Mantenere storico delle modifiche
    """
    
    def __init__(self):
        """
        Inizializza il manager dello schema engines
        
        Returns:
            None
        """
        self.config = self._load_config()
        self.connection = None
    
    def _load_config(self):
        """
        Carica configurazione database dal config.yaml
        
        Returns:
            dict: Configurazione database TAG
            
        Ultima modifica: 2025-08-25
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['tag_database']
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            raise Exception(f"Errore caricamento configurazione: {e}")
    
    def connetti(self):
        """
        Stabilisce connessione al database TAG
        
        Returns:
            bool: True se connessione riuscita
            
        Ultima modifica: 2025-08-25
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            if self.connection.is_connected():
                print("✅ Connessione al database TAG stabilita")
                return True
        except Error as e:
            print(f"❌ Errore connessione database: {e}")
            return False
    
    def disconnetti(self):
        """
        Chiude connessione al database
        
        Ultima modifica: 2025-08-25
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Connessione database chiusa")
    
    def create_engines_table(self):
        """
        Crea tabella engines per configurazioni AI
        
        Struttura tabella:
        - tenant_id: FK verso tenants
        - tenant_name: Nome tenant (duplicato per performance)
        - tenant_slug: Slug tenant (duplicato per performance)  
        - embedding_engine: Tipo engine embedding
        - llm_engine: Tipo engine LLM
        - embedding_config: JSON configurazione embedding
        - llm_config: JSON configurazione LLM
        
        Returns:
            bool: True se creazione riuscita
            
        Ultima modifica: 2025-08-25
        """
        if not self.connetti():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            create_engines_table = """
            CREATE TABLE IF NOT EXISTS engines (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tenant_id VARCHAR(36) NOT NULL,
                tenant_name VARCHAR(255) NOT NULL,
                tenant_slug VARCHAR(100) NOT NULL,
                embedding_engine VARCHAR(50) DEFAULT 'labse',
                llm_engine VARCHAR(50) DEFAULT 'mistral:7b',
                embedding_config JSON DEFAULT NULL,
                llm_config JSON DEFAULT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                UNIQUE KEY unique_tenant (tenant_id),
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                INDEX idx_tenant_slug (tenant_slug),
                INDEX idx_embedding_engine (embedding_engine),
                INDEX idx_llm_engine (llm_engine),
                INDEX idx_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            cursor.execute(create_engines_table)
            print("✅ Tabella engines creata/verificata con successo")
            
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ Errore creazione tabella engines: {e}")
            return False
        finally:
            self.disconnetti()
    
    def migrate_config_yaml_to_database(self):
        """
        Migra configurazioni esistenti da config.yaml al database
        
        Legge tenant_configs da config.yaml e li salva nella tabella engines
        
        Returns:
            bool: True se migrazione riuscita
            
        Ultima modifica: 2025-08-25
        """
        if not self.connetti():
            return False
        
        try:
            # Carica configurazione completa
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            with open(config_path, 'r') as file:
                full_config = yaml.safe_load(file)
            
            tenant_configs = full_config.get('tenant_configs', {})
            
            if not tenant_configs:
                print("ℹ️ Nessuna configurazione tenant trovata in config.yaml")
                return True
            
            cursor = self.connection.cursor()
            
            # Per ogni tenant in config.yaml
            for tenant_slug, config in tenant_configs.items():
                
                # Recupera info tenant dalla tabella tenants
                cursor.execute("""
                    SELECT tenant_id, tenant_name 
                    FROM tenants 
                    WHERE tenant_slug = %s
                """, (tenant_slug,))
                
                tenant_result = cursor.fetchone()
                if not tenant_result:
                    print(f"⚠️ Tenant '{tenant_slug}' non trovato in tabella tenants")
                    continue
                
                tenant_id, tenant_name = tenant_result
                
                # Estrae configurazioni
                embedding_engine = config.get('embedding_engine', 'labse')
                llm_engine = config.get('llm_engine', 'mistral:7b')
                
                # Inserisce/aggiorna nella tabella engines
                upsert_query = """
                INSERT INTO engines (tenant_id, tenant_name, tenant_slug, embedding_engine, llm_engine)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    tenant_name = VALUES(tenant_name),
                    tenant_slug = VALUES(tenant_slug),
                    embedding_engine = VALUES(embedding_engine),
                    llm_engine = VALUES(llm_engine),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                cursor.execute(upsert_query, (
                    tenant_id, tenant_name, tenant_slug, 
                    embedding_engine, llm_engine
                ))
                
                print(f"✅ Migrata configurazione per tenant '{tenant_name}':")
                print(f"   Embedding: {embedding_engine}")
                print(f"   LLM: {llm_engine}")
            
            self.connection.commit()
            cursor.close()
            
            print("✅ Migrazione configurazioni completata")
            return True
            
        except Exception as e:
            print(f"❌ Errore durante migrazione: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            self.disconnetti()
    
    def populate_default_engines_for_all_tenants(self):
        """
        Popola configurazioni default per tutti i tenants esistenti
        
        Per ogni tenant senza configurazione engines, aggiunge defaults
        
        Returns:
            bool: True se popolamento riuscito
            
        Ultima modifica: 2025-08-25  
        """
        if not self.connetti():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Trova tenants senza configurazione engines
            cursor.execute("""
                SELECT t.tenant_id, t.tenant_name, t.tenant_slug
                FROM tenants t
                LEFT JOIN engines e ON t.tenant_id = e.tenant_id
                WHERE e.tenant_id IS NULL AND t.is_active = TRUE
            """)
            
            tenants_without_config = cursor.fetchall()
            
            if not tenants_without_config:
                print("ℹ️ Tutti i tenants hanno già configurazioni engines")
                return True
            
            print(f"🔧 Configurando {len(tenants_without_config)} tenants...")
            
            # Inserisce configurazioni default
            for tenant_id, tenant_name, tenant_slug in tenants_without_config:
                cursor.execute("""
                    INSERT INTO engines (tenant_id, tenant_name, tenant_slug, embedding_engine, llm_engine)
                    VALUES (%s, %s, %s, %s, %s)
                """, (tenant_id, tenant_name, tenant_slug, 'labse', 'mistral:7b'))
                
                print(f"  ✅ Configurato '{tenant_name}' con defaults")
            
            self.connection.commit()
            cursor.close()
            
            print("✅ Popolamento configurazioni default completato")
            return True
            
        except Exception as e:
            print(f"❌ Errore popolamento defaults: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            self.disconnetti()

# Test e setup automatico
if __name__ == "__main__":
    print("=== ENGINES SCHEMA MANAGER SETUP ===\n")
    
    manager = EnginesSchemaManager()
    
    try:
        print("🏗️ Fase 1: Creazione tabella engines...")
        if not manager.create_engines_table():
            print("❌ Fallita creazione tabella")
            exit(1)
        
        print("\n🔄 Fase 2: Migrazione da config.yaml...")
        if not manager.migrate_config_yaml_to_database():
            print("❌ Fallita migrazione configurazioni")
            exit(1)
        
        print("\n⚙️ Fase 3: Configurazioni default tenants...")
        if not manager.populate_default_engines_for_all_tenants():
            print("❌ Fallito popolamento defaults")
            exit(1)
        
        print("\n✅ SETUP ENGINES COMPLETATO CON SUCCESSO!")
        print("📊 Tabella engines pronta per gestire configurazioni AI")
        
    except Exception as e:
        print(f"❌ Errore durante setup: {e}")
        exit(1)
