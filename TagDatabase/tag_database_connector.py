import yaml
import mysql.connector
from mysql.connector import Error
import os
import sys

# Aggiungi path per importare Tenant
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
from tenant import Tenant

class TagDatabaseConnector:
    """Connettore specifico per il database TAG locale con supporto multi-tenant"""
    
    def __init__(self, tenant: Tenant):
        """
        Inizializza il connettore con supporto multi-tenant
        
        PRINCIPIO UNIVERSALE: Accetta SOLO oggetto Tenant
        
        Args:
            tenant: Oggetto Tenant completo
        """
        self.tenant = tenant
        self.tenant_id = tenant.tenant_id  
        self.tenant_name = tenant.tenant_name
            
        self.config = self._load_config()
        self.connection = None
        
    @staticmethod
    def create_for_tenant_resolution():
        """
        FUNZIONE BOOTSTRAP per risoluzione tenant
        
        Crea connettore senza oggetto Tenant per risolvere tenant dal database.
        Da usare SOLO per bootstrap delle funzioni di risoluzione tenant.
        
        Returns:
            TagDatabaseConnector con tenant fittizio per bootstrap
        """
        # Crea tenant fittizio per bootstrap
        class BootstrapTenant:
            def __init__(self):
                self.tenant_id = "bootstrap"
                self.tenant_name = "Bootstrap"
                self.tenant_slug = "bootstrap"
                
        fake_tenant = BootstrapTenant()
        instance = TagDatabaseConnector.__new__(TagDatabaseConnector)
        instance.tenant = fake_tenant
        instance.tenant_id = fake_tenant.tenant_id
        instance.tenant_name = fake_tenant.tenant_name
        instance.config = instance._load_config()
        instance.connection = None
        return instance
    
    def _load_config(self):
        """Carica i parametri di configurazione dal file config.yaml"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['tag_database']
        except FileNotFoundError:
            # Fallback al config.yaml originale
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['tag_database']
        except yaml.YAMLError as e:
            raise Exception(f"Errore nel parsing del file YAML: {e}")
        except KeyError:
            raise Exception("Configurazione 'tag_database' non trovata nel file config.yaml")
    
    def connetti(self):
        """Stabilisce la connessione al database TAG locale"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            if self.connection.is_connected():
                print("Connessione al database TAG stabilita con successo")
                return True
        except Error as e:
            print(f"Errore durante la connessione al database TAG: {e}")
            return False
    
    def disconnetti(self):
        """Chiude la connessione al database"""
    def get_all_tags(self):
        """
        Recupera tutti i tag attivi dal database TAG.tags per il tenant corrente
        
        Returns:
            Lista di dizionari con informazioni sui tag del tenant
        """
        if not self.connetti():
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT tag_name, tag_description, tenant_id, tenant_name, created_at
            FROM tags
            WHERE tag_name IS NOT NULL AND tenant_id = %s
            ORDER BY tag_name
            """
            cursor.execute(query, (self.tenant_id,))
            result = cursor.fetchall()
            cursor.close()
            
            # Converte il risultato nel formato atteso
            tags = []
            for row in result:
                tags.append({
                    'tag_name': row['tag_name'],
                    'tag_description': row.get('tag_description', ''),
                    'tenant_id': row['tenant_id'],
                    'tenant_name': row['tenant_name'],
                    'tag_color': row.get('tag_color', '#2196F3'),
                    'created_at': row.get('created_at')
                })
            
            print(f"📋 Recuperati {len(tags)} tag dal database TAGS per tenant {self.tenant_id}")
            return tags
            
        except Error as e:
            print(f"❌ Errore nel recupero dei tag: {e}")
            return []
        finally:
            self.disconnetti()
    
    def get_tag_by_name(self, tag_name: str):
        """
        Recupera un tag specifico per nome dal tenant corrente
        
        Args:
            tag_name: Nome del tag da cercare
            
        Returns:
            Dizionario con informazioni del tag o None se non trovato
        """
        if not self.connetti():
            return None
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT tag_name, tag_description, tenant_id, tenant_name, created_at
            FROM tags
            WHERE tag_name = %s AND tenant_id = %s
            """
            cursor.execute(query, (tag_name, self.tenant_id))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                    'tag_name': result['tag_name'],
                    'tag_description': result.get('tag_description', ''),
                    'tenant_id': result['tenant_id'],
                    'tenant_name': result['tenant_name'],
                    'tag_color': result.get('tag_color', '#2196F3'),
                    'created_at': result.get('created_at')
                }
            return None
            
        except Error as e:
            print(f"❌ Errore nel recupero del tag '{tag_name}': {e}")
            return None
        finally:
            self.disconnetti()
    
    def add_tag_if_not_exists(self, tag_name: str, tag_description: str = "", 
                             tag_color: str = "#2196F3") -> bool:
        """
        Aggiunge un tag solo se non esiste già (prevenzione duplicati)
        
        Args:
            tag_name: Nome del tag
            tag_description: Descrizione del tag
            tag_color: Colore del tag
            
        Returns:
            True se aggiunto con successo o già esistente
        """
        if not self.connetti():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Verifica se il tag esiste già per questo tenant (case-insensitive)
            check_query = """
            SELECT id, tag_name
            FROM tags 
            WHERE LOWER(tag_name) = LOWER(%s) AND tenant_id = %s
            """
            cursor.execute(check_query, (tag_name, self.tenant_id))
            result = cursor.fetchone()
            
            if result:
                print(f"  ℹ️ Tag '{tag_name}' già esistente per tenant {self.tenant_id}")
                return True
            else:
                # Inserisce nuovo tag con tenant_id e tenant_name
                insert_query = """
                INSERT INTO tags (tenant_id, tenant_name, tag_name, tag_description, created_at, updated_at) 
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                """
                
                cursor.execute(insert_query, (self.tenant_id, self.tenant_name, tag_name, tag_description))
                self.connection.commit()
                print(f"  ✅ Nuovo tag '{tag_name}' aggiunto per tenant {self.tenant_id}")
                return True
                
        except Error as e:
            print(f"❌ Errore nell'aggiunta del tag '{tag_name}': {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            self.disconnetti()
    
    def get_tags_dictionary(self) -> dict:
        """
        Recupera i tag come dizionario tag_name -> tag_description
        
        Returns:
            Dizionario con mappatura nome -> descrizione
        """
        tags = self.get_all_tags()
        return {tag['tag_name']: tag['tag_description'] for tag in tags}
    
    def validate_tags(self, tag_names: list) -> dict:
        """
        Valida una lista di tag verificando se esistono nel database
        
        Args:
            tag_names: Lista di nomi tag da validare
            
        Returns:
            Dizionario tag_name -> exists (True/False)
        """
        existing_tags = {tag['tag_name'] for tag in self.get_all_tags()}
        return {tag: tag in existing_tags for tag in tag_names}
    
    def disconnetti(self):
        """Chiude la connessione al database"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Connessione al database TAG chiusa")
    
    def esegui_query(self, query, parametri=None):
        """Esegue una query SELECT e restituisce i risultati"""
        if not self.connection or not self.connection.is_connected():
            self.connetti()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parametri)
            risultati = cursor.fetchall()
            cursor.close()
            return risultati
        except Error as e:
            print(f"Errore durante l'esecuzione della query: {e}")
            return None
    
    def esegui_comando(self, comando, parametri=None):
        """Esegue un comando INSERT, UPDATE o DELETE"""
        if not self.connection or not self.connection.is_connected():
            self.connetti()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(comando, parametri)
            self.connection.commit()
            righe_modificate = cursor.rowcount
            cursor.close()
            return righe_modificate
        except Error as e:
            print(f"Errore durante l'esecuzione del comando: {e}")
            self.connection.rollback()
            return None

    def crea_database_e_tabelle(self):
        """Crea il database TAG e le tabelle necessarie"""
        try:
            # Prima connetti senza specificare il database per crearlo
            temp_connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            cursor = temp_connection.cursor()
            
            # Crea il database se non esiste
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
            print(f"✅ Database {self.config['database']} creato/verificato")
            
            cursor.close()
            temp_connection.close()
            
            # Ora connetti al database specifico
            if not self.connetti():
                return False
            
            # Crea tabella tenants (cache locale)
            create_tenants_table = """
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id VARCHAR(36) PRIMARY KEY,
                tenant_name VARCHAR(255) NOT NULL,
                tenant_slug VARCHAR(100) NOT NULL UNIQUE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_slug (tenant_slug),
                INDEX idx_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Crea tabella tags
            create_tags_table = """
            CREATE TABLE IF NOT EXISTS tags (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tag_name VARCHAR(100) NOT NULL,
                tag_description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_tag_name (tag_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Crea tabella session_classifications
            create_session_classifications_table = """
            CREATE TABLE IF NOT EXISTS session_classifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(50) NOT NULL,
                tag_name VARCHAR(100) NOT NULL,
                tenant_name VARCHAR(255) NOT NULL,
                tenant_id VARCHAR(36) NOT NULL,
                confidence_score FLOAT DEFAULT NULL,
                classification_method ENUM('AUTOMATIC', 'MANUAL', 'HYBRID') DEFAULT 'MANUAL',
                classified_by VARCHAR(100) DEFAULT NULL,
                notes TEXT DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                UNIQUE KEY unique_session (session_id),
                INDEX idx_session_id (session_id),
                INDEX idx_tag_name (tag_name),
                INDEX idx_tenant_id (tenant_id),
                INDEX idx_tenant_name (tenant_name),
                INDEX idx_classification_method (classification_method),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Esegue le query di creazione (RIMOSSA session_classifications - ora solo MongoDB)
            tables = [
                ("tenants", create_tenants_table),
                ("tags", create_tags_table)
                # RIMOSSO: ("session_classifications", create_session_classifications_table) - usa MongoDB
            ]
            
            for table_name, query in tables:
                print(f"📋 Creazione tabella {table_name}...")
                result = self.esegui_comando(query)
                if result is not None:
                    print(f"✅ Tabella {table_name} creata/verificata")
                else:
                    print(f"❌ Errore nella creazione di {table_name}")
                    return False
            
            # Inserisce tag predefiniti
            self._inserisci_tag_predefiniti()
            
            return True
            
        except Error as e:
            print(f"❌ Errore nella creazione del database: {e}")
            return False
    
    def _inserisci_tag_predefiniti(self):
        """Inserisce i tag predefiniti"""
        tag_predefiniti = [
            ("accesso_portale", "Problemi di accesso al portale/app Humanitas"),
            ("prenotazione_esami", "Richieste di prenotazione visite ed esami"),
            ("ritiro_referti", "Richieste per ritiro referti e risultati"),
            ("problemi_prenotazione", "Problemi con prenotazioni esistenti"),
            ("fatturazione", "Richieste relative a fatture e pagamenti"),
            ("informazioni_generali", "Richieste di informazioni generiche"),
            ("problemi_tecnici", "Problemi tecnici con app/sito"),
            ("emergenze", "Situazioni urgenti o di emergenza"),
            ("altro", "Altre richieste non categorizzate")
        ]
        
        for tag_name, description in tag_predefiniti:
            # Verifica se il tag esiste già
            check_query = "SELECT COUNT(*) FROM tags WHERE tag_name = %s"
            result = self.esegui_query(check_query, (tag_name,))
            
            if result and result[0][0] == 0:
                # Inserisce il nuovo tag
                insert_query = "INSERT INTO tags (tag_name, tag_description) VALUES (%s, %s)"
                if self.esegui_comando(insert_query, (tag_name, description)):
                    print(f"  ✅ Tag '{tag_name}' inserito")
                else:
                    print(f"  ❌ Errore inserimento tag '{tag_name}'")
    
    def sincronizza_tenants_da_remoto(self):
        """Sincronizza i tenants dal database remoto"""
        print("🔄 Sincronizzazione tenants dal database remoto...")
        
        try:
            # Importa il connettore remoto
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MySql'))
            from connettore import MySqlConnettore
            
            # Connetti al database remoto
            remote_connector = MySqlConnettore()
            
            # Recupera tenants dal database remoto (usando i nomi di campo corretti)
            query_tenants = "SELECT tenant_id, tenant_name, tenant_database, tenant_status FROM common.tenants WHERE tenant_status = 1"
            tenants_remoti = remote_connector.esegui_query(query_tenants)
            
            if not tenants_remoti:
                print("❌ Nessun tenant trovato nel database remoto")
                remote_connector.disconnetti()
                return False
            
            print(f"📊 Trovati {len(tenants_remoti)} tenants remoti")
            
            # Sincronizza nella cache locale
            for tenant in tenants_remoti:
                tenant_id, tenant_name, tenant_database, tenant_status = tenant
                
                # Mappa i campi: tenant_database -> tenant_slug, tenant_status -> is_active
                tenant_slug = tenant_database
                is_active = tenant_status == 1
                
                # Inserisce o aggiorna il tenant locale
                upsert_query = """
                INSERT INTO tenants (tenant_id, tenant_name, tenant_slug, is_active) 
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                tenant_name = VALUES(tenant_name),
                tenant_slug = VALUES(tenant_slug),
                is_active = VALUES(is_active),
                updated_at = CURRENT_TIMESTAMP
                """
                
                if self.esegui_comando(upsert_query, (tenant_id, tenant_name, tenant_slug, is_active)):
                    print(f"  ✅ Tenant '{tenant_name}' sincronizzato")
                else:
                    print(f"  ❌ Errore sincronizzazione tenant '{tenant_name}'")
            
            remote_connector.disconnetti()
            print("✅ Sincronizzazione tenants completata")
            return True
            
        except Exception as e:
            print(f"❌ Errore durante la sincronizzazione: {e}")
            return False
    
    def get_tenant_by_slug(self, tenant_slug):
        """Recupera un tenant per slug"""
        query = "SELECT tenant_id, tenant_name, tenant_slug FROM tenants WHERE tenant_slug = %s AND is_active = 1"
        result = self.esegui_query(query, (tenant_slug,))
        
        if result:
            return {
                'tenant_id': result[0][0],
                'tenant_name': result[0][1],
                'tenant_slug': result[0][2]
            }
        return None
    
    def get_all_tenants(self):
        """Recupera tutti i tenants attivi"""
        query = "SELECT tenant_id, tenant_name, tenant_slug FROM tenants WHERE is_active = 1 ORDER BY tenant_name"
        result = self.esegui_query(query)
        
        if result:
            return [
                {
                    'tenant_id': row[0],
                    'tenant_name': row[1],
                    'tenant_slug': row[2]
                }
                for row in result
            ]
        return []
    
    def classifica_sessione(self, session_id, tag_name, tenant_slug, confidence_score=None, method='MANUAL', classified_by=None, notes=None):
        """Classifica una sessione"""
        # Recupera info tenant
        tenant = self.get_tenant_by_slug(tenant_slug)
        if not tenant:
            print(f"❌ Tenant '{tenant_slug}' non trovato")
            return False
        
        # Inserisce o aggiorna la classificazione
        upsert_query = """
        INSERT INTO session_classifications 
        (session_id, tag_name, tenant_name, tenant_id, confidence_score, classification_method, classified_by, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        tag_name = VALUES(tag_name),
        confidence_score = VALUES(confidence_score),
        classification_method = VALUES(classification_method),
        classified_by = VALUES(classified_by),
        notes = VALUES(notes),
        updated_at = CURRENT_TIMESTAMP
        """
        
        result = self.esegui_comando(upsert_query, (
            session_id, tag_name, tenant['tenant_name'], tenant['tenant_id'],
            confidence_score, method, classified_by, notes
        ))
        
        return result is not None
    
    def inserisci_classificazione(self, session_id, tenant_slug, tag, confidence=None, source='automatic'):
        """
        Inserisce una nuova classificazione di sessione.
        
        Args:
            session_id: ID della sessione
            tenant_slug: Slug del tenant
            tag: Tag di classificazione
            confidence: Livello di confidenza (0.0-1.0)
            source: Sorgente della classificazione ('automatic', 'manual', etc.)
            
        Returns:
            bool: True se l'inserimento è riuscito
        """
        return self.classifica_sessione(
            session_id=session_id,
            tag_name=tag,
            tenant_slug=tenant_slug,
            confidence_score=confidence,
            method=source.upper(),
            classified_by='AI_SYSTEM'
        )

    def get_classificazioni_per_tenant(self, tenant_slug):
        """Recupera tutte le classificazioni per un tenant"""
        query = """
        SELECT session_id, tag_name, confidence_score, classification_method, 
               classified_by, notes, created_at, updated_at
        FROM session_classifications sc
        JOIN tenants t ON sc.tenant_id = t.tenant_id
        WHERE t.tenant_slug = %s
        ORDER BY sc.created_at DESC
        """
        
        result = self.esegui_query(query, (tenant_slug,))
        
        if result:
            return [
                {
                    'session_id': row[0],
                    'tag_name': row[1],
                    'confidence_score': row[2],
                    'classification_method': row[3],
                    'classified_by': row[4],
                    'notes': row[5],
                    'created_at': row[6],
                    'updated_at': row[7]
                }
                for row in result
            ]
        return []
    
    def get_statistiche_classificazioni(self):
        """Recupera statistiche sulle classificazioni"""
        stats = {}
        
        # Totale classificazioni
        total_query = "SELECT COUNT(*) FROM session_classifications"
        result = self.esegui_query(total_query)
        stats['total_classificazioni'] = result[0][0] if result else 0
        
        # Per tenant
        tenant_query = """
        SELECT tenant_name, COUNT(*) as count
        FROM session_classifications
        GROUP BY tenant_name
        ORDER BY count DESC
        """
        result = self.esegui_query(tenant_query)
        stats['per_tenant'] = [{'tenant': row[0], 'count': row[1]} for row in result] if result else []
        
        # Per tag
        tag_query = """
        SELECT tag_name, COUNT(*) as count
        FROM session_classifications
        GROUP BY tag_name
        ORDER BY count DESC
        """
        result = self.esegui_query(tag_query)
        stats['per_tag'] = [{'tag': row[0], 'count': row[1]} for row in result] if result else []
        
        # Per metodo
        method_query = """
        SELECT classification_method, COUNT(*) as count
        FROM session_classifications
        GROUP BY classification_method
        """
        result = self.esegui_query(method_query)
        stats['per_metodo'] = [{'metodo': row[0], 'count': row[1]} for row in result] if result else []
        
        return stats

# Test del connettore
if __name__ == "__main__":
    print("=== TEST TAG DATABASE CONNECTOR ===\n")
    
    connector = TagDatabaseConnector()
    
    try:
        # Crea database e tabelle
        print("🏗️  Creazione database e tabelle...")
        if connector.crea_database_e_tabelle():
            print("✅ Database e tabelle create/verificate con successo!")
        else:
            print("❌ Errore nella creazione del database")
            exit(1)
        
        # Sincronizza tenants
        if connector.sincronizza_tenants_da_remoto():
            print("✅ Tenants sincronizzati!")
        
        # Mostra tenants
        print("\n📋 Tenants disponibili:")
        tenants = connector.get_all_tenants()
        for tenant in tenants:
            print(f"  🏥 {tenant['tenant_name']} ({tenant['tenant_slug']})")
        
        # Test classificazione
        print(f"\n🧪 Test classificazione...")
        if connector.classifica_sessione(
            session_id="test-session-001",
            tag_name="accesso_portale",
            tenant_slug="humanitas",
            confidence_score=0.85,
            method="MANUAL",
            classified_by="test_user",
            notes="Test di classificazione"
        ):
            print("✅ Classificazione test inserita")
        
        # Statistiche
        print(f"\n📊 Statistiche:")
        stats = connector.get_statistiche_classificazioni()
        print(f"  Totale classificazioni: {stats['total_classificazioni']}")
        
        if stats['per_tenant']:
            print("  Per tenant:")
            for item in stats['per_tenant']:
                print(f"    {item['tenant']}: {item['count']}")
        
        if stats['per_tag']:
            print("  Per tag:")
            for item in stats['per_tag']:
                print(f"    {item['tag']}: {item['count']}")
        
    finally:
        connector.disconnetti()
