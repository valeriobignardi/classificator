"""
Gestione dello schema database per la classificazione delle sessioni
"""

import sys
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Aggiunge il percorso per importare il connettore
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MySql'))
from connettore import MySqlConnettore

class ClassificationSchemaManager:
    """
    Gestisce lo schema del database per il sistema di classificazione
    """
    
    def __init__(self, schema: str = 'humanitas'):
        """
        Inizializza il gestore dello schema
        
        Args:
            schema: Nome dello schema del database
        """
        self.schema = schema
        self.connettore = MySqlConnettore()
    
    def create_classification_tables(self) -> bool:
        """
        Crea le tabelle necessarie per la classificazione
        
        Returns:
            True se le tabelle sono state create con successo
        """
        print(f"🏗️  Creazione tabelle di classificazione nello schema '{self.schema}'...")
        
        try:
            # Tabella per i tag/categorie
            create_tags_table = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.session_tags (
                tag_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                tag_name VARCHAR(100) NOT NULL UNIQUE,
                tag_description TEXT,
                tag_color VARCHAR(7) DEFAULT '#2196F3',
                is_active BOOLEAN DEFAULT TRUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_tag_name (tag_name),
                INDEX idx_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Tabella per le classificazioni delle sessioni
            create_classifications_table = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.session_classifications (
                classification_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                session_id VARCHAR(50) NOT NULL,
                tag_id VARCHAR(36) NOT NULL,
                confidence_score FLOAT DEFAULT NULL,
                classification_method ENUM('AUTOMATIC', 'MANUAL', 'HYBRID') DEFAULT 'AUTOMATIC',
                classified_by VARCHAR(100) DEFAULT NULL,
                is_reviewed BOOLEAN DEFAULT FALSE,
                review_notes TEXT DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (tag_id) REFERENCES {self.schema}.session_tags(tag_id) ON DELETE CASCADE,
                UNIQUE KEY unique_session_tag (session_id, tag_id),
                INDEX idx_session_id (session_id),
                INDEX idx_tag_id (tag_id),
                INDEX idx_confidence (confidence_score),
                INDEX idx_method (classification_method),
                INDEX idx_reviewed (is_reviewed),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Tabella per gli embedding delle sessioni (cache)
            create_embeddings_table = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.session_embeddings (
                embedding_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                session_id VARCHAR(50) NOT NULL UNIQUE,
                embedding_vector JSON NOT NULL,
                embedding_model VARCHAR(100) NOT NULL DEFAULT 'LaBSE',
                text_hash VARCHAR(64) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_session_id (session_id),
                INDEX idx_model (embedding_model),
                INDEX idx_hash (text_hash),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Tabella per il training data e feedback
            create_training_table = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.classification_training (
                training_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                session_id VARCHAR(50) NOT NULL,
                session_text TEXT NOT NULL,
                true_tag_id VARCHAR(36) NOT NULL,
                predicted_tag_id VARCHAR(36) DEFAULT NULL,
                confidence_score FLOAT DEFAULT NULL,
                is_correct BOOLEAN DEFAULT NULL,
                feedback_source ENUM('EXPERT', 'AUTOMATIC', 'VALIDATION') DEFAULT 'EXPERT',
                training_set ENUM('TRAIN', 'VALIDATION', 'TEST') DEFAULT 'TRAIN',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (true_tag_id) REFERENCES {self.schema}.session_tags(tag_id),
                FOREIGN KEY (predicted_tag_id) REFERENCES {self.schema}.session_tags(tag_id),
                INDEX idx_session_id (session_id),
                INDEX idx_true_tag (true_tag_id),
                INDEX idx_predicted_tag (predicted_tag_id),
                INDEX idx_training_set (training_set),
                INDEX idx_feedback_source (feedback_source)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Tabella per metriche e monitoring
            create_metrics_table = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.classification_metrics (
                metric_id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                model_version VARCHAR(50) NOT NULL,
                metric_name VARCHAR(50) NOT NULL,
                metric_value FLOAT NOT NULL,
                evaluation_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                evaluation_set ENUM('TRAIN', 'VALIDATION', 'TEST') DEFAULT 'VALIDATION',
                additional_info JSON DEFAULT NULL,
                INDEX idx_model_version (model_version),
                INDEX idx_metric_name (metric_name),
                INDEX idx_evaluation_date (evaluation_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Esegue le query di creazione
            tables = [
                ("session_tags", create_tags_table),
                ("session_classifications", create_classifications_table),
                ("session_embeddings", create_embeddings_table),
                ("classification_training", create_training_table),
                ("classification_metrics", create_metrics_table)
            ]
            
            for table_name, query in tables:
                print(f"  📋 Creazione tabella {table_name}...")
                result = self.connettore.esegui_comando(query)
                if result is not None:
                    print(f"  ✅ Tabella {table_name} creata")
                else:
                    print(f"  ❌ Errore nella creazione di {table_name}")
                    return False
            
            print(f"✅ Tutte le tabelle di classificazione create con successo!")
            return True
            
        except Exception as e:
            print(f"❌ Errore nella creazione delle tabelle: {e}")
            return False
    
    def insert_default_tags(self) -> bool:
        """
        Inserisce i tag predefiniti per Humanitas
        
        Returns:
            True se i tag sono stati inseriti con successo
        """
        # Definizione tag predefiniti con descrizioni dettagliate
        default_tags = [
            ("ritiro_cartella_clinica_referti", "Ritiro documenti", 
             "Richieste di ritiro documentazione medica, cartelle cliniche e referti", "#4CAF50"),
            ("prenotazione_esami", "Prenotazioni", 
             "Prenotazioni e appuntamenti per esami diagnostici, visite specialistiche", "#2196F3"),
            ("info_contatti", "Informazioni contatti", 
             "Richieste di informazioni su contatti, numeri di telefono, reparti", "#FF9800"),
            ("problema_accesso_portale", "Problemi portale", 
             "Problemi tecnici di accesso al portale online, credenziali", "#F44336"),
            ("info_esami", "Info esami", 
             "Informazioni generali su procedure di esami, modalità, tempi", "#9C27B0"),
            ("cambio_anagrafica", "Cambio dati", 
             "Modifiche dati anagrafici, aggiornamento informazioni personali", "#607D8B"),
            ("norme_di_preparazione", "Preparazione esami", 
             "Istruzioni per preparazione esami, digiuno, farmaci", "#795548"),
            ("problema_amministrativo", "Problemi amministrativi", 
             "Questioni amministrative, pagamenti, ticket, rimborsi", "#E91E63"),
            ("info_ricovero", "Info ricovero", 
             "Informazioni su ricoveri, degenze, procedure ospedaliere", "#00BCD4"),
            ("info_certificati", "Certificati", 
             "Richieste di certificati medici, attestazioni, documentazione ufficiale", "#8BC34A"),
            ("info_interventi", "Interventi", 
             "Informazioni su interventi chirurgici, procedure operative", "#FF5722"),
            ("info_parcheggio", "Parcheggio", 
             "Informazioni su parcheggi, costi, modalità di accesso", "#3F51B5"),
            ("altro", "Altro", 
             "Richieste non classificabili nelle categorie principali", "#9E9E9E")
        ]
        
        print(f"📋 Inserimento tag predefiniti...")
        
        try:
            success_count = 0
            for tag_name, tag_desc, tag_full_desc, color in default_tags:
                if self.add_tag_if_not_exists(tag_name, tag_full_desc, color):
                    success_count += 1
                else:
                    print(f"  ❌ Errore con tag '{tag_desc}'")
            
            print(f"✅ Tag predefiniti elaborati: {success_count}/{len(default_tags)} completati!")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Errore nell'inserimento dei tag: {e}")
            return False
        
    def add_tag_if_not_exists(self, tag_name: str, tag_description: str = "", tag_color: str = "#2196F3") -> bool:
        """
        Aggiunge un tag solo se non esiste già (prevenzione duplicati)
        
        Args:
            tag_name: Nome del tag
            tag_description: Descrizione del tag
            tag_color: Colore del tag in formato hex
            
        Returns:
            True se il tag è stato aggiunto o esisteva già, False in caso di errore
        """
        try:
            # Verifica se il tag esiste già (case-insensitive)
            check_query = f"""
            SELECT tag_id, tag_name, tag_description, is_active 
            FROM {self.schema}.session_tags 
            WHERE LOWER(tag_name) = LOWER(%s)
            """
            result = self.connettore.esegui_query(check_query, (tag_name,))
            
            if result and len(result) > 0:
                existing_tag = result[0]
                if existing_tag[3]:  # is_active
                    print(f"  ℹ️ Tag '{tag_name}' già esistente e attivo")
                    return True
                else:
                    # Riattiva tag esistente ma disattivato
                    update_query = f"""
                    UPDATE {self.schema}.session_tags 
                    SET is_active = TRUE, tag_description = %s, tag_color = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE tag_id = %s
                    """
                    update_result = self.connettore.esegui_comando(
                        update_query, (tag_description, tag_color, existing_tag[0])
                    )
                    if update_result:
                        print(f"  ✅ Tag '{tag_name}' riattivato")
                        return True
                    else:
                        print(f"  ❌ Errore riattivazione tag '{tag_name}'")
                        return False
            else:
                # Inserisce nuovo tag
                insert_query = f"""
                INSERT INTO {self.schema}.session_tags 
                (tag_name, tag_description, tag_color) 
                VALUES (%s, %s, %s)
                """
                insert_result = self.connettore.esegui_comando(
                    insert_query, (tag_name, tag_description, tag_color)
                )
                
                if insert_result:
                    print(f"  ✅ Nuovo tag '{tag_name}' aggiunto")
                    return True
                else:
                    print(f"  ❌ Errore inserimento tag '{tag_name}'")
                    return False
                    
        except Exception as e:
            print(f"❌ Errore nell'aggiunta del tag '{tag_name}': {e}")
            return False
    
    def get_all_tags(self) -> List[Dict]:
        """
        Recupera tutti i tag attivi
        
        Returns:
            Lista di dizionari con i dati dei tag
        """
        query = f"""
        SELECT tag_id, tag_name, tag_description, tag_color, created_at
        FROM {self.schema}.session_tags
        WHERE is_active = TRUE
        ORDER BY tag_name
        """
        
        try:
            result = self.connettore.esegui_query(query)
            if result:
                tags = []
                for row in result:
                    tags.append({
                        'tag_id': row[0],
                        'tag_name': row[1],
                        'tag_description': row[2],
                        'tag_color': row[3],
                        'created_at': row[4]
                    })
                return tags
            return []
        except Exception as e:
            print(f"❌ Errore nel recupero dei tag: {e}")
            return []
    
    def get_tag_by_name(self, tag_name: str) -> Optional[Dict]:
        """
        Recupera un tag specifico per nome
        
        Args:
            tag_name: Nome del tag da cercare
            
        Returns:
            Dizionario con i dati del tag o None
        """
        query = f"""
        SELECT tag_id, tag_name, tag_description, tag_color
        FROM {self.schema}.session_tags
        WHERE tag_name = %s AND is_active = TRUE
        """
        
        try:
            result = self.connettore.esegui_query(query, (tag_name,))
            if result:
                row = result[0]
                return {
                    'tag_id': row[0],
                    'tag_name': row[1],
                    'tag_description': row[2],
                    'tag_color': row[3]
                }
            return None
        except Exception as e:
            print(f"❌ Errore nel recupero del tag: {e}")
            return None
    
    def classify_session(self, 
                        session_id: str, 
                        tag_name: str,
                        confidence_score: Optional[float] = None,
                        method: str = 'AUTOMATIC',
                        classified_by: Optional[str] = None) -> bool:
        """
        Classifica una sessione con un tag
        
        Args:
            session_id: ID della sessione
            tag_name: Nome del tag da assegnare
            confidence_score: Score di confidenza (0-1)
            method: Metodo di classificazione
            classified_by: Chi ha fatto la classificazione
            
        Returns:
            True se la classificazione è avvenuta con successo
        """
        try:
            # Recupera il tag_id
            tag = self.get_tag_by_name(tag_name)
            if not tag:
                print(f"❌ Tag '{tag_name}' non trovato")
                return False
            
            # Inserisce o aggiorna la classificazione
            query = f"""
            INSERT INTO {self.schema}.session_classifications 
            (session_id, tag_id, confidence_score, classification_method, classified_by)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            confidence_score = VALUES(confidence_score),
            classification_method = VALUES(classification_method),
            classified_by = VALUES(classified_by),
            updated_at = CURRENT_TIMESTAMP
            """
            
            result = self.connettore.esegui_comando(
                query, (session_id, tag['tag_id'], confidence_score, method, classified_by)
            )
            
            return result is not None
            
        except Exception as e:
            print(f"❌ Errore nella classificazione: {e}")
            return False
    
    def get_classification_statistics(self) -> Dict:
        """
        Recupera statistiche sulle classificazioni
        
        Returns:
            Dizionario con statistiche
        """
        try:
            # Query per statistiche generali
            stats_query = f"""
            SELECT 
                COUNT(*) as total_classifications,
                COUNT(CASE WHEN classification_method = 'AUTOMATIC' THEN 1 END) as automatic,
                COUNT(CASE WHEN classification_method = 'MANUAL' THEN 1 END) as manual,
                COUNT(CASE WHEN is_reviewed = TRUE THEN 1 END) as reviewed,
                AVG(confidence_score) as avg_confidence
            FROM {self.schema}.session_classifications
            """
            
            result = self.connettore.esegui_query(stats_query)
            
            if result:
                row = result[0]
                stats = {
                    'total_classifications': row[0] or 0,
                    'automatic_classifications': row[1] or 0,
                    'manual_classifications': row[2] or 0,
                    'reviewed_classifications': row[3] or 0,
                    'avg_confidence': float(row[4]) if row[4] else 0.0
                }
                
                # Statistiche per tag
                tag_stats_query = f"""
                SELECT st.tag_name, COUNT(*) as count
                FROM {self.schema}.session_classifications sc
                JOIN {self.schema}.session_tags st ON sc.tag_id = st.tag_id
                GROUP BY st.tag_name
                ORDER BY count DESC
                """
                
                tag_result = self.connettore.esegui_query(tag_stats_query)
                stats['tags_distribution'] = []
                
                if tag_result:
                    for tag_row in tag_result:
                        stats['tags_distribution'].append({
                            'tag_name': tag_row[0],
                            'count': tag_row[1]
                        })
                
                return stats
            
            return {}
            
        except Exception as e:
            print(f"❌ Errore nel recupero delle statistiche: {e}")
            return {}
    
    def check_tables_exist(self) -> Dict[str, bool]:
        """
        Verifica quali tabelle di classificazione esistono
        
        Returns:
            Dizionario con nome_tabella -> bool
        """
        tables_to_check = [
            'session_tags',
            'session_classifications', 
            'session_embeddings',
            'classification_training',
            'classification_metrics'
        ]
        
        results = {}
        
        for table in tables_to_check:
            query = f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = %s
            """
            
            try:
                result = self.connettore.esegui_query(query, (self.schema, table))
                results[table] = result[0][0] > 0 if result else False
            except:
                results[table] = False
        
        return results
    
    def chiudi_connessione(self):
        """Chiude la connessione al database"""
        self.connettore.disconnetti()

# Test del schema manager
if __name__ == "__main__":
    print("=== TEST CLASSIFICATION SCHEMA MANAGER ===\n")
    
    schema_manager = ClassificationSchemaManager(schema='humanitas')
    
    try:
        # Verifica tabelle esistenti
        print("🔍 Verifica tabelle esistenti...")
        existing_tables = schema_manager.check_tables_exist()
        
        for table, exists in existing_tables.items():
            status = "✅ Esiste" if exists else "❌ Non esiste"
            print(f"  {table}: {status}")
        
        # Crea tabelle se necessario
        if not all(existing_tables.values()):
            print(f"\n🏗️  Creazione tabelle mancanti...")
            if schema_manager.create_classification_tables():
                print(f"✅ Tabelle create con successo!")
            else:
                print(f"❌ Errore nella creazione delle tabelle")
                exit(1)
        
        # Inserisce tag predefiniti
        print(f"\n🏷️  Inserimento tag predefiniti...")
        if schema_manager.insert_default_tags():
            print(f"✅ Tag inseriti con successo!")
        
        # Mostra tag disponibili
        print(f"\n📋 Tag disponibili:")
        tags = schema_manager.get_all_tags()
        for tag in tags:
            print(f"  🏷️  {tag['tag_name']}: {tag['tag_description']}")
        
        # Test classificazione
        print(f"\n🧪 Test classificazione...")
        test_session_id = "test-session-001"
        if schema_manager.classify_session(
            test_session_id, 
            "accesso_portale", 
            confidence_score=0.85,
            method="MANUAL",
            classified_by="test_user"
        ):
            print(f"✅ Classificazione test inserita")
        
        # Statistiche
        print(f"\n📊 Statistiche classificazioni:")
        stats = schema_manager.get_classification_statistics()
        if stats:
            print(f"  Totale classificazioni: {stats['total_classifications']}")
            print(f"  Automatiche: {stats['automatic_classifications']}")
            print(f"  Manuali: {stats['manual_classifications']}")
            print(f"  Revisionate: {stats['reviewed_classifications']}")
            print(f"  Confidenza media: {stats['avg_confidence']:.3f}")
            
            if stats['tags_distribution']:
                print(f"  Distribuzione per tag:")
                for tag_stat in stats['tags_distribution']:
                    print(f"    {tag_stat['tag_name']}: {tag_stat['count']}")
        
    finally:
        schema_manager.chiudi_connessione()
