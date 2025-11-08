import os
import yaml
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any, Optional
from datetime import datetime

# Utilities and connectors from the project
from Utils.tenant import Tenant
from TagDatabase.tag_database_connector import TagDatabaseConnector


class RemoteTagSyncService:
    """
    Syncs session tags to the remote MySQL per-tenant schema.

    Responsibilities:
    - Connect to remote MySQL using config.yaml:database
    - Select database/schema equal to tenant_slug (create if missing)
    - Ensure tables conversations_tags and ai_session_tags exist
    - Upsert tags into conversations_tags (insert or update description)
    - Upsert session→tag rows into ai_session_tags (insert or update)
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        self.config_path = config_path
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        if 'database' not in self.config:
            raise RuntimeError("Configurazione 'database' mancante in config.yaml")

        self.db_cfg = self.config['database']

    # --------------- Connection helpers ---------------
    def _connect_no_db(self):
        return mysql.connector.connect(
            host=self.db_cfg['host'],
            port=self.db_cfg.get('port', 3306),
            user=self.db_cfg['user'],
            password=self.db_cfg['password'],
            autocommit=True,
        )

    def _connect_schema(self, schema: str):
        return mysql.connector.connect(
            host=self.db_cfg['host'],
            port=self.db_cfg.get('port', 3306),
            user=self.db_cfg['user'],
            password=self.db_cfg['password'],
            database=schema,
            autocommit=True,
        )

    def _ensure_schema(self, tenant_slug: str) -> None:
        """Create the tenant schema if it does not exist."""
        conn = None
        try:
            conn = self._connect_no_db()
            cur = conn.cursor()
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{tenant_slug}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.close()
        finally:
            if conn:
                conn.close()

    # --------------- DDL ---------------
    def _ensure_tables(self, connection) -> None:
        """Create required tables if not present in the current schema."""
        cur = connection.cursor()

        # conversations_tags: tag_id unique, description, system flag, timestamps
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations_tags (
                tag_id VARCHAR(191) NOT NULL,
                tag_description TEXT NULL,
                is_system_tag TINYINT(1) NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (tag_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )

        # ai_session_tags: one tag per session (update on duplicate)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_session_tags (
                session_id VARCHAR(191) NOT NULL,
                tag_id VARCHAR(191) NOT NULL,
                confidence_score FLOAT NULL,
                classification_method VARCHAR(100) NULL,
                classified_by VARCHAR(100) NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id),
                KEY idx_tag_id (tag_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )

        cur.close()

    # --------------- Data helpers ---------------
    @staticmethod
    def _normalize_tag(tag: Optional[str]) -> Optional[str]:
        if not tag:
            return None
        t = tag.strip()
        if not t:
            return None
        return t.upper()

    def _get_local_tag_descriptions(self, tenant: Tenant) -> Dict[str, str]:
        """Return dict TAG_NAME(upper) -> description from local TAG database."""
        try:
            local = TagDatabaseConnector(tenant)
            tag_dict = local.get_tags_dictionary()  # name -> description
            # Normalize keys to UPPER for robust lookups
            return {str(k).upper(): (v or '') for k, v in tag_dict.items()}
        except Exception:
            return {}

    # --------------- Public API ---------------
    def sync_session_tags(self, tenant: Tenant, documenti: List[Any]) -> Dict[str, Any]:
        """
        Upsert tags and session→tag rows for a batch of classified documents.

        Args:
            tenant: Tenant object with tenant_slug/tenant_id
            documenti: List of DocumentoProcessing (or objects exposing same attributes)

        Returns:
            Summary dict with counters and optional error
        """
        # 🚨 LOGGING AGGIUNTO PER DEBUG
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🔍 [{timestamp}] RemoteTagSyncService.sync_session_tags() CHIAMATO")
        print(f"   📋 Tenant: {getattr(tenant, 'tenant_slug', 'N/A')} (ID: {getattr(tenant, 'tenant_id', 'N/A')})")
        print(f"   📊 Documenti ricevuti: {len(documenti) if documenti else 0}")
        
        tenant_slug = getattr(tenant, 'tenant_slug', None) or getattr(tenant, 'tenant_name', None)
        if not tenant_slug:
            error_msg = 'tenant_slug non disponibile'
            print(f"   ❌ ERRORE: {error_msg}")
            return {'success': False, 'error': error_msg}
        
        print(f"   🎯 Target schema: {tenant_slug}")

        # Prepare description lookup from local TAG database
        tag_desc_map = self._get_local_tag_descriptions(tenant)

        # Ensure schema and connect
        try:
            self._ensure_schema(tenant_slug)
            conn = self._connect_schema(tenant_slug)
        except Error as e:
            return {'success': False, 'error': f"Connessione schema remoto fallita: {e}"}

        try:
            self._ensure_tables(conn)
            cur = conn.cursor()

            upsert_tag_sql = (
                "INSERT INTO conversations_tags (tag_id, tag_description, is_system_tag) "
                "VALUES (%s, %s, 1) "
                "ON DUPLICATE KEY UPDATE tag_description=VALUES(tag_description), is_system_tag=VALUES(is_system_tag), updated_at=CURRENT_TIMESTAMP"
            )
            upsert_session_sql = (
                "INSERT INTO ai_session_tags (session_id, tag_id, confidence_score, classification_method, classified_by) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE tag_id=VALUES(tag_id), confidence_score=VALUES(confidence_score), "
                "classification_method=VALUES(classification_method), classified_by=VALUES(classified_by), updated_at=CURRENT_TIMESTAMP"
            )

            tag_inserts = 0
            tag_updates = 0
            session_inserts = 0
            session_updates = 0

            for doc in documenti:
                try:
                    # Extract fields from DocumentoProcessing
                    session_id = getattr(doc, 'session_id', None)
                    if not session_id:
                        continue

                    label = (
                        getattr(doc, 'predicted_label', None)
                        or getattr(doc, 'propagated_label', None)
                        or getattr(doc, 'llm_prediction', None)
                        or getattr(doc, 'ml_prediction', None)
                    )
                    tag_id = self._normalize_tag(label)
                    if not tag_id:
                        continue

                    confidence = (
                        getattr(doc, 'confidence', None)
                        or getattr(doc, 'llm_confidence', None)
                        or getattr(doc, 'ml_confidence', None)
                        or 0.0
                    )
                    classification_method = getattr(doc, 'classification_method', None) or 'unified_pipeline'
                    classified_by = getattr(doc, 'classified_by', None) or 'unified_pipeline'

                    # Resolve description from local map
                    tag_description = tag_desc_map.get(tag_id, '')

                    # Upsert into conversations_tags
                    cur.execute(upsert_tag_sql, (tag_id, tag_description))
                    # mysql-connector does not expose whether it was insert vs update directly.
                    # We can approximate using rowcount: 1 for insert, 2 for update in some versions.
                    if cur.rowcount == 1:
                        tag_inserts += 1
                    elif cur.rowcount == 2:
                        tag_updates += 1

                    # Upsert into ai_session_tags
                    cur.execute(
                        upsert_session_sql,
                        (session_id, tag_id, float(confidence), str(classification_method), str(classified_by)),
                    )
                    if cur.rowcount == 1:
                        session_inserts += 1
                    elif cur.rowcount == 2:
                        session_updates += 1

                except Exception:
                    # Continue with other documents without failing the whole batch
                    continue

            conn.commit()

            result = {
                'success': True,
                'tag_inserts': tag_inserts,
                'tag_updates': tag_updates,
                'session_inserts': session_inserts,
                'session_updates': session_updates,
            }
            
            # 🚨 LOGGING RISULTATO
            print(f"   ✅ SYNC COMPLETATO: {result}")
            return result
        except Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            error_result = {'success': False, 'error': f"Errore durante sync remoto: {e}"}
            print(f"   ❌ SYNC FALLITO: {error_result}")
            return error_result
        finally:
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

