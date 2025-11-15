"""
Servizio per la gestione delle connessioni database per-tenant con supporto SSH Tunnel.

Tabella: tenant_db_connections (nel database TAG locale)
Campi principali:
  - tenant_id (UUID) / tenant_slug
  - use_ssh_tunnel (bool) + parametri SSH (host, port, username, password/chiave)
  - credenziali database remoto (host, port, database, user, password)

Il servizio fornisce:
  - Cache in memoria per accesso rapido
  - Caricamento all'avvio di tutte le configurazioni
  - Metodi di lettura/salvataggio per API e componenti backend
  - Normalizzazione dei dati (conversione tipi, fallback default)
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import mysql.connector
from mysql.connector import Error  # type: ignore

from config_loader import load_config


class TenantDBConnectionService:
    """Gestisce la tabella tenant_db_connections nel database TAG locale."""

    _instance: Optional["TenantDBConnectionService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._config = load_config().get("tag_database", {})
        if not self._config:
            raise RuntimeError("Configurazione 'tag_database' mancante in config.yaml")

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._table_checked = False
        self._table_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "TenantDBConnectionService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --------------------------------------------------------------------- #
    # Utilities di connessione
    # --------------------------------------------------------------------- #
    def _connect(self) -> mysql.connector.MySQLConnection:
        connection = mysql.connector.connect(
            host=self._config["host"],
            port=int(self._config["port"]),
            user=self._config["user"],
            password=self._config["password"],
            database=self._config["database"],
            autocommit=True,
        )
        return connection

    def _ensure_table(self, connection: mysql.connector.MySQLConnection) -> None:
        if self._table_checked:
            return

        with self._table_lock:
            if self._table_checked:
                return

            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_db_connections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tenant_id VARCHAR(36) NOT NULL,
                    tenant_slug VARCHAR(100) NOT NULL,
                    use_ssh_tunnel TINYINT(1) NOT NULL DEFAULT 0,
                    ssh_host VARCHAR(255) NULL,
                    ssh_port INT DEFAULT 22,
                    ssh_username VARCHAR(255) NULL,
                    ssh_auth_method ENUM('password','key','both') DEFAULT 'password',
                    ssh_password TEXT NULL,
                    ssh_key_name VARCHAR(255) NULL,
                    ssh_key LONGTEXT NULL,
                    ssh_key_passphrase TEXT NULL,
                    db_host VARCHAR(255) NULL,
                    db_port INT DEFAULT 3306,
                    db_database VARCHAR(255) NULL,
                    db_user VARCHAR(255) NULL,
                    db_password TEXT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uniq_tenant (tenant_id),
                    INDEX idx_tenant_slug (tenant_slug),
                    INDEX idx_use_ssh (use_ssh_tunnel)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            cursor.close()
            self._table_checked = True

    # --------------------------------------------------------------------- #
    # Cache helpers
    # --------------------------------------------------------------------- #
    def load_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Carica tutte le configurazioni in cache (da chiamare all'avvio server)."""
        connection = self._connect()
        try:
            self._ensure_table(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM tenant_db_connections")
            results = cursor.fetchall() or []
            cursor.close()

            normalized = {row["tenant_id"]: self._normalize_record(row) for row in results}
            with self._cache_lock:
                self._cache = normalized

            return normalized
        finally:
            connection.close()

    def invalidate_cache(self, tenant_id: Optional[str] = None) -> None:
        """Invalidazione cache totale o per singolo tenant."""
        with self._cache_lock:
            if tenant_id is None:
                self._cache = {}
            else:
                self._cache.pop(tenant_id, None)

    # --------------------------------------------------------------------- #
    # CRUD operations
    # --------------------------------------------------------------------- #
    def get_connection_config(
        self, tenant_id: str, tenant_slug: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Recupera configurazione per tenant (cache first)."""
        with self._cache_lock:
            cached = self._cache.get(tenant_id)

        if cached is not None:
            return dict(cached)

        connection = self._connect()
        try:
            self._ensure_table(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM tenant_db_connections WHERE tenant_id = %s", (tenant_id,)
            )
            row = cursor.fetchone()
            cursor.close()

            if not row and tenant_slug:
                # Se non esiste ancora una configurazione, crea record default
                self._create_default_record(connection, tenant_id, tenant_slug)
                row = {
                    "tenant_id": tenant_id,
                    "tenant_slug": tenant_slug,
                    "use_ssh_tunnel": 0,
                    "ssh_host": None,
                    "ssh_port": 22,
                    "ssh_username": None,
                    "ssh_auth_method": "password",
                    "ssh_password": None,
                    "ssh_key_name": None,
                    "ssh_key": None,
                    "ssh_key_passphrase": None,
                    "db_host": None,
                    "db_port": None,
                    "db_database": None,
                    "db_user": None,
                    "db_password": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }

            if row:
                normalized = self._normalize_record(row)
                with self._cache_lock:
                    self._cache[tenant_id] = dict(normalized)
                return normalized

            return None
        finally:
            connection.close()

    def save_connection_config(
        self,
        tenant_id: str,
        tenant_slug: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Salva/aggiorna configurazione per tenant."""
        connection = self._connect()
        try:
            self._ensure_table(connection)

            normalized_payload = self._prepare_payload(payload)
            query = """
                INSERT INTO tenant_db_connections
                    (tenant_id, tenant_slug, use_ssh_tunnel, ssh_host, ssh_port, ssh_username,
                     ssh_auth_method, ssh_password, ssh_key_name, ssh_key, ssh_key_passphrase,
                     db_host, db_port, db_database, db_user, db_password)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    tenant_slug = VALUES(tenant_slug),
                    use_ssh_tunnel = VALUES(use_ssh_tunnel),
                    ssh_host = VALUES(ssh_host),
                    ssh_port = VALUES(ssh_port),
                    ssh_username = VALUES(ssh_username),
                    ssh_auth_method = VALUES(ssh_auth_method),
                    ssh_password = VALUES(ssh_password),
                    ssh_key_name = VALUES(ssh_key_name),
                    ssh_key = VALUES(ssh_key),
                    ssh_key_passphrase = VALUES(ssh_key_passphrase),
                    db_host = VALUES(db_host),
                    db_port = VALUES(db_port),
                    db_database = VALUES(db_database),
                    db_user = VALUES(db_user),
                    db_password = VALUES(db_password),
                    updated_at = CURRENT_TIMESTAMP
            """

            cursor = connection.cursor()
            cursor.execute(
                query,
                (
                    tenant_id,
                    tenant_slug,
                    normalized_payload["use_ssh_tunnel"],
                    normalized_payload["ssh_host"],
                    normalized_payload["ssh_port"],
                    normalized_payload["ssh_username"],
                    normalized_payload["ssh_auth_method"],
                    normalized_payload["ssh_password"],
                    normalized_payload["ssh_key_name"],
                    normalized_payload["ssh_key"],
                    normalized_payload["ssh_key_passphrase"],
                    normalized_payload["db_host"],
                    normalized_payload["db_port"],
                    normalized_payload["db_database"],
                    normalized_payload["db_user"],
                    normalized_payload["db_password"],
                ),
            )
            connection.commit()
            cursor.close()

            # Ricarica il record aggiornato
            self.invalidate_cache(tenant_id)
            result = self.get_connection_config(tenant_id, tenant_slug)
            if result is None:
                raise RuntimeError("Impossibile ricaricare la configurazione appena salvata")
            return result
        finally:
            connection.close()

    # --------------------------------------------------------------------- #
    # Helper privati
    # --------------------------------------------------------------------- #
    def _create_default_record(
        self,
        connection: mysql.connector.MySQLConnection,
        tenant_id: str,
        tenant_slug: str,
    ) -> None:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT IGNORE INTO tenant_db_connections
                (tenant_id, tenant_slug, use_ssh_tunnel, ssh_port, ssh_auth_method, db_port)
            VALUES (%s, %s, 0, 22, 'password', 3306)
            """,
            (tenant_id, tenant_slug),
        )
        connection.commit()
        cursor.close()

    @staticmethod
    def _normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizza tipi e aggiunge flag utili per il frontend."""
        normalized = dict(row)
        normalized["use_ssh_tunnel"] = bool(row.get("use_ssh_tunnel"))
        normalized["ssh_port"] = _safe_int(row.get("ssh_port"), default=22)
        normalized["db_port"] = _safe_int(row.get("db_port"), default=3306)

        # Flag utili per UI
        normalized["has_ssh_password"] = bool(row.get("ssh_password"))
        normalized["has_ssh_key"] = bool(row.get("ssh_key"))
        normalized["has_db_password"] = bool(row.get("db_password"))
        normalized["has_ssh_key_passphrase"] = bool(row.get("ssh_key_passphrase"))

        # Converti datetime in stringa ISO se necessario
        for key in ("created_at", "updated_at"):
            value = row.get(key)
            if isinstance(value, datetime):
                normalized[key] = value.isoformat()
            elif value is None:
                normalized[key] = None

        return normalized

    @staticmethod
    def _prepare_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara e valida i dati prima del salvataggio."""
        use_ssh = bool(payload.get("use_ssh_tunnel"))

        ssh_port = _safe_int(payload.get("ssh_port"), default=22)
        db_port = _safe_int(payload.get("db_port"), default=3306)

        ssh_auth_method = payload.get("ssh_auth_method") or "password"
        ssh_auth_method = ssh_auth_method.lower()
        if ssh_auth_method not in {"password", "key", "both"}:
            ssh_auth_method = "password"

        ssh_key = payload.get("ssh_key")
        if ssh_key:
            # Normalizza newline per compatibilità cross-platform
            ssh_key = ssh_key.replace("\r\n", "\n")

        normalized = {
            "use_ssh_tunnel": 1 if use_ssh else 0,
            "ssh_host": payload.get("ssh_host") or None,
            "ssh_port": ssh_port,
            "ssh_username": payload.get("ssh_username") or None,
            "ssh_auth_method": ssh_auth_method,
            "ssh_password": payload.get("ssh_password") or None,
            "ssh_key_name": payload.get("ssh_key_name") or None,
            "ssh_key": ssh_key or None,
            "ssh_key_passphrase": payload.get("ssh_key_passphrase") or None,
            "db_host": payload.get("db_host") or None,
            "db_port": db_port,
            "db_database": payload.get("db_database") or None,
            "db_user": payload.get("db_user") or None,
            "db_password": payload.get("db_password") or None,
        }

        # Se tunnel attivo, garantisce credenziali DB obbligatorie
        if use_ssh:
            ssh_missing = [
                field
                for field in ("ssh_host", "ssh_username")
                if not normalized.get(field)
            ]
            if ssh_missing:
                raise ValueError(
                    "Parametri SSH mancanti: " + ", ".join(ssh_missing)
                )

            if ssh_auth_method == "password" and not normalized.get("ssh_password"):
                raise ValueError("Password SSH mancante per autenticazione 'password'")
            if ssh_auth_method == "key" and not normalized.get("ssh_key"):
                raise ValueError("Chiave SSH mancante per autenticazione 'key'")
            if (
                ssh_auth_method == "both"
                and (not normalized.get("ssh_password") or not normalized.get("ssh_key"))
            ):
                raise ValueError(
                    "Autenticazione 'both' richiede sia password che chiave SSH"
                )

            missing_fields = [
                field
                for field in ("db_host", "db_port", "db_database", "db_user", "db_password")
                if not normalized.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    "Credenziali database incomplete quando SSH tunnel è abilitato: "
                    + ", ".join(missing_fields)
                )

        return normalized


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


# Istanza singleton pronta all'uso
tenant_db_connection_service = TenantDBConnectionService.instance()
