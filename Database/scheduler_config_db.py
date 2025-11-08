import os
import yaml
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error
from config_loader import load_config


def _load_tag_db_config() -> Dict[str, Any]:
    """Carica i parametri di connessione al DB TAG da config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    data = load_config() or {}
    return data.get('tag_database') or {}


class SchedulerConfigDB:
    """Gestione tabella configurazione scheduler per-tenant nel DB TAG.

    Tabella: scheduler_configs
      - tenant_id (PK univoco logico)
      - tenant_slug (per comodità debug)
      - enabled (bool)
      - frequency_unit ('minutes'|'hours'|'days'|'weeks')
      - frequency_value (int)
      - start_at (DATETIME, opzionale)
      - next_run_at (DATETIME, opzionale)
      - last_run_at (DATETIME, opzionale)
      - timezone (string, default 'Europe/Rome')
    """

    def __init__(self) -> None:
        self.db_conf = _load_tag_db_config()
        self.conn: Optional[mysql.connector.MySQLConnection] = None

    def connect(self) -> None:
        if self.conn and getattr(self.conn, 'is_connected', lambda: False)():
            return
        self.conn = mysql.connector.connect(
            host=self.db_conf['host'],
            port=self.db_conf['port'],
            user=self.db_conf['user'],
            password=self.db_conf['password'],
            database=self.db_conf['database']
        )

    def close(self) -> None:
        if self.conn and getattr(self.conn, 'is_connected', lambda: False)():
            self.conn.close()
            self.conn = None

    def ensure_table(self) -> None:
        self.connect()
        ddl = """
        CREATE TABLE IF NOT EXISTS scheduler_configs (
          id INT AUTO_INCREMENT PRIMARY KEY,
          tenant_id VARCHAR(36) NOT NULL,
          tenant_slug VARCHAR(100) NOT NULL,
          enabled BOOLEAN NOT NULL DEFAULT FALSE,
          frequency_unit ENUM('minutes','hours','days','weeks') NOT NULL DEFAULT 'hours',
          frequency_value INT NOT NULL DEFAULT 24,
          start_at DATETIME NULL,
          next_run_at DATETIME NULL,
          last_run_at DATETIME NULL,
          timezone VARCHAR(50) NOT NULL DEFAULT 'Europe/Rome',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uniq_tenant (tenant_id),
          INDEX idx_enabled (enabled),
          INDEX idx_next_run (next_run_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        cur = self.conn.cursor()
        cur.execute(ddl)
        self.conn.commit()
        cur.close()

    @staticmethod
    def _compute_next_run(from_dt: datetime, unit: str, value: int) -> datetime:
        if value <= 0:
            value = 1
        unit = (unit or 'hours').lower()
        if unit == 'minutes':
            return from_dt + timedelta(minutes=value)
        if unit == 'days':
            return from_dt + timedelta(days=value)
        if unit == 'weeks':
            return from_dt + timedelta(weeks=value)
        # default hours
        return from_dt + timedelta(hours=value)

    def get_config(self, tenant_id: str, tenant_slug: str) -> Dict[str, Any]:
        """Restituisce la config del tenant; crea default se mancante (senza abilitare)."""
        self.ensure_table()
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM scheduler_configs WHERE tenant_id=%s", (tenant_id,))
        row = cur.fetchone()
        cur.close()

        if row:
            return row

        # Inserisci default non abilitato
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO scheduler_configs
              (tenant_id, tenant_slug, enabled, frequency_unit, frequency_value, start_at, next_run_at, last_run_at)
            VALUES (%s, %s, FALSE, 'hours', 24, NULL, NULL, NULL)
            ON DUPLICATE KEY UPDATE tenant_slug=VALUES(tenant_slug)
            """,
            (tenant_id, tenant_slug)
        )
        self.conn.commit()
        cur.close()

        return {
            'tenant_id': tenant_id,
            'tenant_slug': tenant_slug,
            'enabled': False,
            'frequency_unit': 'hours',
            'frequency_value': 24,
            'start_at': None,
            'next_run_at': None,
            'last_run_at': None,
            'timezone': 'Europe/Rome'
        }

    def list_configs(self) -> List[Dict[str, Any]]:
        self.ensure_table()
        cur = self.conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM scheduler_configs ORDER BY tenant_slug")
        rows = cur.fetchall() or []
        cur.close()
        return rows

    def upsert_config(
        self,
        tenant_id: str,
        tenant_slug: str,
        enabled: bool,
        frequency_unit: str,
        frequency_value: int,
        start_at_iso: Optional[str]
    ) -> Dict[str, Any]:
        self.ensure_table()
        start_at: Optional[datetime] = None
        if start_at_iso:
            try:
                # Supporta formati "YYYY-MM-DDTHH:MM" (HTML datetime-local) e ISO completo
                start_at = datetime.fromisoformat(start_at_iso.replace('Z', '+00:00').split('+')[0])
            except Exception:
                start_at = None

        # Calcola next_run_at coerente
        now = datetime.now()
        next_run_at: Optional[datetime] = None
        if enabled:
            if start_at and start_at > now:
                next_run_at = start_at
            else:
                next_run_at = self._compute_next_run(now, frequency_unit, int(frequency_value))

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO scheduler_configs
              (tenant_id, tenant_slug, enabled, frequency_unit, frequency_value, start_at, next_run_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              tenant_slug=VALUES(tenant_slug),
              enabled=VALUES(enabled),
              frequency_unit=VALUES(frequency_unit),
              frequency_value=VALUES(frequency_value),
              start_at=VALUES(start_at),
              next_run_at=VALUES(next_run_at)
            """,
            (tenant_id, tenant_slug, bool(enabled), frequency_unit, int(frequency_value), start_at, next_run_at)
        )
        self.conn.commit()
        cur.close()

        return self.get_config(tenant_id, tenant_slug)

    def mark_run_completed(self, tenant_id: str, frequency_unit: str, frequency_value: int) -> None:
        """Aggiorna last_run_at e next_run_at in base alla frequenza."""
        self.ensure_table()
        now = datetime.now()
        next_run = self._compute_next_run(now, frequency_unit, int(frequency_value))
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE scheduler_configs
            SET last_run_at=%s, next_run_at=%s
            WHERE tenant_id=%s
            """,
            (now, next_run, tenant_id)
        )
        self.conn.commit()
        cur.close()

