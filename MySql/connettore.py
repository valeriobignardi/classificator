import os
import stat
import tempfile
from typing import Any, Dict, Optional

import mysql.connector
from mysql.connector import Error  # type: ignore

from config_loader import load_config
from Database.tenant_db_connection_service import tenant_db_connection_service

try:
    from sshtunnel import SSHTunnelForwarder  # type: ignore

    SSHTUNNEL_AVAILABLE = True
except ImportError:  # pragma: no cover - dipendenza opzionale
    SSHTUNNEL_AVAILABLE = False


class MySqlConnettore:
    """
    Gestore connessione MySQL con supporto configurazioni per-tenant e SSH tunnel.

    Se viene fornito un oggetto Tenant (o tenant_id/slug), recupera le credenziali
    personalizzate dal database locale (tenant_db_connections). In caso di tunnel SSH
    attivo, crea automaticamente un forward locale verso il database remoto.
    """

    def __init__(
        self,
        tenant: Optional[Any] = None,
        tenant_id: Optional[str] = None,
        tenant_slug: Optional[str] = None,
    ) -> None:
        config = load_config()
        if "database" not in config:
            raise RuntimeError("Sezione 'database' mancante in config.yaml")

        self._base_db_config: Dict[str, Any] = config["database"]
        self.connection: Optional[mysql.connector.MySQLConnection] = None
        self.ssh_tunnel: Optional[SSHTunnelForwarder] = None  # type: ignore
        self._temp_key_file: Optional[str] = None

        self._tenant = tenant
        self._tenant_id = getattr(tenant, "tenant_id", None) if tenant else tenant_id
        self._tenant_slug = (
            getattr(tenant, "tenant_slug", None) if tenant else tenant_slug
        )

        self._dynamic_config: Optional[Dict[str, Any]] = None
        self.database_name: str = self._base_db_config.get("database")

        self._load_dynamic_config()

    # ------------------------------------------------------------------ #
    # Configurazione dinamica
    # ------------------------------------------------------------------ #
    def _load_dynamic_config(self, force_reload: bool = False) -> None:
        if not self._tenant_id:
            self._dynamic_config = None
            self.database_name = self._base_db_config.get("database")
            return

        if force_reload:
            tenant_db_connection_service.invalidate_cache(self._tenant_id)

        try:
            config = tenant_db_connection_service.get_connection_config(
                self._tenant_id, self._tenant_slug
            )
            self._dynamic_config = config or None
            if config and config.get("db_database"):
                self.database_name = config["db_database"]
            else:
                self.database_name = self._base_db_config.get("database")
        except Exception as exc:
            print(f"⚠️ [MySqlConnettore] Errore caricamento configurazione tenant: {exc}")
            self._dynamic_config = None
            self.database_name = self._base_db_config.get("database")

    def refresh_tenant_config(self, force_reload: bool = False) -> None:
        """Ricarica la configurazione dinamica per il tenant associato."""
        self._load_dynamic_config(force_reload=force_reload)

    def get_dynamic_config(self) -> Optional[Dict[str, Any]]:
        """Restituisce la configurazione dinamica (override) corrente, se presente."""
        if self._dynamic_config is None:
            return None
        return dict(self._dynamic_config)

    def get_database_name(self) -> str:
        """Restituisce il nome del database target (considerando override per-tenant)."""
        return self.database_name or self._base_db_config.get("database")

    def _get_effective_db_config(self) -> Dict[str, Any]:
        """Merge tra configurazione base e override per-tenant."""
        base = self._base_db_config
        dynamic = self._dynamic_config or {}

        use_ssh = bool(dynamic.get("use_ssh_tunnel"))

        try:
            db_port = int(dynamic.get("db_port") or base["port"])
        except (ValueError, TypeError):
            db_port = int(base["port"])

        effective = {
            "use_ssh_tunnel": use_ssh,
            "db_host": dynamic.get("db_host") or base["host"],
            "db_port": db_port,
            "db_user": dynamic.get("db_user") or base["user"],
            "db_password": dynamic.get("db_password") or base["password"],
            "db_database": dynamic.get("db_database") or base["database"],
        }

        if use_ssh:
            try:
                ssh_port = int(dynamic.get("ssh_port") or 22)
            except (ValueError, TypeError):
                ssh_port = 22

            effective.update(
                {
                    "ssh_host": dynamic.get("ssh_host"),
                    "ssh_port": ssh_port,
                    "ssh_username": dynamic.get("ssh_username"),
                    "ssh_password": dynamic.get("ssh_password"),
                    "ssh_auth_method": dynamic.get("ssh_auth_method") or "password",
                    "ssh_key": dynamic.get("ssh_key"),
                    "ssh_key_name": dynamic.get("ssh_key_name"),
                    "ssh_key_passphrase": dynamic.get("ssh_key_passphrase"),
                }
            )

        self.database_name = effective["db_database"]
        return effective

    # ------------------------------------------------------------------ #
    # Gestione connessione
    # ------------------------------------------------------------------ #
    def connetti(self) -> bool:
        """Stabilisce (o riusa) la connessione al database MySQL."""
        if self.connection and self.connection.is_connected():
            return True

        # Aggiorna configurazione dinamica (eventuali modifiche recenti)
        self._load_dynamic_config()

        cfg = self._get_effective_db_config()

        target_host = cfg["db_host"]
        target_port = cfg["db_port"]

        try:
            if cfg.get("use_ssh_tunnel"):
                self._ensure_sshtunnel_available()
                self._start_ssh_tunnel(cfg)
                target_host = "127.0.0.1"
                target_port = int(self.ssh_tunnel.local_bind_port)  # type: ignore

            self.connection = mysql.connector.connect(
                host=target_host,
                port=int(target_port),
                user=cfg["db_user"],
                password=cfg["db_password"],
                database=cfg["db_database"],
                #connect_timeout=300,  # 5 minuti per la connessione iniziale
                #connection_timeout=300,  # 5 minuti per mantenere la connessione attiva
            )

            if self.connection.is_connected():
                tenant_info = (
                    f"tenant={self._tenant_slug or self._tenant_id}"
                    if (self._tenant_slug or self._tenant_id)
                    else "default"
                )
                mode = "SSH tunnel" if cfg.get("use_ssh_tunnel") else "direct"
                print(
                    f"🔌 Connessione MySQL ({mode}) stabilita ({tenant_info}) → "
                    f"{target_host}:{target_port}/{cfg['db_database']}"
                )
                return True

            return False

        except Error as db_error:
            print(f"❌ Errore connessione MySQL: {db_error}")
            self._cleanup_tunnel()
            return False
        except Exception as generic_error:
            print(f"❌ Errore generico connessione MySQL: {generic_error}")
            self._cleanup_tunnel()
            return False

    def disconnetti(self) -> None:
        """Chiude connessione e tunnel SSH (se attivo)."""
        if self.connection and self.connection.is_connected():
            try:
                self.connection.close()
                print("🔌 Connessione al database chiusa")
            except Error as exc:
                print(f"⚠️ Errore durante la chiusura connessione MySQL: {exc}")
        self.connection = None
        self._cleanup_tunnel()

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #
    def esegui_query(self, query: str, parametri: Optional[tuple] = None):
        """Esegue una query SELECT e restituisce i risultati."""
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return None

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, parametri)
            risultati = cursor.fetchall()
            cursor.close()
            return risultati
        except Error as exc:
            print(f"❌ Errore esecuzione query MySQL: {exc}")
            return None

    def esegui_comando(self, comando: str, parametri: Optional[tuple] = None):
        """Esegue un comando INSERT/UPDATE/DELETE e restituisce righe modificate."""
        if not self.connection or not self.connection.is_connected():
            if not self.connetti():
                return None

        try:
            cursor = self.connection.cursor()
            cursor.execute(comando, parametri)
            self.connection.commit()
            righe_modificate = cursor.rowcount
            cursor.close()
            return righe_modificate
        except Error as exc:
            print(f"❌ Errore esecuzione comando MySQL: {exc}")
            try:
                if self.connection:
                    self.connection.rollback()
            except Error:
                pass
            return None

    # ------------------------------------------------------------------ #
    # SSH helper methods
    # ------------------------------------------------------------------ #
    def _ensure_sshtunnel_available(self) -> None:
        if not SSHTUNNEL_AVAILABLE:
            raise RuntimeError(
                "Libreria 'sshtunnel' non disponibile. Installare il requisito con 'pip install sshtunnel'."
            )

    def _start_ssh_tunnel(self, cfg: Dict[str, Any]) -> None:
        if self.ssh_tunnel and getattr(self.ssh_tunnel, "is_active", False):
            return

        ssh_host = cfg.get("ssh_host")
        ssh_port = cfg.get("ssh_port", 22)
        ssh_username = cfg.get("ssh_username")

        if not ssh_host or not ssh_username:
            raise ValueError("Parametri SSH mancanti: host e username sono obbligatori")

        tunnel_kwargs: Dict[str, Any] = {}

        auth_method = (cfg.get("ssh_auth_method") or "password").lower()
        ssh_password = cfg.get("ssh_password")
        ssh_key = cfg.get("ssh_key")
        ssh_key_passphrase = cfg.get("ssh_key_passphrase")

        if auth_method in {"password", "both"} and ssh_password:
            tunnel_kwargs["ssh_password"] = ssh_password

        if auth_method in {"key", "both"} and ssh_key:
            key_path = self._create_temp_key_file(ssh_key, cfg.get("ssh_key_name"))
            tunnel_kwargs["ssh_pkey"] = key_path
            if ssh_key_passphrase:
                tunnel_kwargs["ssh_private_key_password"] = ssh_key_passphrase

        if not tunnel_kwargs:
            raise ValueError("Credenziali SSH non valide: specificare password e/o chiave")

        remote_host = cfg["db_host"]
        remote_port = int(cfg["db_port"])

        self.ssh_tunnel = SSHTunnelForwarder(  # type: ignore
            (ssh_host, int(ssh_port)),
            ssh_username=ssh_username,
            remote_bind_address=(remote_host, remote_port),
            local_bind_address=("127.0.0.1", 0),
            **tunnel_kwargs,
        )
        self.ssh_tunnel.start()

        print(
            f"🔐 SSH tunnel attivo {ssh_username}@{ssh_host}:{ssh_port} → "
            f"{remote_host}:{remote_port} (porta locale {self.ssh_tunnel.local_bind_port})"
        )

    def _create_temp_key_file(self, key_content: str, key_name: Optional[str]) -> str:
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, prefix="ssh_key_", suffix="_tmp.pem"
        )
        temp_file.write(key_content)
        temp_file.flush()
        temp_file.close()

        os.chmod(temp_file.name, stat.S_IRUSR | stat.S_IWUSR)
        self._temp_key_file = temp_file.name

        display_name = key_name or os.path.basename(temp_file.name)
        print(f"🗝️  Chiave SSH temporanea creata ({display_name})")
        return temp_file.name

    def _cleanup_tunnel(self) -> None:
        if self.ssh_tunnel:
            try:
                if getattr(self.ssh_tunnel, "is_active", False):
                    self.ssh_tunnel.stop()
                    print("🔐 Tunnel SSH chiuso")
            except Exception as exc:
                print(f"⚠️ Errore nella chiusura del tunnel SSH: {exc}")
            finally:
                self.ssh_tunnel = None

        if self._temp_key_file and os.path.exists(self._temp_key_file):
            try:
                os.remove(self._temp_key_file)
            except OSError:
                pass
            self._temp_key_file = None

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def __del__(self) -> None:  # pragma: no cover - cleanup best effort
        try:
            self.disconnetti()
        except Exception:
            pass
