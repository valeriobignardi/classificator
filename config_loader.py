"""
Config Loader - Utility centralizzata per caricare config.yaml con variabili ambiente
Autore: Valerio Bignardi
Data: 2025-11-08

UTILIZZO:
    from config_loader import load_config
    
    config = load_config()
    db_host = config['database']['host']

Questa utility:
- Carica automaticamente .env
- Sostituisce ${VAR_NAME} in config.yaml con valori da .env
- Caching per performance (carica config una sola volta)
- Thread-safe

Data ultima modifica: 2025-11-08
"""

import os
import yaml
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Optional
import threading

# Cache globale per config (thread-safe)
_config_cache: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def _load_env_once():
    """Carica .env una sola volta"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)


def _substitute_env_vars(value: Any) -> Any:
    """
    Sostituisce ricorsivamente ${VAR_NAME} con valori da environment
    
    Args:
        value: Valore da processare
        
    Returns:
        Valore con variabili sostituite
    """
    if isinstance(value, str):
        # Pattern per trovare ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            
            if env_value is None:
                # Mantieni valore originale se variabile non trovata
                return match.group(0)
            
            return env_value
        
        result = re.sub(pattern, replace_var, value)
        
        # Converti tipi dopo sostituzione
        if result != value:  # Se c'è stata una sostituzione
            # Prova a convertire in numero
            try:
                # Prova prima int, poi float (più specifico → più generico)
                if '.' in result:
                    return float(result)
                return int(result)
            except (ValueError, AttributeError, TypeError):
                # Se non è un numero, controlla boolean
                if isinstance(result, str):
                    if result.lower() == 'true':
                        return True
                    elif result.lower() == 'false':
                        return False
        
        return result
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    
    else:
        return value


def load_config(force_reload: bool = False, config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Carica config.yaml con sostituzione variabili ambiente
    
    Args:
        force_reload: Se True, ricarica config anche se già in cache
        config_path: Path al file config.yaml (relativo alla root del progetto)
        
    Returns:
        Configurazione completa con variabili ambiente sostituite
        
    Thread-safe: Usa lock per garantire caricamento una sola volta anche in ambiente multi-thread
    
    Data ultima modifica: 2025-11-08
    """
    global _config_cache
    
    # Check cache (fast path senza lock)
    if not force_reload and _config_cache is not None:
        return _config_cache
    
    # Caricamento con lock (slow path)
    with _config_lock:
        # Double-check dopo acquisizione lock
        if not force_reload and _config_cache is not None:
            return _config_cache
        
        # Carica .env
        _load_env_once()
        
        # Trova config.yaml (supporta chiamate da subdirectory)
        config_file = Path(__file__).parent / config_path
        
        if not config_file.exists():
            # Fallback: cerca nella directory corrente
            config_file = Path.cwd() / config_path
            
        if not config_file.exists():
            raise FileNotFoundError(
                f"File {config_path} non trovato né in {Path(__file__).parent} "
                f"né in {Path.cwd()}"
            )
        
        # Leggi e processa config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Sostituisci variabili ambiente
        config = _substitute_env_vars(config)
        
        # Salva in cache
        _config_cache = config
        
        return config


def get_database_config() -> Dict[str, Any]:
    """
    Helper per ottenere solo la configurazione database
    
    Returns:
        Configurazione database con credenziali da .env
    """
    config = load_config()
    return config.get('database', {})


def get_mongodb_config() -> Dict[str, Any]:
    """
    Helper per ottenere solo la configurazione MongoDB
    
    Returns:
        Configurazione MongoDB con credenziali da .env
    """
    config = load_config()
    return config.get('mongodb', {})


def get_llm_config() -> Dict[str, Any]:
    """
    Helper per ottenere solo la configurazione LLM
    
    Returns:
        Configurazione LLM con URL e credenziali da .env
    """
    config = load_config()
    return config.get('llm', {})


def get_tag_database_config() -> Dict[str, Any]:
    """
    Helper per ottenere solo la configurazione tag database
    
    Returns:
        Configurazione tag database con credenziali da .env
    """
    config = load_config()
    return config.get('tag_database', {})


# Backward compatibility: alias per mantenere compatibilità con codice esistente
get_config = load_config


if __name__ == "__main__":
    # Test
    print("🧪 Test config_loader")
    print("=" * 80)
    
    config = load_config()
    
    print("\n✅ Config caricato con successo!")
    print(f"\n📊 Sezioni disponibili: {', '.join(config.keys())}")
    
    print("\n🗄️  Database config:")
    db_config = get_database_config()
    print(f"   Host: {db_config.get('host')}")
    print(f"   Database: {db_config.get('database')}")
    print(f"   User: {db_config.get('user')}")
    
    print("\n🗄️  MongoDB config:")
    mongo_config = get_mongodb_config()
    print(f"   URL: {mongo_config.get('url')}")
    print(f"   Database: {mongo_config.get('database')}")
    
    print("\n🤖 LLM config:")
    llm_config = get_llm_config()
    print(f"   Ollama URL: {llm_config.get('ollama', {}).get('url')}")
    print(f"   Azure Endpoint: {llm_config.get('azure_openai', {}).get('endpoint')}")
    
    print("\n" + "=" * 80)
    print("✅ Tutti i test passati!")
