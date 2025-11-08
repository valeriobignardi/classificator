#!/usr/bin/env python3
"""
Utility per caricare config.yaml con sostituzione variabili ambiente
Autore: Valerio Bignardi
Data: 2025-11-08

Carica config.yaml sostituendo le variabili ${VAR_NAME} con i valori da .env
"""

import os
import yaml
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict

def load_env_variables() -> Dict[str, str]:
    """
    Carica variabili da .env
    
    Returns:
        Dizionario con tutte le variabili ambiente
    """
    # Carica .env
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    # Crea dizionario con tutte le variabili ambiente
    env_vars = dict(os.environ)
    
    print(f"✅ Caricate {len(env_vars)} variabili ambiente da .env")
    return env_vars


def substitute_env_vars(value: Any, env_vars: Dict[str, str]) -> Any:
    """
    Sostituisce ricorsivamente ${VAR_NAME} con valori da env_vars
    
    Args:
        value: Valore da processare (str, dict, list, ecc.)
        env_vars: Dizionario variabili ambiente
        
    Returns:
        Valore con variabili sostituite
    """
    if isinstance(value, str):
        # Pattern per trovare ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            env_value = env_vars.get(var_name)
            
            if env_value is None:
                print(f"⚠️  Variabile ${{{var_name}}} non trovata in .env, uso valore default")
                return match.group(0)  # Lascia ${VAR_NAME} invariato
            
            # Converti in tipo appropriato
            if env_value.isdigit():
                return env_value  # Mantieni come stringa se è un numero
            elif env_value.lower() in ('true', 'false'):
                return env_value.lower()
            else:
                return env_value
        
        return re.sub(pattern, replace_var, value)
    
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v, env_vars) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [substitute_env_vars(item, env_vars) for item in value]
    
    else:
        return value


def load_config_with_env(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Carica config.yaml sostituendo variabili ambiente
    
    Args:
        config_path: Path al file config.yaml
        
    Returns:
        Configurazione con variabili ambiente sostituite
        
    Data ultima modifica: 2025-11-08
    """
    # Carica variabili ambiente
    env_vars = load_env_variables()
    
    # Leggi config.yaml
    config_file = Path(__file__).parent / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"File {config_path} non trovato")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Caricato {config_path}")
    
    # Sostituisci variabili ambiente
    config = substitute_env_vars(config, env_vars)
    
    print(f"✅ Variabili ambiente sostituite in config.yaml")
    
    return config


def get_config() -> Dict[str, Any]:
    """
    Funzione helper per ottenere config processato
    
    Returns:
        Configurazione completa con variabili ambiente sostituite
        
    Data ultima modifica: 2025-11-08
    """
    return load_config_with_env()


if __name__ == "__main__":
    # Test caricamento config
    print("=" * 80)
    print("🧪 TEST CARICAMENTO CONFIG CON VARIABILI AMBIENTE")
    print("=" * 80)
    
    config = load_config_with_env()
    
    # Mostra sezioni critiche
    print("\n📋 CONFIGURAZIONI CARICATE:")
    print("\n🗄️  Database:")
    print(f"   Host: {config['database']['host']}")
    print(f"   Database: {config['database']['database']}")
    print(f"   User: {config['database']['user']}")
    print(f"   Password: {'*' * len(config['database']['password'])}")
    
    print("\n🗄️  Tag Database:")
    print(f"   Host: {config['tag_database']['host']}")
    print(f"   Database: {config['tag_database']['database']}")
    print(f"   User: {config['tag_database']['user']}")
    print(f"   Password: {'*' * len(config['tag_database']['password'])}")
    
    print("\n🗄️  MongoDB:")
    print(f"   URL: {config['mongodb']['url']}")
    print(f"   Database: {config['mongodb']['database']}")
    
    print("\n🤖 LLM - Ollama:")
    print(f"   URL: {config['llm']['ollama']['url']}")
    print(f"   Timeout: {config['llm']['ollama']['timeout']}")
    
    print("\n🤖 LLM - OpenAI:")
    print(f"   API Base: {config['llm']['openai']['api_base']}")
    print(f"   Timeout: {config['llm']['openai']['timeout']}")
    print(f"   Max Parallel: {config['llm']['openai']['max_parallel_calls']}")
    
    print("\n🌩️  LLM - Azure OpenAI:")
    print(f"   Endpoint: {config['llm']['azure_openai']['endpoint']}")
    print(f"   API Version: {config['llm']['azure_openai']['api_version']}")
    print(f"   GPT-4o Deployment: {config['llm']['azure_openai']['deployments']['gpt-4o']}")
    print(f"   GPT-5 Deployment: {config['llm']['azure_openai']['deployments']['gpt-5']}")
    
    print("\n" + "=" * 80)
    print("✅ CONFIG CARICATO CORRETTAMENTE!")
    print("=" * 80)
    
    print("\n💡 USO NEL CODICE:")
    print("   from load_config_with_env import get_config")
    print("   config = get_config()")
    print("   db_host = config['database']['host']")
