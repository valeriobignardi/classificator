#!/usr/bin/env python3
"""
Utilities per la conversione di tipi NumPy in tipi Python nativi per serializzazione JSON.

Questo modulo centralizza la logica di conversione per evitare duplicazioni di codice
e garantire consistenza in tutto il progetto.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Converte ricorsivamente i tipi NumPy in tipi Python nativi per serializzazione JSON.
    
    Questa funzione gestisce:
    - np.integer -> int
    - np.floating -> float  
    - np.str_ e np.bytes_ -> str
    - np.bool_ -> bool
    - np.ndarray -> list
    - dict -> dict con chiavi e valori convertiti
    - list -> list con elementi convertiti
    - tuple -> tuple con elementi convertiti
    
    Args:
        obj: Oggetto da convertire (può essere dict, list, numpy types, etc.)
        
    Returns:
        Oggetto con tipi Python nativi compatibili con JSON
        
    Examples:
        >>> import numpy as np
        >>> result = convert_numpy_types({'label': np.str_('test'), 'confidence': np.float32(0.95)})
        >>> print(result)  # {'label': 'test', 'confidence': 0.95}
        
        >>> probs = {np.str_('class1'): np.float32(0.7), np.str_('class2'): np.float32(0.3)}
        >>> result = convert_numpy_types(probs)
        >>> print(result)  # {'class1': 0.7, 'class2': 0.3}
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_) or obj is np.True_ or obj is np.False_:
        return bool(obj)
    elif isinstance(obj, np.str_) or isinstance(obj, np.bytes_):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Converte sia le chiavi che i valori ricorsivamente
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def ensure_json_serializable(data: Any) -> Any:
    """
    Alias per convert_numpy_types per maggiore chiarezza semantica.
    
    Args:
        data: Dati da rendere serializzabili in JSON
        
    Returns:
        Dati con tipi compatibili con JSON
    """
    return convert_numpy_types(data)


def validate_json_serializable(obj: Any) -> bool:
    """
    Valida se un oggetto è serializzabile in JSON senza errori.
    
    Args:
        obj: Oggetto da validare
        
    Returns:
        True se l'oggetto è serializzabile, False altrimenti
    """
    import json
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serializza un oggetto in JSON convertendo automaticamente i tipi NumPy.
    
    Args:
        obj: Oggetto da serializzare
        **kwargs: Argomenti aggiuntivi per json.dumps
        
    Returns:
        Stringa JSON
        
    Raises:
        TypeError: Se l'oggetto non è serializzabile dopo la conversione
    """
    import json
    converted_obj = convert_numpy_types(obj)
    return json.dumps(converted_obj, **kwargs)
