"""
Factory per embedders con supporto client remoto
Autore: Valerio Bignardi
Data: 2025-08-29

Scopo: Factory pattern per gestire istanze embedder locali e remote,
       centralizzando la logica di creazione e configurazione
"""

import os
import sys
from typing import Optional, Dict, Any, Union
from enum import Enum

# Aggiungi path per importare embedders
sys.path.append(os.path.dirname(__file__))

class EmbedderType(Enum):
    """Tipi di embedder supportati"""
    LABSE_LOCAL = "labse_local"
    LABSE_REMOTE = "labse_remote"
    LABSE_AUTO = "labse_auto"  # Prova remoto, fallback su locale

class EmbedderFactory:
    """
    Factory per creazione istanze embedder
    
    Gestisce creazione ottimizzata di embedders locali e remoti,
    implementando pattern singleton per istanze condivise
    """
    
    _instances = {}  # Cache istanze condivise
    
    @classmethod
    def create_embedder(cls, 
                       embedder_type: Union[str, EmbedderType] = EmbedderType.LABSE_AUTO,
                       config: Optional[Dict[str, Any]] = None,
                       shared_instance: bool = True,
                       **kwargs) -> Any:
        """
        Crea istanza embedder del tipo specificato
        
        Args:
            embedder_type: Tipo di embedder (EmbedderType o stringa)
            config: Configurazione specifica
            shared_instance: Se usare istanza condivisa (singleton)
            **kwargs: Parametri aggiuntivi per costruttore
        
        Returns:
            Istanza embedder configurata
        """
        # Normalizza tipo
        if isinstance(embedder_type, str):
            try:
                embedder_type = EmbedderType(embedder_type.lower())
            except ValueError:
                raise ValueError(f"Tipo embedder non supportato: {embedder_type}")
        
        # Genera chiave cache se necessario
        cache_key = None
        if shared_instance:
            cache_key = f"{embedder_type.value}_{hash(str(sorted((config or {}).items())))}"
            
            # Restituisci istanza cached se disponibile
            if cache_key in cls._instances:
                print(f"🔄 Utilizzo istanza embedder condivisa: {embedder_type.value}")
                return cls._instances[cache_key]
        
        # Crea nuova istanza
        print(f"🔧 Creazione nuova istanza embedder: {embedder_type.value}")
        
        embedder = cls._create_embedder_instance(embedder_type, config, **kwargs)
        
        # Salva in cache se richiesto
        if shared_instance and cache_key:
            cls._instances[cache_key] = embedder
            print(f"💾 Istanza embedder salvata in cache: {cache_key}")
        
        return embedder
    
    @classmethod
    def _create_embedder_instance(cls, 
                                 embedder_type: EmbedderType,
                                 config: Optional[Dict[str, Any]],
                                 **kwargs) -> Any:
        """
        Crea istanza specifica di embedder
        
        Args:
            embedder_type: Tipo embedder da creare
            config: Configurazione
            **kwargs: Parametri costruttore
        
        Returns:
            Istanza embedder
        """
        # Merge configurazioni
        final_config = {}
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        if embedder_type == EmbedderType.LABSE_LOCAL:
            return cls._create_labse_local(final_config)
            
        elif embedder_type == EmbedderType.LABSE_REMOTE:
            return cls._create_labse_remote(final_config)
            
        elif embedder_type == EmbedderType.LABSE_AUTO:
            return cls._create_labse_auto(final_config)
            
        else:
            raise ValueError(f"Tipo embedder non implementato: {embedder_type}")
    
    @classmethod
    def _create_labse_local(cls, config: Dict[str, Any]) -> Any:
        """
        Crea istanza LaBSE locale
        
        Args:
            config: Configurazione per LaBSEEmbedder
        
        Returns:
            Istanza LaBSEEmbedder locale
        """
        try:
            from labse_embedder import LaBSEEmbedder
            
            # Parametri default
            local_config = {
                'test_on_init': config.get('test_on_init', False),
                'model_name': config.get('model_name', 'sentence-transformers/LaBSE'),
                'device': config.get('device', None)
            }
            
            print(f"📦 Caricamento LaBSEEmbedder locale...")
            embedder = LaBSEEmbedder(**local_config)
            print(f"✅ LaBSEEmbedder locale caricato")
            
            return embedder
            
        except ImportError as e:
            raise RuntimeError(f"Impossibile importare LaBSEEmbedder: {e}")
        except Exception as e:
            raise RuntimeError(f"Errore creazione LaBSEEmbedder locale: {e}")
    
    @classmethod  
    def _create_labse_remote(cls, config: Dict[str, Any]) -> Any:
        """
        Crea client LaBSE remoto
        
        Args:
            config: Configurazione per LaBSERemoteClient
        
        Returns:
            Istanza LaBSERemoteClient
        """
        try:
            from labse_remote_client import LaBSERemoteClient
            
            # Parametri default
            remote_config = {
                'service_url': config.get('service_url', 'http://localhost:8080'),
                'timeout': config.get('timeout', 300),
                'max_retries': config.get('max_retries', 3),
                'fallback_local': config.get('fallback_local', False)  # Solo remoto
            }
            
            print(f"🔗 Creazione client LaBSE remoto...")
            client = LaBSERemoteClient(**remote_config)
            print(f"✅ Client LaBSE remoto creato")
            
            return client
            
        except ImportError as e:
            raise RuntimeError(f"Impossibile importare LaBSERemoteClient: {e}")
        except Exception as e:
            raise RuntimeError(f"Errore creazione client LaBSE remoto: {e}")
    
    @classmethod
    def _create_labse_auto(cls, config: Dict[str, Any]) -> Any:
        """
        Crea LaBSE automatico (prova remoto, fallback locale)
        
        Args:
            config: Configurazione combinata
        
        Returns:
            Istanza LaBSERemoteClient con fallback locale
        """
        try:
            from labse_remote_client import LaBSERemoteClient
            
            # Configurazione auto con fallback
            auto_config = {
                'service_url': config.get('service_url', 'http://localhost:8080'),
                'timeout': config.get('timeout', 300),
                'max_retries': config.get('max_retries', 2),  # Meno retry per auto
                'fallback_local': True  # Sempre abilitato per auto
            }
            
            print(f"🤖 Creazione LaBSE automatico (remoto + fallback locale)...")
            client = LaBSERemoteClient(**auto_config)
            print(f"✅ LaBSE automatico configurato")
            
            return client
            
        except ImportError as e:
            # Se non disponibile remoto, usa locale
            print(f"⚠️ Client remoto non disponibile, uso LaBSE locale: {e}")
            return cls._create_labse_local(config)
        except Exception as e:
            # Se errore remoto, prova locale
            print(f"⚠️ Errore client remoto, uso LaBSE locale: {e}")
            return cls._create_labse_local(config)
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Ottieni lista tipi embedder disponibili
        
        Returns:
            Lista EmbedderType supportati
        """
        return list(EmbedderType)
    
    @classmethod
    def clear_cache(cls, embedder_type: Optional[Union[str, EmbedderType]] = None):
        """
        Pulisce cache istanze condivise
        
        Args:
            embedder_type: Tipo specifico da rimuovere (None per tutti)
        """
        if embedder_type is None:
            # Pulisci tutto
            count = len(cls._instances)
            cls._instances.clear()
            print(f"🧹 Cache embedder pulita: {count} istanze rimosse")
            
        else:
            # Normalizza tipo
            if isinstance(embedder_type, str):
                embedder_type = EmbedderType(embedder_type.lower())
            
            # Rimuovi istanze di tipo specifico
            keys_to_remove = [k for k in cls._instances.keys() if k.startswith(embedder_type.value)]
            for key in keys_to_remove:
                del cls._instances[key]
            
            print(f"🧹 Cache embedder pulita per tipo {embedder_type.value}: {len(keys_to_remove)} istanze rimosse")
    
    @classmethod
    def get_cached_instances(cls) -> Dict[str, Any]:
        """
        Ottieni dizionario istanze in cache
        
        Returns:
            Dizionario con chiavi cache e tipi istanze
        """
        return {
            key: type(instance).__name__ 
            for key, instance in cls._instances.items()
        }
    
    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """
        Verifica salute istanze in cache
        
        Returns:
            Report salute embedders
        """
        health_report = {
            'total_instances': len(cls._instances),
            'instances_status': {},
            'healthy_count': 0,
            'unhealthy_count': 0
        }
        
        for key, instance in cls._instances.items():
            try:
                # Test generico funzionamento
                if hasattr(instance, 'test_model'):
                    status = instance.test_model()
                else:
                    # Test base con encoding
                    result = instance.encode("test")
                    status = result is not None and len(result.shape) == 2
                
                health_report['instances_status'][key] = {
                    'healthy': status,
                    'type': type(instance).__name__
                }
                
                if status:
                    health_report['healthy_count'] += 1
                else:
                    health_report['unhealthy_count'] += 1
                    
            except Exception as e:
                health_report['instances_status'][key] = {
                    'healthy': False,
                    'error': str(e),
                    'type': type(instance).__name__
                }
                health_report['unhealthy_count'] += 1
        
        return health_report


# Funzioni di convenienza per backward compatibility
def get_embedder(config: Optional[Dict[str, Any]] = None, 
                embedder_type: str = "labse_auto",
                **kwargs) -> Any:
    """
    Funzione di convenienza per ottenere embedder
    
    Args:
        config: Configurazione
        embedder_type: Tipo embedder (default: auto)
        **kwargs: Parametri aggiuntivi
    
    Returns:
        Istanza embedder
    """
    return EmbedderFactory.create_embedder(
        embedder_type=embedder_type,
        config=config,
        shared_instance=True,
        **kwargs
    )

def create_remote_embedder(service_url: str = "http://localhost:8080",
                          fallback_local: bool = True,
                          **kwargs) -> Any:
    """
    Funzione di convenienza per client remoto
    
    Args:
        service_url: URL servizio embedding
        fallback_local: Abilita fallback locale
        **kwargs: Parametri aggiuntivi
    
    Returns:
        Istanza LaBSERemoteClient
    """
    config = {
        'service_url': service_url,
        'fallback_local': fallback_local
    }
    config.update(kwargs)
    
    return EmbedderFactory.create_embedder(
        embedder_type=EmbedderType.LABSE_REMOTE,
        config=config,
        shared_instance=True
    )
