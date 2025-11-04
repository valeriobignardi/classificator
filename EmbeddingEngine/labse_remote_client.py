"""
Client remoto per servizio LaBSE embedding dockerizzato
Autore: Valerio Bignardi
Data: 2025-08-29

Scopo: Client HTTP per comunicare con servizio embedding dockerizzato,
       sostituendo le istanze locali di LaBSEEmbedder per ottimizzare memoria
"""

import os
import sys
import requests
import numpy as np
import time
from typing import List, Optional, Dict, Any, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Aggiungi path per importare base class
sys.path.append(os.path.dirname(__file__))
from base_embedder import BaseEmbedder

class LaBSERemoteClient(BaseEmbedder):
    """
    Client remoto per servizio LaBSE embedding dockerizzato
    
    Sostituisce LaBSEEmbedder locale con chiamate REST al servizio dedicato,
    eliminando il carico di memoria locale del modello LaBSE
    """
    
    def __init__(self, 
                 service_url: str = "http://localhost:8081",  # Aggiornato per servizio Docker
                 timeout: int = 14400,  # 4 ore per batch grandi
                 max_retries: int = 3,
                 fallback_local: bool = True):
        """
        Inizializza client remoto LaBSE
        
        Args:
            service_url: URL del servizio embedding
            timeout: Timeout richieste HTTP (secondi) - Default 4 ore per batch grandi
            max_retries: Numero massimo tentativi
            fallback_local: Se usare LaBSEEmbedder locale come fallback
        """
        super().__init__()
        
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_local = fallback_local
        self.embedding_dim = 768  # LaBSE standard dimension
        
        # Configurazione sessione HTTP con connection pooling e retry
        self.session = requests.Session()
        
        # Retry strategy per resilienza
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],  # Nuovo parametro compatibile
            backoff_factor=0.5
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers default
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'LaBSERemoteClient/1.0'
        })
        
        # Cache per fallback locale
        self._local_embedder = None
        
        # Statistiche client
        self.stats = {
            'requests_count': 0,
            'total_processing_time': 0.0,
            'fallback_count': 0,
            'error_count': 0,
            'last_request_time': None
        }
        
        print(f"🔗 LaBSE Remote Client inizializzato")
        print(f"   📡 Service URL: {self.service_url}")
        print(f"   ⏱️  Timeout: {timeout}s")
        print(f"   🔄 Max retries: {max_retries}")
        print(f"   🛡️ Fallback locale: {fallback_local}")
        
        # Carica/inizializza il modello (verifica connessione)
        self.load_model()
    
    def load_model(self):
        """
        Carica/inizializza il modello di embedding
        
        Per il client remoto questo significa verificare la connessione al servizio.
        Il modello effettivo viene caricato nel servizio Docker.
        
        Last update: 2025-01-28
        """
        print(f"🔧 Inizializzazione client remoto LaBSE...")
        
        # Per il client remoto, "caricare il modello" significa verificare 
        # che il servizio sia raggiungibile e operativo
        if not self._test_connection():
            if self.fallback_local:
                print(f"⚠️ Servizio remoto non disponibile, preparazione fallback locale...")
                # Prepara fallback locale per uso futuro
                self._get_local_fallback()
            else:
                raise RuntimeError("Servizio embedding remoto non disponibile e fallback disabilitato")
        
        # Imposta parametri modello standard per LaBSE
        self.model_name = "sentence-transformers/LaBSE"
        self.embedding_dim = 768
        self.model = "remote_service"  # Placeholder per indicare servizio remoto
        
        print(f"✅ Client remoto LaBSE inizializzato con successo")
    
    def get_embedding_dimension(self) -> int:
        """
        Restituisce la dimensione degli embedding
        
        Returns:
            Dimensione degli embedding LaBSE (768)
        
        Last update: 2025-01-28
        """
        return self.embedding_dim
    
    def _test_connection(self) -> bool:
        """
        Test connessione al servizio embedding
        
        Returns:
            True se servizio raggiungibile e funzionante
        """
        try:
            response = self.session.get(
                f"{self.service_url}/health", 
                timeout=10
            )
            
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    print(f"✅ Servizio embedding online e funzionante")
                    return True
                else:
                    print(f"⚠️ Servizio embedding online ma non healthy: {health_data}")
                    return False
            else:
                print(f"⚠️ Servizio embedding non risponde correttamente: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Impossibile raggiungere servizio embedding: {e}")
            if self.fallback_local:
                print(f"🔄 Fallback locale disponibile se necessario")
            return False
    
    def _get_local_fallback(self):
        """
        🚫 FALLBACK LOCALE DISABILITATO - Solo Docker service
        
        Returns:
            None - Nessun fallback locale disponibile
        """
        if self.fallback_local:
            print(f"⚠️ Fallback locale richiesto ma DISABILITATO - solo Docker")
        return None
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: Optional[int] = None,
               show_progress_bar: bool = False,
               session_ids: Optional[List[str]] = None,
               **kwargs) -> np.ndarray:
        """
        Genera embeddings tramite servizio remoto
        
        Args:
            texts: Testo singolo o lista di testi
            normalize_embeddings: Normalizza embeddings
            batch_size: Dimensione batch (max 32)
            show_progress_bar: Mostra progress bar (ignorato per compatibilità)
            session_ids: ID sessioni opzionali
            **kwargs: Parametri aggiuntivi (ignorati)
        
        Returns:
            Array numpy con embeddings generati
        """
        start_time = time.time()
        
        # Normalizza input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise ValueError("Lista testi vuota")
        
        # Aggiorna statistiche
        self.stats['requests_count'] += 1
        self.stats['last_request_time'] = start_time
        
        try:
            # Prepara richiesta
            request_data = {
                "texts": texts,
                "normalize_embeddings": normalize_embeddings,
                "session_ids": session_ids
            }
            
            if batch_size is not None:
                request_data["batch_size"] = batch_size
            
            # Chiamata al servizio remoto
            print(f"📡 Richiesta embedding remoto per {len(texts)} testi...")
            
            response = self.session.post(
                f"{self.service_url}/embed",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Successo - processa risposta
                result = response.json()
                embeddings = np.array(result["embeddings"])
                
                processing_time = time.time() - start_time
                self.stats['total_processing_time'] += processing_time
                
                print(f"✅ Embedding remoto completato in {processing_time:.3f}s")
                print(f"   📊 Shape: {embeddings.shape}")
                print(f"   ⚡ Tempo servizio: {result['processing_time']:.3f}s")
                
                return embeddings
                
            else:
                # Errore HTTP - prova fallback
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Errore servizio embedding: {error_msg}")
                return self._fallback_encode(texts, normalize_embeddings, batch_size, error_msg)
                
        except requests.exceptions.RequestException as e:
            # Errore connessione - prova fallback  
            error_msg = f"Errore connessione: {e}"
            print(f"❌ Errore connessione servizio embedding: {error_msg}")
            return self._fallback_encode(texts, normalize_embeddings, batch_size, error_msg)
            
        except Exception as e:
            # Errore generico - prova fallback
            error_msg = f"Errore generico: {e}"
            print(f"❌ Errore embedding remoto: {error_msg}")
            return self._fallback_encode(texts, normalize_embeddings, batch_size, error_msg)
    
    def _fallback_encode(self, 
                        texts: List[str], 
                        normalize_embeddings: bool,
                        batch_size: Optional[int],
                        error_msg: str) -> np.ndarray:
        """
        Fallback su embedding locale in caso di errore servizio remoto
        
        Args:
            texts: Testi da processare
            normalize_embeddings: Normalizza embeddings
            batch_size: Dimensione batch
            error_msg: Messaggio errore originale
        
        Returns:
            Array embeddings da fallback locale
            
        Raises:
            RuntimeError: Se anche il fallback fallisce
        """
        if not self.fallback_local:
            self.stats['error_count'] += 1
            raise RuntimeError(f"Servizio embedding non disponibile e fallback disabilitato: {error_msg}")
        
        local_embedder = self._get_local_fallback()
        if local_embedder is None:
            self.stats['error_count'] += 1
            raise RuntimeError(f"Servizio embedding non disponibile e fallback locale non caricabile: {error_msg}")
        
        try:
            print(f"🔄 Utilizzo fallback locale per {len(texts)} testi...")
            
            embeddings = local_embedder.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size or 32,
                show_progress_bar=False
            )
            
            self.stats['fallback_count'] += 1
            processing_time = time.time() - self.stats['last_request_time']
            self.stats['total_processing_time'] += processing_time
            
            print(f"✅ Fallback locale completato in {processing_time:.3f}s")
            return embeddings
            
        except Exception as fallback_error:
            self.stats['error_count'] += 1
            raise RuntimeError(f"Sia servizio remoto che fallback locale falliti. Remoto: {error_msg}, Locale: {fallback_error}")

    def encode_single(self, text: str, normalize_embeddings: bool = True, **kwargs):
        """
        Convenience wrapper to encode a single text and return a single vector.

        Args:
            text: Input text to embed
            normalize_embeddings: Whether to normalize the vector
            **kwargs: Forwarded to encode()

        Returns:
            1D embedding vector
        """
        emb = self.encode([text], normalize_embeddings=normalize_embeddings, **kwargs)
        return emb[0]
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Ottieni informazioni sul servizio embedding
        
        Returns:
            Dizionario con info servizio o errore
        """
        try:
            response = self.session.get(f"{self.service_url}/info", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_client_stats(self) -> Dict[str, Any]:
        """
        Ottieni statistiche client
        
        Returns:
            Dizionario con statistiche utilizzo
        """
        stats_copy = self.stats.copy()
        
        if stats_copy['requests_count'] > 0:
            stats_copy['average_request_time'] = stats_copy['total_processing_time'] / stats_copy['requests_count']
            stats_copy['success_rate'] = 1 - (stats_copy['error_count'] / stats_copy['requests_count'])
            stats_copy['fallback_rate'] = stats_copy['fallback_count'] / stats_copy['requests_count']
        
        return stats_copy
    
    def clear_cache(self) -> bool:
        """
        Pulisce cache servizio remoto
        
        Returns:
            True se cache pulita con successo
        """
        try:
            response = self.session.delete(f"{self.service_url}/cache", timeout=30)
            return response.status_code == 200
        except:
            return False
    
    def test_model(self) -> bool:
        """
        Test funzionamento modello (compatibilità con BaseEmbedder)
        
        Returns:
            True se test superato
        """
        try:
            test_texts = ["Test di funzionamento", "Secondo testo di prova"]
            embeddings = self.encode(test_texts)
            
            # Verifica formato output
            if embeddings.shape[0] == 2 and embeddings.shape[1] == self.embedding_dim:
                print(f"✅ Test modello superato: shape {embeddings.shape}")
                return True
            else:
                print(f"❌ Test modello fallito: shape errata {embeddings.shape}")
                return False
                
        except Exception as e:
            print(f"❌ Test modello fallito: {e}")
            return False
    
    def __del__(self):
        """Cleanup risorse"""
        if hasattr(self, 'session'):
            self.session.close()
