#!/usr/bin/env python3
"""
============================================================================
OpenAI Service - Gestione API OpenAI con Parallelismo
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Servizio specializzato per gestire chiamate API OpenAI con supporto
    per parallelismo controllato, rate limiting e retry logic.

Funzionalità principali:
    - Chiamate parallele controllate (max 200 concurrent)
    - Rate limiting intelligente
    - Retry automatico con backoff exponential
    - Monitoring delle chiamate e latenza
    - Cache per ottimizzare le performance

============================================================================
"""

import os
import time
import asyncio
import aiohttp
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from queue import Queue
import backoff


@dataclass
class OpenAICallStats:
    """
    Struttura per tracking delle statistiche chiamate OpenAI
    
    Scopo:
        Mantiene statistiche dettagliate sulle chiamate API per monitoring
        e ottimizzazione delle performance
        
    Data ultima modifica: 2025-01-31
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_used: int = 0
    average_latency: float = 0.0
    current_parallel_calls: int = 0
    max_parallel_reached: int = 0
    rate_limit_hits: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class OpenAIService:
    """
    Servizio per gestione chiamate OpenAI API con parallelismo controllato
    
    Scopo:
        Fornisce interfaccia ottimizzata per chiamate OpenAI API con controllo
        del parallelismo, rate limiting e gestione errori avanzata
        
    Parametri input:
        api_key: Chiave API OpenAI
        max_parallel_calls: Numero massimo chiamate parallele (default: 200)
        rate_limit_per_minute: Limite chiamate per minuto
        
    Parametri output:
        Istanza servizio configurato per chiamate OpenAI
        
    Data ultima modifica: 2025-01-31
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        max_parallel_calls: int = 200,
        rate_limit_per_minute: int = 10000,
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        Inizializza servizio OpenAI con configurazione parallelismo
        
        Args:
            api_key: Chiave API OpenAI (da .env se non specificata)
            max_parallel_calls: Max chiamate parallele simultanee
            rate_limit_per_minute: Limite chiamate per minuto
            base_url: URL base API OpenAI
        """
        # Carica API key da environment se non fornita
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY non trovata in environment o parametri")
        
        # Configurazione parallelismo e rate limiting
        self.max_parallel_calls = max_parallel_calls
        self.rate_limit_per_minute = rate_limit_per_minute
        self.base_url = base_url.rstrip('/')
        
        # Controllo concorrenza
        self.semaphore = asyncio.Semaphore(max_parallel_calls)
        self.call_queue = Queue()
        self.stats = OpenAICallStats()
        self.stats_lock = threading.Lock()
        
        # Rate limiting con sliding window
        self.call_timestamps = []
        self.rate_limit_lock = threading.Lock()
        
        # Cache per evitare chiamate duplicate
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minuti TTL
        self.cache_lock = threading.Lock()
        
        # Headers HTTP standard
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'ClassificatoreAI/1.0'
        }
        
        print(f"🤖 [OpenAIService] Servizio inizializzato - Max parallel: {max_parallel_calls}")
    
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica se possiamo effettuare una chiamata rispettando rate limits
        
        Returns:
            True se chiamata permessa, False se in rate limit
            
        Data ultima modifica: 2025-01-31
        """
        with self.rate_limit_lock:
            current_time = time.time()
            minute_ago = current_time - 60
            
            # Rimuovi chiamate più vecchie di 1 minuto
            self.call_timestamps = [
                ts for ts in self.call_timestamps 
                if ts > minute_ago
            ]
            
            # Controlla se possiamo effettuare la chiamata
            if len(self.call_timestamps) >= self.rate_limit_per_minute:
                with self.stats_lock:
                    self.stats.rate_limit_hits += 1
                return False
            
            # Aggiungi timestamp corrente
            self.call_timestamps.append(current_time)
            return True
    
    
    def _generate_cache_key(self, model: str, messages: List[Dict], **kwargs) -> str:
        """
        Genera chiave cache per la richiesta
        
        Args:
            model: Nome modello OpenAI
            messages: Lista messaggi conversazione
            **kwargs: Altri parametri richiesta
            
        Returns:
            Stringa chiave cache
            
        Data ultima modifica: 2025-01-31
        """
        # Crea hash dei parametri rilevanti per cache
        cache_data = {
            'model': model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.1),
            'max_tokens': kwargs.get('max_tokens', 150)
        }
        
        import hashlib
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Recupera risposta dalla cache se disponibile e valida
        
        Args:
            cache_key: Chiave cache da cercare
            
        Returns:
            Risposta cached o None se non trovata/scaduta
            
        Data ultima modifica: 2025-01-31
        """
        with self.cache_lock:
            if cache_key in self.response_cache:
                cached_entry = self.response_cache[cache_key]
                if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                    print(f"♻️  [OpenAIService] Cache hit per chiave: {cache_key[:8]}...")
                    return cached_entry['response']
                else:
                    # Cache scaduta, rimuovi
                    del self.response_cache[cache_key]
        
        return None
    
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """
        Salva risposta in cache
        
        Args:
            cache_key: Chiave cache
            response: Risposta da cachare
            
        Data ultima modifica: 2025-01-31
        """
        with self.cache_lock:
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            # Pulizia cache se troppo grande (max 1000 entries)
            if len(self.response_cache) > 1000:
                # Rimuovi le entry più vecchie
                sorted_entries = sorted(
                    self.response_cache.items(),
                    key=lambda x: x[1]['timestamp']
                )
                # Mantieni solo le ultime 800
                self.response_cache = dict(sorted_entries[-800:])
    
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def _make_api_call(
        self, 
        session: aiohttp.ClientSession,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Effettua chiamata HTTP asincrona all'API OpenAI con retry
        
        Args:
            session: Sessione HTTP asincrona
            endpoint: Endpoint API (es. 'chat/completions')
            payload: Dati da inviare
            
        Returns:
            Risposta API OpenAI
            
        Data ultima modifica: 2025-01-31
        """
        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload, headers=self.headers) as response:
                # Aggiorna statistiche chiamate parallele
                with self.stats_lock:
                    self.stats.current_parallel_calls += 1
                    if self.stats.current_parallel_calls > self.stats.max_parallel_reached:
                        self.stats.max_parallel_reached = self.stats.current_parallel_calls
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Aggiorna statistiche successo
                    call_duration = time.time() - start_time
                    with self.stats_lock:
                        self.stats.successful_calls += 1
                        self.stats.total_calls += 1
                        self.stats.current_parallel_calls -= 1
                        
                        # Aggiorna latenza media
                        total_successful = self.stats.successful_calls
                        self.stats.average_latency = (
                            (self.stats.average_latency * (total_successful - 1) + call_duration) 
                            / total_successful
                        )
                        
                        # Conta token se disponibili
                        if 'usage' in result:
                            self.stats.total_tokens_used += result['usage'].get('total_tokens', 0)
                    
                    return result
                
                else:
                    # Gestione errori HTTP
                    error_text = await response.text()
                    with self.stats_lock:
                        self.stats.failed_calls += 1
                        self.stats.total_calls += 1
                        self.stats.current_parallel_calls -= 1
                    
                    raise aiohttp.ClientError(
                        f"HTTP {response.status}: {error_text}"
                    )
        
        except Exception as e:
            # Aggiorna statistiche errore
            with self.stats_lock:
                self.stats.failed_calls += 1
                self.stats.total_calls += 1
                self.stats.current_parallel_calls -= 1
            
            print(f"❌ [OpenAIService] Errore chiamata API: {e}")
            raise
    
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 150,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effettua chiamata chat completion OpenAI con controllo parallelismo
        
        Args:
            model: Nome modello OpenAI (es. 'gpt-4o')
            messages: Lista messaggi conversazione
            temperature: Creatività generazione (0.0-2.0)
            max_tokens: Token massimi risposta
            top_p: Nucleus sampling
            frequency_penalty: Penalità ripetizione frequenza
            presence_penalty: Penalità ripetizione presenza
            **kwargs: Altri parametri OpenAI
            
        Returns:
            Risposta API OpenAI con testo generato
            
        Data ultima modifica: 2025-01-31
        """
        # Genera chiave cache
        cache_key = self._generate_cache_key(model, messages, 
                                           temperature=temperature, 
                                           max_tokens=max_tokens)
        
        # Controlla cache
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Controllo rate limiting
        while not self._check_rate_limit():
            print("⏳ [OpenAIService] Rate limit raggiunto, attendo...")
            await asyncio.sleep(1.0)
        
        # Acquisci semaforo per controllo concorrenza
        async with self.semaphore:
            payload = {
                'model': model,
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'frequency_penalty': frequency_penalty,
                'presence_penalty': presence_penalty,
                **kwargs
            }
            
            # Effettua chiamata con sessione HTTP ottimizzata
            connector = aiohttp.TCPConnector(limit=self.max_parallel_calls)
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout
            ) as session:
                response = await self._make_api_call(
                    session, 
                    'chat/completions', 
                    payload
                )
                
                # Cache la risposta
                self._cache_response(cache_key, response)
                
                return response
    
    
    async def batch_chat_completions(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = None
    ) -> List[Dict[str, Any]]:
        """
        Effettua multiple chat completions in parallelo
        
        Args:
            requests: Lista richieste, ognuna con model, messages, etc.
            max_concurrent: Override limite concorrenza per questo batch
            
        Returns:
            Lista risposte corrispondenti alle richieste
            
        Data ultima modifica: 2025-01-31
        """
        if max_concurrent:
            # Usa semaforo temporaneo per questo batch
            temp_semaphore = asyncio.Semaphore(min(max_concurrent, self.max_parallel_calls))
        else:
            temp_semaphore = self.semaphore
        
        async def process_request(request_data):
            async with temp_semaphore:
                return await self.chat_completion(**request_data)
        
        # Esegui tutte le richieste in parallelo
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Converte eccezioni in errori strutturati
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'request_index': i,
                    'success': False
                })
            else:
                processed_results.append({
                    **result,
                    'request_index': i,
                    'success': True
                })
        
        return processed_results
    
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Recupera statistiche correnti del servizio
        
        Returns:
            Dizionario con statistiche dettagliate
            
        Data ultima modifica: 2025-01-31
        """
        with self.stats_lock:
            uptime_seconds = (datetime.now() - self.stats.last_reset).total_seconds()
            
            return {
                'total_calls': self.stats.total_calls,
                'successful_calls': self.stats.successful_calls,
                'failed_calls': self.stats.failed_calls,
                'success_rate': (
                    self.stats.successful_calls / max(self.stats.total_calls, 1) * 100
                ),
                'total_tokens_used': self.stats.total_tokens_used,
                'average_latency_seconds': round(self.stats.average_latency, 3),
                'current_parallel_calls': self.stats.current_parallel_calls,
                'max_parallel_reached': self.stats.max_parallel_reached,
                'rate_limit_hits': self.stats.rate_limit_hits,
                'uptime_seconds': round(uptime_seconds, 1),
                'cache_size': len(self.response_cache),
                'calls_per_minute': round(
                    self.stats.total_calls / max(uptime_seconds / 60, 1), 2
                )
            }
    
    
    def reset_stats(self):
        """
        Resetta le statistiche del servizio
        
        Data ultima modifica: 2025-01-31
        """
        with self.stats_lock:
            self.stats = OpenAICallStats()
            print("📊 [OpenAIService] Statistiche resettate")
    
    
    def clear_cache(self):
        """
        Pulisce la cache delle risposte
        
        Data ultima modifica: 2025-01-31
        """
        with self.cache_lock:
            cache_size = len(self.response_cache)
            self.response_cache.clear()
            print(f"🗑️ [OpenAIService] Cache pulita ({cache_size} entries rimosse)")


# =============================================================================
# Funzioni di utilità per integrazione con il sistema esistente
# =============================================================================

def create_openai_service_from_config(config: Dict[str, Any]) -> OpenAIService:
    """
    Crea istanza OpenAIService da configurazione YAML
    
    Args:
        config: Dizionario configurazione da config.yaml
        
    Returns:
        Istanza OpenAIService configurata
        
    Data ultima modifica: 2025-01-31
    """
    openai_config = config.get('llm', {}).get('openai', {})
    
    return OpenAIService(
        max_parallel_calls=openai_config.get('max_parallel_calls', 200),
        rate_limit_per_minute=openai_config.get('rate_limit_per_minute', 10000),
        base_url=openai_config.get('api_base', 'https://api.openai.com/v1')
    )


def sync_chat_completion(
    service: OpenAIService,
    model: str,
    messages: List[Dict[str, str]],
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper sincrono per chat_completion (per compatibilità)
    
    Args:
        service: Istanza OpenAIService
        model: Nome modello
        messages: Messaggi conversazione
        **kwargs: Altri parametri
        
    Returns:
        Risposta OpenAI
        
    Data ultima modifica: 2025-01-31
    """
    try:
        # Ottieni o crea event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Esegui chiamata asincrona
        return loop.run_until_complete(
            service.chat_completion(model, messages, **kwargs)
        )
    
    except Exception as e:
        print(f"❌ [OpenAIService] Errore chiamata sincrona: {e}")
        raise
