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

# 🔍 Import tracing centralizzato per monitoring batch processing
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Utils'))
    from tracing import trace_all
except ImportError as ie:
    print(f"🔍 DEBUG: ImportError per tracing centralizzato: {ie}")
    def trace_all(function_name: str, action: str = "ENTER", called_from: str = None, **kwargs):
        """Fallback tracing function se il modulo principale non è disponibile"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Crea messaggio di tracing
        message = f"[{timestamp}] {action} {function_name}"
        if called_from:
            message += f" (from {called_from})"
            
        # Aggiungi parametri se presenti
        if kwargs:
            params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            message += f" - {params}"
            
        print(f"🔍 TRACE FALLBACK: {message}")
    
    print("🔍 DEBUG: Funzione trace_all fallback definita")


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
            **kwargs: Altri parametri richiesta (inclusi tools)
            
        Returns:
            Stringa chiave cache
            
        Data ultima modifica: 2025-09-07 - Aggiunto supporto tools per function calling
        """
        # Crea hash dei parametri rilevanti per cache
        cache_data = {
            'model': model,
            'messages': messages,
            'temperature': kwargs.get('temperature', 0.1),
            'max_tokens': kwargs.get('max_tokens', 150),
            'tools': kwargs.get('tools'),
            'tool_choice': kwargs.get('tool_choice')
        }
        
        import hashlib
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
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
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effettua chiamata chat completion OpenAI con controllo parallelismo e supporto tools
        
        Args:
            model: Nome modello OpenAI (es. 'gpt-4o')
            messages: Lista messaggi conversazione
            temperature: Creatività generazione (0.0-2.0)
            max_tokens: Token massimi risposta
            top_p: Nucleus sampling
            frequency_penalty: Penalità ripetizione frequenza
            presence_penalty: Penalità ripetizione presenza
            tools: Lista tools/funzioni disponibili per function calling
            tool_choice: Controllo selezione tool ("auto", "none", o specifico tool)
            **kwargs: Altri parametri OpenAI
            
        Returns:
            Risposta API OpenAI con testo generato e eventuali tool calls
            
        Data ultima modifica: 2025-09-07 - Aggiunto supporto OpenAI tools/function calling
        """
        # Genera chiave cache
        cache_key = self._generate_cache_key(model, messages, 
                                           temperature=temperature, 
                                           max_tokens=max_tokens,
                                           tools=tools,
                                           tool_choice=tool_choice)
        
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
            
            # 🔥 COMPATIBILITÀ GPT-4o: Verifica se response_format è specificato
            # GPT-4o usa 'response_format' NON 'text' (come GPT-5)
            # Se non specificato, lascia che il modello usi il default
            
            # Aggiungi tools se forniti (OpenAI function calling)
            if tools:
                payload['tools'] = tools
                if tool_choice:
                    payload['tool_choice'] = tool_choice
                else:
                    payload['tool_choice'] = "auto"  # Default per function calling
            
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
    
    
    async def responses_completion(
        self,
        model: str,
        input_data: Union[str, List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effettua chiamata GPT-5 Responses API con controllo parallelismo
        
        VINCOLO COMPATIBILITÀ OPENAI:
        ============================
        GPT-5 usa API 'responses' (NON 'chat/completions'):
        
        ✅ Parametri OBBLIGATORI GPT-5:
           - model: str
           - input: str | List[Dict]
           - text: { "format": { "type": "text" } }  ← RICHIESTO!
        
        ✅ Parametri OPZIONALI GPT-5:
           - max_output_tokens (sostituisce max_tokens)
           - tools, tool_choice
           - reasoning, metadata, etc.
        
        ❌ Parametri NON supportati (rimossi automaticamente):
           - temperature, top_p, frequency_penalty, presence_penalty
           - max_tokens (usare max_output_tokens)
           - response_format (usare text.format)
        
        📚 Riferimento: https://platform.openai.com/docs/api-reference/responses/create
        
        Args:
            model: Nome modello (es. 'gpt-5')
            input_data: Input per il modello - può essere:
                       - stringa semplice
                       - lista di dict con role/content (convertita internamente)
            **kwargs: Altri parametri compatibili (es. max_output_tokens, tools, reasoning)
            
        Returns:
            Risposta API OpenAI con output_text generato
            
        Data ultima modifica: 2025-11-03 - Fix: aggiunto text.format.name obbligatorio
        """
        # Genera chiave cache
        cache_key = self._generate_cache_key(model, input_data)
        
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
            # Converti input se è una lista di messaggi (formato chat)
            if isinstance(input_data, list):
                # Converti da formato messages a formato input per GPT-5
                input_text = self._convert_messages_to_input(input_data)
            else:
                input_text = input_data
            
            payload = {
                'model': model,
                'input': input_text,
            }

            # 🔥 VINCOLO OPENAI GPT-5: Aggiungi SEMPRE parametro 'text' obbligatorio
            # Documentazione: https://platform.openai.com/docs/api-reference/responses/create
            # GPT-5 richiede: text: { format: { type: "text" } } o { type: "json_schema", json_schema: {...} }
            if 'text' not in kwargs:
                payload['text'] = {
                    'format': {
                        'type': 'text'
                        # ❌ NON aggiungere 'name': OpenAI ritorna "Unknown parameter: 'text.format.name'"
                    }
                }

            # Mappa max_tokens -> max_output_tokens se presente
            if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
                payload['max_output_tokens'] = kwargs.pop('max_tokens')

            # Aggiungi tools e tool_choice se forniti
            tools = kwargs.pop('tools', None)
            tool_choice = kwargs.pop('tool_choice', None)
            if tools:
                payload['tools'] = tools
            if tool_choice:
                payload['tool_choice'] = tool_choice

            # Aggiungi altri kwargs ammessi senza imporre parametri obsoleti
            # Nota: per tua richiesta, possiamo ignorare temperature esplicitamente
            allowed_passthrough = ['top_p', 'reasoning', 'metadata', 'prompt_cache_key',
                                   'previous_response_id', 'conversation', 'background',
                                   'service_tier', 'user', 'text']
            for k in list(kwargs.keys()):
                if k in allowed_passthrough and kwargs[k] is not None:
                    payload[k] = kwargs.pop(k)
            
            # 🆕 GPT-5: Rimuovi TUTTI i parametri non supportati
            # GPT-5 API 'responses' supporta SOLO: model, input, text, max_output_tokens, tools, etc.
            unsupported_params = ['temperature', 'frequency_penalty', 'presence_penalty',
                                   'max_tokens', 'max_completion_tokens', 'response_format']
            for param in unsupported_params:
                payload.pop(param, None)
            
            # Effettua chiamata con sessione HTTP ottimizzata
            connector = aiohttp.TCPConnector(limit=self.max_parallel_calls)
            timeout = aiohttp.ClientTimeout(total=60)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout
            ) as session:
                response = await self._make_api_call(
                    session, 
                    'responses',  # Nuovo endpoint per GPT-5
                    payload
                )
                
                # Normalizza la risposta Responses API aggiungendo 'output_text' se assente
                try:
                    if isinstance(response, dict) and 'output_text' not in response:
                        text_parts: List[str] = []

                        # Preferisci il campo 'output' standard della Responses API
                        output = response.get('output')
                        if isinstance(output, list):
                            for item in output:
                                # Ogni item può essere un messaggio con 'content'
                                content = item.get('content') if isinstance(item, dict) else None
                                if isinstance(content, list):
                                    for block in content:
                                        if isinstance(block, dict):
                                            # Formati possibili: {'type': 'text', 'text': {'value': '...'}}
                                            # oppure {'type': 'output_text', 'text': {'value': '...'}}
                                            text_obj = block.get('text')
                                            if isinstance(text_obj, dict):
                                                val = text_obj.get('value')
                                                if isinstance(val, str):
                                                    text_parts.append(val)
                                            elif isinstance(text_obj, str):
                                                text_parts.append(text_obj)
                                            # Fallback: alcuni SDK espongono direttamente 'value'
                                            elif 'value' in block and isinstance(block.get('value'), str):
                                                text_parts.append(block.get('value'))

                        # Alcune varianti possono mettere direttamente 'content' in root
                        if not text_parts:
                            root_content = response.get('content')
                            if isinstance(root_content, list):
                                for block in root_content:
                                    if isinstance(block, dict):
                                        text_obj = block.get('text')
                                        if isinstance(text_obj, dict) and isinstance(text_obj.get('value'), str):
                                            text_parts.append(text_obj['value'])
                                        elif isinstance(text_obj, str):
                                            text_parts.append(text_obj)

                        normalized_text = ''.join(text_parts).strip()
                        # Imposta sempre il campo per semplificare i chiamanti
                        response['output_text'] = normalized_text
                except Exception as _normalize_err:
                    # Non bloccare la risposta in caso di formati inattesi
                    print(f"⚠️ [OpenAIService] Impossibile normalizzare Responses API: {_normalize_err}")

                # Cache la risposta
                self._cache_response(cache_key, response)
                
                return response


    def extract_responses_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Estrae function tool calls da una risposta della Responses API.

        Args:
            response: Dizionario risposta (Responses API)

        Returns:
            Lista di dict con struttura {"function": {"name": str, "arguments": str}}

        Data ultima modifica: 2025-10-25
        """
        tool_calls: List[Dict[str, Any]] = []
        try:
            output_items = response.get('output')
            if not isinstance(output_items, list):
                return tool_calls

            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get('type', '')

                # Varianti possibili: 'function_call', 'response_function_tool_call', 'tool_call'
                if item_type in ('function_call', 'response_function_tool_call', 'tool_call'):
                    name = item.get('name') or item.get('function', {}).get('name')
                    arguments = item.get('arguments') or item.get('function', {}).get('arguments')
                    if name and arguments is not None:
                        tool_calls.append({
                            'function': {
                                'name': name,
                                'arguments': arguments
                            }
                        })

                # Alcuni modelli possono annidare in 'content' la chiamata tool
                content = item.get('content')
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get('type', '')
                        if btype in ('function_call', 'response_function_tool_call', 'tool_call'):
                            name = block.get('name') or block.get('function', {}).get('name')
                            arguments = block.get('arguments') or block.get('function', {}).get('arguments')
                            if name and arguments is not None:
                                tool_calls.append({
                                    'function': {
                                        'name': name,
                                        'arguments': arguments
                                    }
                                })
        except Exception as e:
            print(f"⚠️ [OpenAIService] Errore estrazione tool calls (Responses): {e}")
        return tool_calls
    
    
    def _convert_messages_to_input(self, messages: List[Dict[str, str]]) -> str:
        """
        Converte formato messages (chat) in formato input per GPT-5
        
        Args:
            messages: Lista di dict con 'role' e 'content'
            
        Returns:
            Stringa formattata per GPT-5
            
        Data ultima modifica: 2025-10-25
        """
        # Converti messaggi in formato leggibile per GPT-5
        input_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Formatta in base al ruolo
            if role == 'system':
                input_parts.append(f"<|system|>\n{content}")
            elif role == 'user':
                input_parts.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                input_parts.append(f"<|assistant|>\n{content}")
        
        return "\n\n".join(input_parts)
    
    
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
        print(f"🔍 DEBUG: batch_chat_completions chiamato con {len(requests)} richieste")
        trace_all("OpenAIService.batch_chat_completions", "ENTER", 
                 requests_count=len(requests), max_concurrent=max_concurrent)
        print(f"🔍 DEBUG: trace_all chiamata completata")
        
        if not requests:
            trace_all("OpenAIService.batch_chat_completions", "EXIT", message="empty_requests")
            return []
        
        start_time = time.time()
        
        if max_concurrent:
            # Usa semaforo temporaneo per questo batch
            temp_semaphore = asyncio.Semaphore(min(max_concurrent, self.max_parallel_calls))
            effective_concurrency = min(max_concurrent, self.max_parallel_calls)
        else:
            temp_semaphore = self.semaphore
            effective_concurrency = self.max_parallel_calls
        
        trace_all("OpenAIService.batch_chat_completions", "INFO", 
                 effective_concurrency=effective_concurrency)
        
        async def process_request(request_data):
            async with temp_semaphore:
                return await self.chat_completion(**request_data)
        
        try:
            # Esegui tutte le richieste in parallelo
            trace_all("OpenAIService.batch_chat_completions", "INFO", 
                     message="starting_parallel_execution")
            
            tasks = [process_request(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Converte eccezioni in errori strutturati
            processed_results = []
            success_count = 0
            error_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'error': str(result),
                        'request_index': i,
                        'success': False
                    })
                    error_count += 1
                else:
                    processed_results.append({
                        **result,
                        'request_index': i,
                        'success': True
                    })
                    if result.get('error'):
                        error_count += 1
                    else:
                        success_count += 1
            
            execution_time = time.time() - start_time
            
            trace_all("OpenAIService.batch_chat_completions", "EXIT", 
                     success_count=success_count, error_count=error_count, 
                     execution_time=f"{execution_time:.2f}s", 
                     avg_per_request=f"{execution_time/len(requests):.3f}s")
            
            return processed_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            trace_all("OpenAIService.batch_chat_completions", "ERROR", 
                     error=str(e), execution_time=f"{execution_time:.2f}s")
            raise
    
    
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


    def create_function_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crea un tool per OpenAI function calling
        
        Args:
            name: Nome della funzione
            description: Descrizione della funzione
            parameters: Schema JSON dei parametri (JSON Schema format)
            
        Returns:
            Dizionario tool formato OpenAI
            
        Data ultima modifica: 2025-09-07
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }


    def validate_tool_schema(self, tool: Dict[str, Any]) -> bool:
        """
        Valida che un tool segua il formato OpenAI corretto
        
        Args:
            tool: Dizionario tool da validare
            
        Returns:
            True se valido, False altrimenti
            
        Data ultima modifica: 2025-09-07
        """
        required_fields = ["type", "function"]
        if not all(field in tool for field in required_fields):
            return False
            
        if tool["type"] != "function":
            return False
            
        function = tool["function"]
        function_required = ["name", "description", "parameters"]
        if not all(field in function for field in function_required):
            return False
            
        # Valida che parameters sia un schema JSON valido
        parameters = function["parameters"]
        if not isinstance(parameters, dict):
            return False
            
        # Deve avere type e properties
        if "type" not in parameters or "properties" not in parameters:
            return False
            
        return True


    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Estrae le tool calls dalla risposta OpenAI
        
        Args:
            response: Risposta completa da OpenAI API
            
        Returns:
            Lista di tool calls se presenti, altrimenti lista vuota
            
        Data ultima modifica: 2025-09-07
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return []
                
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            return tool_calls
            
        except Exception as e:
            print(f"⚠️ [OpenAIService] Errore estrazione tool calls: {e}")
            return []


    def create_tool_message(
        self,
        tool_call_id: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Crea un messaggio di risposta per tool call
        
        Args:
            tool_call_id: ID del tool call da rispondere
            content: Contenuto della risposta (solitamente JSON)
            
        Returns:
            Messaggio formato per continuare la conversazione
            
        Data ultima modifica: 2025-09-07
        """
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "content": content
        }


    def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        available_functions: Dict[str, callable]
    ) -> List[Dict[str, Any]]:
        """
        Esegue una lista di tool calls con le funzioni disponibili
        
        Args:
            tool_calls: Lista tool calls estratti dalla risposta OpenAI
            available_functions: Dizionario nome_funzione -> funzione callable
            
        Returns:
            Lista messaggi di risposta per continuare la conversazione
            
        Data ultima modifica: 2025-09-07
        """
        tool_messages = []
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                print(f"🔧 [OpenAIService] Eseguendo tool call: {function_name}")
                print(f"   📋 Argomenti: {function_args}")
                
                if function_name in available_functions:
                    # Esegui la funzione
                    function_response = available_functions[function_name](**function_args)
                    
                    # Converti la risposta in JSON se non è già una stringa
                    if not isinstance(function_response, str):
                        function_response = json.dumps(function_response, default=str, ensure_ascii=False)
                    
                    print(f"   ✅ Risultato: {function_response[:100]}...")
                    
                    # Crea messaggio di risposta
                    tool_message = self.create_tool_message(tool_call_id, function_response)
                    tool_messages.append(tool_message)
                    
                else:
                    error_msg = f"Funzione '{function_name}' non disponibile"
                    print(f"   ❌ Errore: {error_msg}")
                    
                    tool_message = self.create_tool_message(
                        tool_call_id, 
                        json.dumps({"error": error_msg})
                    )
                    tool_messages.append(tool_message)
                    
            except Exception as e:
                error_msg = f"Errore esecuzione tool call: {str(e)}"
                print(f"   ❌ {error_msg}")
                
                tool_message = self.create_tool_message(
                    tool_call.get("id", "unknown"),
                    json.dumps({"error": error_msg})
                )
                tool_messages.append(tool_message)
        
        return tool_messages


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
