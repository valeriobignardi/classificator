#!/usr/bin/env python3
"""
File: bge_m3_embedder.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Engine di embedding BGE-M3 tramite Ollama per sistema di configurazione AI

Storia aggiornamenti:
2025-08-25 - Creazione iniziale con supporto Ollama BGE-M3
2025-08-26 - Aggiunta tokenizzazione preventiva per conversazioni lunghe
"""

import sys
import os
import numpy as np
import requests
import json
import time
from typing import Union, List, Optional, Dict, Any
import logging

# Aggiunta del percorso per BaseEmbedder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_embedder import BaseEmbedder

# Import TokenizationManager per gestione conversazioni lunghe
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
try:
    from tokenization_utils import TokenizationManager
    TOKENIZATION_AVAILABLE = True
except ImportError:
    TokenizationManager = None
    TOKENIZATION_AVAILABLE = False
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_embedder import BaseEmbedder


class BGE_M3_Embedder(BaseEmbedder):
    """
    Classe per generazione embedding usando BGE-M3 tramite Ollama
    
    Scopo: Fornire un'alternativa a LaBSE usando il modello BGE-M3
    tramite server Ollama locale per embedding multilingue avanzati
    
    Parametri di input:
    - ollama_url: URL del server Ollama (default: http://localhost:11434)
    - model_name: Nome del modello BGE-M3 in Ollama
    - embedding_dim: Dimensione degli embedding BGE-M3 (1024)
    - timeout: Timeout per le richieste HTTP
    
    Valori di ritorno:
    - embeddings normalizzati numpy array shape (n_samples, 1024)
    
    Data ultima modifica: 2025-08-25
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "bge-m3:latest",
                 timeout: int = 300,
                 test_on_init: bool = True):
        """
        Inizializza BGE-M3 embedder con connessione Ollama
        
        Args:
            ollama_url: URL del server Ollama
            model_name: Nome modello BGE-M3 in Ollama
            timeout: Timeout richieste in secondi
            test_on_init: Test modello dopo inizializzazione
        """
        super().__init__()
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.embedding_dim = 1024  # BGE-M3 produce embedding 1024D
        self.timeout = timeout
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Inizializza TokenizationManager per gestione conversazioni lunghe
        self.tokenizer = None
        if TOKENIZATION_AVAILABLE:
            try:
                self.tokenizer = TokenizationManager()
                print(f"✅ TokenizationManager integrato in BGE-M3 per gestione conversazioni lunghe")
            except Exception as e:
                print(f"⚠️  Errore inizializzazione TokenizationManager in BGE-M3: {e}")
                self.tokenizer = None
        else:
            print(f"⚠️  TokenizationManager non disponibile per BGE-M3")
        
        print(f"🚀 Inizializzazione BGE-M3 embedder:")
        print(f"   📡 Server Ollama: {self.ollama_url}")
        print(f"   🎯 Modello: {self.model_name}")
        print(f"   📏 Dimensione embedding: {self.embedding_dim}")
        
        # Verifica disponibilità del modello
        self._check_model_availability()
        
        if test_on_init:
            if not self.test_model():
                print(f"⚠️  Attenzione: il modello BGE-M3 potrebbe non funzionare correttamente")
            else:
                print(f"🎯 Modello BGE-M3 pronto per l'uso!")
    
    def _check_model_availability(self):
        """
        Verifica che il modello BGE-M3 sia disponibile in Ollama
        """
        try:
            # Lista modelli disponibili
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
            response.raise_for_status()
            
            available_models = [model['name'] for model in response.json().get('models', [])]
            
            if self.model_name not in available_models:
                print(f"⚠️  Modello {self.model_name} non trovato in Ollama")
                print(f"📋 Modelli disponibili: {available_models}")
                raise RuntimeError(f"Modello BGE-M3 non disponibile. Esegui: ollama pull {self.model_name}")
            
            print(f"✅ Modello {self.model_name} trovato e disponibile")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Impossibile connettersi a Ollama: {e}")
    
    def load_model(self):
        """
        Carica il modello BGE-M3 tramite Ollama
        
        Scopo: Implementa il metodo astratto per inizializzazione modello
        Note: BGE-M3 viene caricato automaticamente da Ollama al primo utilizzo
        
        Data ultima modifica: 2025-08-25
        """
        try:
            # Per BGE-M3 su Ollama, il modello viene caricato automaticamente
            # Verifichiamo solo che sia disponibile
            self._check_model_availability()
            
            # Impostiamo i parametri del modello
            self.model_name = self.model_name
            self.model = f"{self.ollama_url}/api/embeddings"
            
            print(f"✅ Modello BGE-M3 caricato e pronto")
            return True
            
        except Exception as e:
            print(f"❌ Errore nel caricamento modello BGE-M3: {e}")
            raise RuntimeError(f"Impossibile caricare modello BGE-M3: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Restituisce la dimensione degli embedding BGE-M3
        
        Scopo: Implementa il metodo astratto per ottenere dimensione embedding
        
        Returns:
            int: Dimensione embedding (1024 per BGE-M3)
            
        Data ultima modifica: 2025-08-25
        """
        return self.embedding_dim
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: int = 16,
               show_progress_bar: bool = False,
               session_ids: List[str] = None,  # Nuovo parametro per session_ids
               **kwargs) -> np.ndarray:
        """
        Genera embeddings per i testi usando BGE-M3 tramite Ollama
        con tokenizzazione preventiva per conversazioni lunghe
        
        Args:
            texts: Testo singolo o lista di testi
            normalize_embeddings: Se normalizzare gli embeddings
            batch_size: Dimensione batch per processing
            show_progress_bar: Mostra barra progresso
            session_ids: Lista degli ID di sessione corrispondenti ai testi (opzionale)
            
        Returns:
            Array numpy con embeddings shape (n_samples, 1024)
        """
        # Normalizza input
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"🔍 Encoding {len(texts)} testi con BGE-M3...")
        
        # ========================================================================
        # 🔥 TOKENIZZAZIONE PREVENTIVA PER BGE-M3
        # ========================================================================
        
        processed_texts = texts
        
        if self.tokenizer:
            print(f"\n🔍 TOKENIZZAZIONE PREVENTIVA BGE-M3 CLUSTERING")
            print(f"=" * 60)
            
            try:
                processed_texts, tokenization_stats = self.tokenizer.process_conversations_for_clustering(
                    texts, session_ids
                )
                
                print(f"✅ Tokenizzazione BGE-M3 completata:")
                print(f"   📊 Conversazioni processate: {tokenization_stats['processed_count']}")
                print(f"   📊 Conversazioni troncate: {tokenization_stats['truncated_count']}")
                print(f"   📊 Token limite configurato: {tokenization_stats['max_tokens']}")
                if tokenization_stats['truncated_count'] > 0:
                    print(f"   ✂️  {tokenization_stats['truncated_count']} conversazioni TRONCATE per rispettare limite token")
                else:
                    print(f"   ✅ Tutte le conversazioni entro i limiti, nessun troncamento")
                print(f"=" * 60)
                
            except Exception as e:
                print(f"⚠️  Errore durante tokenizzazione BGE-M3: {e}")
                print(f"🔄 Fallback a preprocessing legacy")
                processed_texts = texts
        else:
            print(f"⚠️  TokenizationManager non disponibile per BGE-M3")
            print(f"📏 Usando preprocessing legacy")
        
        embeddings = []
        
        try:
            # Processa in batch per gestire grandi volumi
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i+batch_size]
                batch_embeddings = []
                
                if show_progress_bar:
                    print(f"📊 Batch {i//batch_size + 1}/{(len(processed_texts)-1)//batch_size + 1}")
                
                for text in batch_texts:
                    # Preprocessing testo
                    if not text or not text.strip():
                        text = "testo vuoto"
                    else:
                        text = text.strip()
                    
                    # Richiesta embedding a Ollama
                    embedding = self._request_embedding(text)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
            
            # Converti a numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalizza se richiesto
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Evita divisione per zero
                embeddings_array = embeddings_array / norms
            
            print(f"✅ Embeddings BGE-M3 generati: shape {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            raise RuntimeError(f"Errore durante encoding BGE-M3: {e}")
    
    def _request_embedding(self, text: str) -> np.ndarray:
        """
        Richiede embedding per singolo testo a Ollama
        
        Args:
            text: Testo da processare
            
        Returns:
            Embedding vector come numpy array
        """
        try:
            # Payload per richiesta embedding
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            # Richiesta POST a Ollama
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Estrai embedding dalla risposta
            result = response.json()
            embedding = result.get('embedding')
            
            if not embedding:
                raise ValueError("Embedding non trovato nella risposta Ollama")
            
            return np.array(embedding, dtype=np.float32)
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout durante richiesta embedding per: {text[:50]}...")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore HTTP durante richiesta embedding: {e}")
        except Exception as e:
            raise RuntimeError(f"Errore processing embedding per '{text[:50]}...': {e}")
    
    def test_model(self) -> bool:
        """
        Testa il funzionamento del modello BGE-M3
        
        Returns:
            True se il test ha successo, False altrimenti
        """
        try:
            print(f"🧪 Test del modello BGE-M3...")
            
            # Testi di test multilingue per BGE-M3
            test_texts = [
                "Questo è un test di embedding in italiano",
                "This is an embedding test in English",
                "Ceci est un test d'embedding en français"
            ]
            
            # Genera embeddings di test
            embeddings = self.encode(test_texts, normalize_embeddings=True)
            
            # Validazioni
            if embeddings.shape != (3, self.embedding_dim):
                print(f"❌ Shape embedding errata: {embeddings.shape} != (3, {self.embedding_dim})")
                return False
            
            # Verifica assenza di NaN/infiniti
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print(f"❌ Embedding contengono NaN o infiniti")
                return False
            
            # Verifica normalizzazione
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                print(f"⚠️  Embedding non normalizzati: norme {norms}")
            
            print(f"✅ Test BGE-M3 completato con successo!")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Norme: {norms}")
            print(f"   Server: {self.ollama_url}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test BGE-M3 fallito: {e}")
            return False
    
    def get_embedding_dim(self) -> int:
        """
        Restituisce la dimensione degli embedding BGE-M3
        
        Returns:
            Dimensione embedding (1024)
        """
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """
        Verifica se BGE-M3 è disponibile
        
        Returns:
            True se disponibile, False altrimenti
        """
        try:
            # Test rapido di connessione
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            available_models = [model['name'] for model in response.json().get('models', [])]
            return self.model_name in available_models
            
        except Exception:
            return False
    
    def benchmark_speed(self, sample_texts: List[str], iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark delle performance di BGE-M3
        
        Args:
            sample_texts: Testi di test per benchmark
            iterations: Numero di iterazioni per media
            
        Returns:
            Metriche di performance
        """
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            self.encode(sample_texts, show_progress_bar=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        texts_per_second = len(sample_texts) / avg_time
        
        return {
            'avg_time_seconds': avg_time,
            'texts_per_second': texts_per_second,
            'total_texts': len(sample_texts),
            'iterations': iterations,
            'model': 'BGE-M3 (Ollama)'
        }
