#!/usr/bin/env python3
"""
File: openai_embedder.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Engine di embedding OpenAI con tokenizzazione preventiva per gestire conversazioni lunghe

Storia aggiornamenti:
2025-08-25 - Creazione iniziale con supporto modelli OpenAI embedding
2025-08-26 - Aggiunta tokenizzazione preventiva con tiktoken per conversazioni lunghe
"""

import sys
import os
import numpy as np
import requests
import json
import time
from typing import Union, List, Optional, Dict, Any
import logging

# Aggiunta del percorso per BaseEmbedder e TokenizationManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_embedder import BaseEmbedder

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from Utils.tokenization_utils import TokenizationManager

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIEmbedder(BaseEmbedder):
    """
    Classe per generazione embedding usando modelli OpenAI text-embedding
    
    Scopo: Fornire embedding di alta qualità tramite API OpenAI
    per confronto con LaBSE e BGE-M3 nel sistema di configurazione AI
    
    Parametri di input:
    - api_key: Chiave API OpenAI
    - model_name: Nome modello (text-embedding-3-large/text-embedding-3-small)
    - embedding_dim: Dimensione embedding (dipende dal modello)
    - max_tokens: Massimo token per richiesta
    
    Valori di ritorno:
    - embeddings normalizzati numpy array shape (n_samples, embedding_dim)
    
    Data ultima modifica: 2025-08-25
    """
    
    # Dimensioni embedding per i modelli OpenAI
    MODEL_DIMENSIONS = {
        'text-embedding-3-large': 3072,
        'text-embedding-3-small': 1536,
        'text-embedding-ada-002': 1536,  # Legacy model
    }
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "text-embedding-3-large",
                 max_tokens: int = 8000,
                 timeout: int = 300,
                 test_on_init: bool = True):
        """
        Inizializza OpenAI embedder con tokenizzazione preventiva
        
        Args:
            api_key: Chiave API OpenAI (se None, cerca in variabile ambiente)
            model_name: Nome modello OpenAI embedding
            max_tokens: Massimo token per richiesta
            timeout: Timeout richieste in secondi
            test_on_init: Test modello dopo inizializzazione
        """
        super().__init__()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("Libreria openai non installata. Esegui: pip install openai")
        
        # Configurazione API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("API key OpenAI non fornita. Impostare OPENAI_API_KEY o passare api_key")
        
        # Configurazione modello
        if model_name not in self.MODEL_DIMENSIONS:
            raise ValueError(f"Modello {model_name} non supportato. Disponibili: {list(self.MODEL_DIMENSIONS.keys())}")
        
        self.model_name = model_name
        self.embedding_dim = self.MODEL_DIMENSIONS[model_name]
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Setup OpenAI client (nuova sintassi v1.0+)
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Inizializza TokenizationManager per gestione conversazioni lunghe
        try:
            self.tokenizer = TokenizationManager()
            print(f"✅ TokenizationManager integrato con successo")
        except Exception as e:
            print(f"⚠️  Errore inizializzazione TokenizationManager: {e}")
            self.tokenizer = None
        
        print(f"🚀 Inizializzazione OpenAI embedder:")
        print(f"   🎯 Modello: {self.model_name}")
        print(f"   📏 Dimensione embedding: {self.embedding_dim}")
        print(f"   🔢 Max tokens: {self.max_tokens}")
        
        if test_on_init:
            if not self.test_model():
                print(f"⚠️  Attenzione: il modello OpenAI potrebbe non funzionare correttamente")
            else:
                print(f"🎯 Modello OpenAI pronto per l'uso!")
    
    def load_model(self):
        """
        Carica il modello OpenAI (metodo astratto richiesto da BaseEmbedder)
        
        Per OpenAI non c'è un modello da caricare localmente,
        il modello viene acceduto via API
        """
        # OpenAI non richiede caricamento locale
        # Il "modello" è già configurato con self.client
        pass
    
    def get_embedding_dimension(self) -> int:
        """
        Restituisce la dimensione degli embedding (metodo astratto richiesto da BaseEmbedder)
        
        Returns:
            Dimensione embedding del modello corrente
        """
        return self.embedding_dim
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: int = 100,  # OpenAI supporta batch più grandi
               show_progress_bar: bool = False,
               session_ids: List[str] = None,  # Nuovo parametro per session_ids
               **kwargs) -> np.ndarray:
        """
        Genera embeddings per i testi usando modello OpenAI con tokenizzazione preventiva
        
        Args:
            texts: Testo singolo o lista di testi
            normalize_embeddings: Se normalizzare gli embeddings
            batch_size: Dimensione batch (max 100 per OpenAI)
            show_progress_bar: Mostra barra progresso
            session_ids: Lista opzionale di session_id corrispondenti ai testi
            
        Returns:
            Array numpy con embeddings shape (n_samples, embedding_dim)
        """
        # Normalizza input
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"🔍 Encoding {len(texts)} testi con OpenAI {self.model_name}...")
        
        # Preprocessing con tokenizzazione preventiva
        if self.tokenizer:
            processed_texts, tokenization_stats = self.tokenizer.process_conversations_for_clustering(
                texts, session_ids
            )
            print(f"✅ Tokenizzazione completata: {tokenization_stats['truncated_count']} conversazioni troncate")
        else:
            print(f"⚠️  TokenizationManager non disponibile, uso preprocessing legacy")
            processed_texts = []
            for text in texts:
                if not text or not text.strip():
                    processed_texts.append("testo vuoto")
                else:
                    processed_texts.append(text.strip()[:30000])  # Limite caratteri di emergenza
        
        embeddings = []
        
        try:
            # Processa in batch
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i+batch_size]
                batch_session_ids = session_ids[i:i+batch_size] if session_ids else None
                
                if show_progress_bar:
                    print(f"📊 Batch {i//batch_size + 1}/{(len(processed_texts)-1)//batch_size + 1}")
                
                # Richiesta embedding batch a OpenAI
                batch_embeddings = self._request_embeddings_batch(batch_texts, batch_session_ids)
                embeddings.extend(batch_embeddings)
            
            # Converti a numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalizza se richiesto
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Evita divisione per zero
                embeddings_array = embeddings_array / norms
            
            print(f"✅ Embeddings OpenAI generati: shape {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            raise RuntimeError(f"Errore durante encoding OpenAI: {e}")
    
    def _request_embeddings_batch(self, texts: List[str], session_ids: List[str] = None) -> List[np.ndarray]:
        """
        Richiede embeddings per batch di testi a OpenAI (con tokenizzazione preventiva)
        
        Args:
            texts: Lista testi da processare (già tokenizzati e troncati)
            session_ids: Lista session_id corrispondenti ai testi (opzionale)
            
        Returns:
            Lista di embedding vectors
        """
        try:
            # Richiesta API OpenAI (nuova sintassi v1.0+)
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            # Estrai embeddings dalla risposta
            embeddings = []
            for item in response.data:
                embedding = np.array(item.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except openai.RateLimitError as e:
            # Gestione rate limit con retry
            print(f"⏰ Rate limit raggiunto, attesa 60 secondi...")
            time.sleep(60)
            return self._request_embeddings_batch(texts, session_ids)
            
        except openai.APIError as e:
            # Log semplificato per errori API (i token limit sono già gestiti preventivamente)
            error_msg = str(e)
            print(f"❌ Errore API OpenAI: {error_msg}")
            
            # Se è ancora un errore di token (non dovrebbe succedere con la tokenizzazione preventiva)
            if "maximum context length" in error_msg or "token" in error_msg.lower():
                print(f"⚠️  ERRORE TOKEN IMPREVISTO (nonostante tokenizzazione preventiva):")
                print(f"   🔍 Numero testi nel batch: {len(texts)}")
                for i, text in enumerate(texts):
                    session_info = f" (Session: {session_ids[i]})" if session_ids and i < len(session_ids) else ""
                    print(f"   📝 Testo {i+1}{session_info}: {len(text)} caratteri")
            
            raise RuntimeError(f"Errore durante encoding OpenAI: Errore API OpenAI: {e}")
        except Exception as e:
            raise RuntimeError(f"Errore processing embedding batch: {e}")
    
    def test_model(self) -> bool:
        """
        Testa il funzionamento del modello OpenAI
        
        Returns:
            True se il test ha successo, False altrimenti
        """
        try:
            print(f"🧪 Test del modello OpenAI {self.model_name}...")
            
            # Testi di test multilingue
            test_texts = [
                "This is an embedding test for OpenAI model",
                "Questo è un test di embedding per il modello OpenAI",
                "Ceci est un test d'embedding pour le modèle OpenAI"
            ]
            
            # Genera embeddings di test
            embeddings = self.encode(test_texts, normalize_embeddings=True)
            
            # Validazioni
            expected_shape = (3, self.embedding_dim)
            if embeddings.shape != expected_shape:
                print(f"❌ Shape embedding errata: {embeddings.shape} != {expected_shape}")
                return False
            
            # Verifica assenza di NaN/infiniti
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                print(f"❌ Embedding contengono NaN o infiniti")
                return False
            
            # Verifica normalizzazione
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                print(f"⚠️  Embedding non normalizzati: norme {norms}")
            
            # Test similarità semantica
            sim_01 = np.dot(embeddings[0], embeddings[1])  # EN-IT dovrebbe essere simile
            sim_02 = np.dot(embeddings[0], embeddings[2])  # EN-FR dovrebbe essere simile
            
            print(f"✅ Test OpenAI completato con successo!")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Norme: {norms}")
            print(f"   Similarità EN-IT: {sim_01:.3f}")
            print(f"   Similarità EN-FR: {sim_02:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test OpenAI fallito: {e}")
            return False
    
    def get_embedding_dim(self) -> int:
        """
        Restituisce la dimensione degli embedding
        
        Returns:
            Dimensione embedding del modello corrente
        """
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """
        Verifica se OpenAI è disponibile
        
        Returns:
            True se disponibile, False altrimenti
        """
        if not OPENAI_AVAILABLE:
            return False
        
        if not self.api_key:
            return False
        
        try:
            # Test rapido di connessione (nuova sintassi v1.0+)
            response = self.client.models.list()
            return True
            
        except Exception:
            return False
    
    def benchmark_speed(self, sample_texts: List[str], iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark delle performance di OpenAI
        
        Args:
            sample_texts: Testi di test per benchmark
            iterations: Numero di iterazioni per media
            
        Returns:
            Metriche di performance
        """
        times = []
        
        print(f"⚠️  Attenzione: il benchmark può consumare crediti OpenAI")
        
        for i in range(iterations):
            start_time = time.time()
            self.encode(sample_texts[:5], show_progress_bar=False)  # Usa solo 5 testi per ridurre costi
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        texts_per_second = 5 / avg_time  # Solo 5 testi usati
        
        return {
            'avg_time_seconds': avg_time,
            'texts_per_second': texts_per_second,
            'total_texts': 5,  # Limitato per costi
            'iterations': iterations,
            'model': f'OpenAI {self.model_name}',
            'note': 'Benchmark limitato per ridurre costi API'
        }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Restituisce la lista dei modelli OpenAI supportati
        
        Returns:
            Lista nomi modelli supportati
        """
        return list(cls.MODEL_DIMENSIONS.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """
        Restituisce informazioni sul modello corrente
        
        Returns:
            Dizionario con informazioni modello
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_tokens': self.max_tokens,
            'provider': 'OpenAI',
            'api_key_set': bool(self.api_key),
            'available': self.is_available()
        }
