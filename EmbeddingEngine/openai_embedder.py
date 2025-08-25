#!/usr/bin/env python3
"""
File: openai_embedder.py
Autore: GitHub Copilot
Data creazione: 2025-08-25
Descrizione: Engine di embedding OpenAI (text-embedding-003-large/medium) per configurazione AI

Storia aggiornamenti:
2025-08-25 - Creazione iniziale con supporto modelli OpenAI embedding
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
        Inizializza OpenAI embedder
        
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
               **kwargs) -> np.ndarray:
        """
        Genera embeddings per i testi usando modello OpenAI
        
        Args:
            texts: Testo singolo o lista di testi
            normalize_embeddings: Se normalizzare gli embeddings
            batch_size: Dimensione batch (max 100 per OpenAI)
            show_progress_bar: Mostra barra progresso
            
        Returns:
            Array numpy con embeddings shape (n_samples, embedding_dim)
        """
        # Normalizza input
        if isinstance(texts, str):
            texts = [texts]
        
        print(f"🔍 Encoding {len(texts)} testi con OpenAI {self.model_name}...")
        
        embeddings = []
        
        try:
            # Processa in batch
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                if show_progress_bar:
                    print(f"📊 Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Preprocessing testi con gestione token limit
                processed_texts = []
                for idx, text in enumerate(batch_texts):
                    if not text or not text.strip():
                        processed_texts.append("testo vuoto")
                    else:
                        # Usa la nuova logica di truncamento
                        text = text.strip()
                        truncated_text = self._truncate_text_to_tokens(text, max_tokens=7500)  # Margine sicurezza
                        processed_texts.append(truncated_text)
                
                # Richiesta embedding batch a OpenAI
                batch_embeddings = self._request_embeddings_batch(processed_texts)
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
    
    def _request_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Richiede embeddings per batch di testi a OpenAI
        
        Args:
            texts: Lista testi da processare
            
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
            return self._request_embeddings_batch(texts)
            
        except openai.APIError as e:
            # Log dettagliato per errori di token limit
            error_msg = str(e)
            if "maximum context length" in error_msg or "token" in error_msg.lower():
                print(f"🚨 ERRORE TOKEN LIMIT OPENAI:")
                print(f"   Errore: {error_msg}")
                print(f"   Numero testi nel batch: {len(texts)}")
                
                # Mostra i testi più lunghi che potrebbero aver causato l'errore
                for i, text in enumerate(texts):
                    estimated_tokens = self._estimate_tokens(text)
                    if estimated_tokens > 7000:  # Soglia sospetta
                        print(f"   📝 Testo {i+1} sospetto ({estimated_tokens} token stimati):")
                        print(f"      Inizio: '{text[:300]}...'")
                        print(f"      Fine: '...{text[-300:]}'")
                        print(f"      Lunghezza: {len(text)} caratteri")
                        print(f"      " + "="*80)
            
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
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Stima approssimativa del numero di token per un testo
        
        Args:
            text: Testo da stimare
            
        Returns:
            Numero stimato di token
            
        Ultima modifica: 2025-08-25
        """
        # Approssimazione: ~4 caratteri per token in italiano/inglese
        # Include spazi, punteggiatura, ecc.
        return len(text) // 4 + 1
    
    def _truncate_text_to_tokens(self, text: str, max_tokens: int = 8000) -> str:
        """
        Tronca un testo per rispettare il limite di token
        
        Args:
            text: Testo da troncare
            max_tokens: Limite massimo di token
            
        Returns:
            Testo troncato
            
        Ultima modifica: 2025-08-25
        """
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
            
        print(f"⚠️  Testo troppo lungo: {estimated_tokens} token stimati (max: {max_tokens})")
        print(f"📝 Inizio conversazione problematica: '{text[:200]}...'")
        print(f"📝 Fine conversazione problematica: '...{text[-200:]}'")
        
        # Tronca mantenendo circa max_tokens caratteri
        max_chars = max_tokens * 4
        truncated = text[:max_chars]
        
        # Cerca di terminare su una parola completa
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.9:  # Se lo spazio è vicino alla fine
            truncated = truncated[:last_space]
        
        print(f"✂️  Testo troncato da {len(text)} a {len(truncated)} caratteri")
        return truncated
    
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
