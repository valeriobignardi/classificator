"""
Implementazione LaBSE per embedding multilingue con focus italiano
"""

import os
import sys
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Aggiunge il path per importare la base class
sys.path.append(os.path.dirname(__file__))
from base_embedder import BaseEmbedder

# Import TokenizationManager per gestione conversazioni lunghe
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
try:
    from tokenization_utils import TokenizationManager
    TOKENIZATION_AVAILABLE = True
except ImportError:
    TokenizationManager = None
    TOKENIZATION_AVAILABLE = False

class LaBSEEmbedder(BaseEmbedder):
    """
    Embedder basato su LaBSE (Language-agnostic BERT Sentence Embedding)
    Ottimizzato per testi multilingue con eccellente supporto italiano
    """
    
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: Optional[str] = None, test_on_init: bool = True):
        """
        Inizializza il modello LaBSE
        
        Args:
            model_name: Nome del modello su HuggingFace
            device: Device per l'inferenza ('cuda', 'cpu', None per auto-detect)
            test_on_init: Se eseguire un test del modello dopo l'inizializzazione
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = 768
        
        # Auto-detect device se non specificato
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Inizializza TokenizationManager per gestione conversazioni lunghe
        self.tokenizer = None
        if TOKENIZATION_AVAILABLE:
            try:
                self.tokenizer = TokenizationManager()
                print(f"✅ TokenizationManager integrato in LaBSE per gestione conversazioni lunghe")
            except Exception as e:
                print(f"⚠️  Errore inizializzazione TokenizationManager in LaBSE: {e}")
                self.tokenizer = None
        else:
            print(f"⚠️  TokenizationManager non disponibile per LaBSE")
            
        print(f"🚀 Inizializzazione LaBSE embedder su device: {self.device}")
        self.load_model()
        
        # Test automatico del modello se richiesto
        if test_on_init:
            if not self.test_model():
                print(f"⚠️  Attenzione: il modello potrebbe non funzionare correttamente")
            else:
                print(f"🎯 Modello pronto per l'uso!")
    
    def load_model(self):
        """Carica il modello LaBSE direttamente sul device target"""
        try:
            print(f"📥 Caricamento modello {self.model_name}...")
            
            # Tentativo di caricamento diretto sul device target
            if self.device == "cuda" and torch.cuda.is_available():
                print(f"🚀 Caricamento diretto su GPU...")
                
                # Per evitare problemi meta tensor, usiamo parametri specifici
                try:
                    # Caricamento con parametri ottimizzati per GPU
                    self.model = SentenceTransformer(
                        self.model_name, 
                        device=self.device,
                        trust_remote_code=True  # Necessario per alcuni modelli più recenti
                    )
                    
                except Exception as gpu_error:
                    print(f"⚠️  Caricamento diretto su GPU fallito: {gpu_error}")
                    print(f"🔄 Fallback: carico su CPU e trasferisco su GPU...")
                    
                    # Fallback: carica su CPU e trasferisci
                    self.model = SentenceTransformer(self.model_name, device="cpu")
                    self.model = self.model.to(self.device)
                
            else:
                # Caricamento diretto su CPU
                print(f"💻 Caricamento diretto su CPU...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
            
            print(f"✅ Modello caricato con successo!")
            print(f"📊 Dimensione embedding: {self.embedding_dim}")
            print(f"🔧 Device: {self.device}")
            
            if torch.cuda.is_available() and self.device == "cuda":
                print(f"🎮 GPU: {torch.cuda.get_device_name()}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                print(f"📈 GPU Memory utilizzata: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Verifica che il modello sia effettivamente su GPU
                try:
                    first_param = next(self.model.parameters())
                    print(f"� Primo parametro su device: {first_param.device}")
                except:
                    print(f"🔍 Modello caricato, device verification skipped")
                
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello LaBSE: {e}")
            print(f"🔄 Tentativo di fallback completo su CPU...")
            
            try:
                # Fallback completo su CPU
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device="cpu")
                print(f"✅ Modello caricato con successo su CPU (fallback)")
                
            except Exception as e2:
                raise Exception(f"Errore critico nel caricamento del modello LaBSE anche su CPU: {e2}")
    
    def encode(self, texts: Union[str, List[str]], 
               normalize_embeddings: bool = True,
               batch_size: int = 32,
               show_progress_bar: bool = False,
               session_ids: List[str] = None,  # Nuovo parametro per session_ids
               **kwargs) -> np.ndarray:
        """
        Converte testo(i) in embedding(s) con gestione errori migliorata
        e tokenizzazione preventiva per conversazioni lunghe
        
        Args:
            texts: Singolo testo o lista di testi
            normalize_embeddings: Se normalizzare gli embedding (raccomandato)
            batch_size: Dimensione del batch per l'inferenza
            show_progress_bar: Mostra progress bar per batch grandi
            session_ids: Lista degli ID di sessione corrispondenti ai testi (opzionale)
            **kwargs: Parametri aggiuntivi per SentenceTransformer
            
        Returns:
            Array numpy con gli embedding normalizzati
        """
        if self.model is None:
            raise Exception("Modello non caricato. Chiamare load_model() prima.")
        
        try:
            # Converte singolo testo in lista
            if isinstance(texts, str):
                texts = [texts]
            
            print(f"🔍 Encoding {len(texts)} testi con LaBSE su device {self.device}...")
            
            # ========================================================================
            # 🔥 TOKENIZZAZIONE PREVENTIVA PER LABSE 
            # ========================================================================
            
            processed_texts = texts
            
            if self.tokenizer:
                print(f"\n🔍 TOKENIZZAZIONE PREVENTIVA LABSE CLUSTERING")
                print(f"=" * 60)
                
                try:
                    processed_texts, tokenization_stats = self.tokenizer.process_conversations_for_clustering(
                        texts, session_ids
                    )
                    
                    print(f"✅ Tokenizzazione LaBSE completata:")
                    print(f"   📊 Conversazioni processate: {tokenization_stats['total_conversations']}")
                    print(f"   📊 Conversazioni troncate: {tokenization_stats['truncated_count']}")
                    print(f"   📊 Token limite configurato: {self.tokenizer.max_tokens}")
                    if tokenization_stats['truncated_count'] > 0:
                        print(f"   ✂️  {tokenization_stats['truncated_count']} conversazioni TRONCATE per rispettare limite token")
                    else:
                        print(f"   ✅ Tutte le conversazioni entro i limiti, nessun troncamento")
                    print(f"=" * 60)
                    
                except Exception as e:
                    print(f"⚠️  Errore durante tokenizzazione LaBSE: {e}")
                    print(f"🔄 Fallback a preprocessing legacy")
                    processed_texts = texts
            else:
                print(f"⚠️  TokenizationManager non disponibile per LaBSE")
                print(f"� Usando preprocessing legacy")
            
            # Preprocessing legacy: rimuove testi vuoti
            final_processed_texts = []
            for text in processed_texts:
                if text and text.strip():
                    final_processed_texts.append(text.strip())
                else:
                    final_processed_texts.append("testo vuoto")  # Placeholder per testi vuoti
            
            # Verifica stato del modello prima dell'encoding
            if hasattr(self.model, '_modules') and self.device == "cuda":
                # Verifica che il modello sia ancora su GPU
                try:
                    first_param_device = next(self.model.parameters()).device
                    # Normalizza device comparison per evitare cuda:0 vs cuda
                    if str(first_param_device).split(':')[0] != self.device.split(':')[0]:
                        print(f"⚠️  Rilevato cambio device: {first_param_device} -> {self.device}")
                        self.model = self.model.to(self.device)
                except:
                    pass  # Ignore se non ci sono parametri
            
            # Genera embedding con gestione memory-efficient per GPU
            if self.device == "cuda":
                # Per GPU, usiamo batch più piccoli per evitare OOM
                safe_batch_size = min(batch_size, 16)  # Limita batch size su GPU
                with torch.amp.autocast('cuda'):  # Mixed precision per efficienza
                    embeddings = self.model.encode(
                        final_processed_texts,
                        normalize_embeddings=normalize_embeddings,
                        batch_size=safe_batch_size,
                        show_progress_bar=show_progress_bar,
                        convert_to_numpy=True,
                        **kwargs
                    )
            else:
                # Per CPU, usiamo il batch size originale
                embeddings = self.model.encode(
                    final_processed_texts,
                    normalize_embeddings=normalize_embeddings,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    **kwargs
                )
            
            print(f"✅ Embedding generati: shape {embeddings.shape}")
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"💾 GPU Out of Memory durante encoding: {e}")
            print(f"🔄 Tentativo di fallback su CPU...")
            
            # Fallback temporaneo su CPU
            original_device = self.device
            try:
                self.model = self.model.to("cpu")
                self.device = "cpu"
                
                embeddings = self.model.encode(
                    processed_texts,
                    normalize_embeddings=normalize_embeddings,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    **kwargs
                )
                
                print(f"✅ Embedding generati su CPU (fallback): shape {embeddings.shape}")
                
                # Ripristina device originale per il prossimo uso
                if original_device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(original_device)
                    self.device = original_device
                    
                return embeddings
                
            except Exception as fallback_error:
                raise Exception(f"Errore sia su GPU che su CPU: {fallback_error}")
                
        except Exception as e:
            raise Exception(f"Errore durante l'encoding: {e}")
    
    def encode_single(self, text: str, **kwargs) -> np.ndarray:
        """
        Converte un singolo testo in embedding
        
        Args:
            text: Testo da convertire
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Array numpy con l'embedding del testo
        """
        embeddings = self.encode([text], **kwargs)
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Restituisce la dimensione degli embedding"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """
        Restituisce informazioni sul modello
        
        Returns:
            Dizionario con informazioni del modello
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'N/A'),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
    
    def benchmark_speed(self, sample_texts: List[str], iterations: int = 3) -> dict:
        """
        Benchmark delle prestazioni del modello
        
        Args:
            sample_texts: Testi di esempio per il benchmark
            iterations: Numero di iterazioni per il test
            
        Returns:
            Dizionario con metriche di performance
        """
        import time
        
        print(f"🏃 Benchmark speed con {len(sample_texts)} testi, {iterations} iterazioni...")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            _ = self.encode(sample_texts, show_progress_bar=False)
            end_time = time.time()
            iteration_time = end_time - start_time
            times.append(iteration_time)
            print(f"  Iterazione {i+1}: {iteration_time:.3f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        texts_per_second = len(sample_texts) / avg_time
        
        results = {
            "avg_time_seconds": avg_time,
            "std_time_seconds": std_time,
            "texts_per_second": texts_per_second,
            "total_texts": len(sample_texts),
            "iterations": iterations
        }
        
        print(f"📊 Risultati benchmark:")
        print(f"  Tempo medio: {avg_time:.3f}s (±{std_time:.3f}s)")
        print(f"  Throughput: {texts_per_second:.1f} testi/secondo")
        
        return results
    
    def test_model(self) -> bool:
        """
        Test rapido per verificare che il modello funzioni correttamente
        
        Returns:
            True se il modello funziona, False altrimenti
        """
        try:
            print("🧪 Test del modello LaBSE...")
            
            # Test con frasi semplici
            test_texts = [
                "Ciao, come stai?",
                "Hello, how are you?",
                "Bonjour, comment allez-vous?"
            ]
            
            embeddings = self.encode(test_texts, show_progress_bar=False)
            
            # Verifica dimensioni
            expected_shape = (len(test_texts), self.embedding_dim)
            if embeddings.shape != expected_shape:
                print(f"❌ Shape errata: {embeddings.shape} vs {expected_shape}")
                return False
            
            # Verifica che non ci siano NaN o infiniti
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                print(f"❌ Embedding contengono NaN o infiniti")
                return False
            
            # Verifica che gli embedding siano normalizzati (circa norma 1)
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                print(f"⚠️  Embedding non normalizzati correttamente: norme {norms}")
            
            print(f"✅ Test modello completato con successo!")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Norme: {norms}")
            print(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test modello fallito: {e}")
            return False
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calcola similarità semantica tra due testi usando embedding
        
        Scopo della funzione: Fornisce interfaccia per calcolo similarità semantica
        Parametri di input: text1, text2 (stringhe da confrontare)
        Parametri di output: float (similarità coseno tra -1 e 1)
        Valori di ritorno: Valore di similarità, 0.0 in caso di errore
        Tracciamento aggiornamenti: 2025-08-28 - Aggiunto per compatibilità AltroTagValidator
        
        Args:
            text1: Primo testo da confrontare
            text2: Secondo testo da confrontare
            
        Returns:
            Similarità coseno tra i due testi (0.0-1.0)
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        try:
            if not text1 or not text2:
                return 0.0
                
            # Genera embedding per entrambi i testi
            emb1 = self.encode([text1])[0]
            emb2 = self.encode([text2])[0] 
            
            # Calcola similarità coseno usando il metodo della classe base
            similarity = self.cosine_similarity(emb1, emb2)
            
            # Normalizza a range [0, 1] per compatibilità 
            return max(0.0, float(similarity))
            
        except Exception as e:
            print(f"⚠️ Errore calcolo similarità tra '{text1[:50]}...' e '{text2[:50]}...': {e}")
            return 0.0
