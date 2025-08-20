"""
Base interface per i motori di embedding
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseEmbedder(ABC):
    """
    Interfaccia astratta per tutti i motori di embedding
    """
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.embedding_dim = None
    
    @abstractmethod
    def load_model(self):
        """Carica il modello di embedding"""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Converte testo(i) in embedding(s)
        
        Args:
            texts: Singolo testo o lista di testi
            **kwargs: Parametri aggiuntivi per l'encoding
            
        Returns:
            Array numpy con gli embedding
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Restituisce la dimensione degli embedding"""
        pass
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcola la similarità coseno tra due embedding
        
        Args:
            emb1: Primo embedding
            emb2: Secondo embedding
            
        Returns:
            Valore di similarità coseno tra -1 e 1
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def batch_cosine_similarity(self, embeddings: np.ndarray, target_embedding: np.ndarray) -> np.ndarray:
        """
        Calcola la similarità coseno tra un embedding target e un batch di embedding
        
        Args:
            embeddings: Array di embedding (n_samples, embedding_dim)
            target_embedding: Embedding target (embedding_dim,)
            
        Returns:
            Array delle similarità coseno
        """
        # Normalizza gli embedding
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        target_norm = target_embedding / np.linalg.norm(target_embedding)
        
        # Calcola similarità coseno
        similarities = np.dot(embeddings_norm, target_norm)
        return similarities
