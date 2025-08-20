"""
Clustering HDBSCAN per scoperta automatica di categorie
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import yaml

# Aggiunge i percorsi per importare gli altri moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Preprocessing'))

from labse_embedder import LaBSEEmbedder
from session_aggregator import SessionAggregator

class HDBSCANClusterer:
    """
    Clustering delle sessioni usando HDBSCAN per scoperta automatica di categorie
    """
    
    def __init__(self, 
                 min_cluster_size: Optional[int] = None,
                 min_samples: Optional[int] = None,
                 cluster_selection_epsilon: Optional[float] = None,
                 metric: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Inizializza il clusterer HDBSCAN con parametri da configurazione
        
        Args:
            min_cluster_size: Dimensione minima dei cluster (sovrascrive config)
            min_samples: Numero minimo di campioni (sovrascrive config)
            cluster_selection_epsilon: Distanza massima per cluster (sovrascrive config)
            metric: Metrica di distanza (sovrascrive config)
            config_path: Percorso del file di configurazione
        """
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.config_path = config_path  # Salva il percorso per uso futuro
        self.config = self._load_config(config_path)
        
        # Usa parametri da config se non specificati esplicitamente
        clustering_config = self.config.get('clustering', {})
        
        self.min_cluster_size = min_cluster_size or clustering_config.get('min_cluster_size', 5)
        self.min_samples = min_samples or clustering_config.get('min_samples', 3)
        self.cluster_selection_epsilon = cluster_selection_epsilon or clustering_config.get('cluster_selection_epsilon', 0.05)
        self.metric = metric or clustering_config.get('metric', 'cosine')
        
        # Parametri avanzati da config
        self.cluster_selection_method = clustering_config.get('cluster_selection_method', 'leaf')
        self.allow_single_cluster = clustering_config.get('allow_single_cluster', False)
        self.n_representatives = clustering_config.get('n_representatives', 3)
        self.min_silhouette_score = clustering_config.get('min_silhouette_score', 0.2)
        self.max_outlier_ratio = clustering_config.get('max_outlier_ratio', 0.7)
        
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_probabilities = None
        self.outlier_scores = None
        
        print(f"🔧 HDBSCANClusterer inizializzato con parametri:")
        print(f"   min_cluster_size: {self.min_cluster_size}")
        print(f"   min_samples: {self.min_samples}")
        print(f"   metric: {self.metric}")
        print(f"   cluster_selection_epsilon: {self.cluster_selection_epsilon}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carica la configurazione dal file YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"⚠️ Errore nel caricamento config da {config_path}: {e}")
            print("📝 Uso parametri predefiniti")
            return {}
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Esegue il clustering sugli embedding
        
        Args:
            embeddings: Array di embedding (n_samples, embedding_dim)
            
        Returns:
            Array delle etichette dei cluster (-1 per outlier)
        """
        print(f"🔍 Clustering con HDBSCAN su {embeddings.shape[0]} embedding...")
        print(f"⚙️  Parametri: min_cluster_size={self.min_cluster_size}, "
              f"min_samples={self.min_samples}, metric={self.metric}")
        
        # Normalizza gli embedding se necessario
        if self.metric == 'cosine':
            # Per coseno, usa euclidean su embedding normalizzati
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            metric_for_hdbscan = 'euclidean'
        else:
            embeddings_norm = embeddings
            metric_for_hdbscan = self.metric
        
        # Inizializza e esegue clustering con parametri da configurazione
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=metric_for_hdbscan,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster
        )
        
        self.cluster_labels = self.clusterer.fit_predict(embeddings_norm)
        self.cluster_probabilities = self.clusterer.probabilities_
        self.outlier_scores = self.clusterer.outlier_scores_
        
        # Statistiche clustering
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_outliers = list(self.cluster_labels).count(-1)
        
        print(f"✅ Clustering completato!")
        print(f"📊 Cluster trovati: {n_clusters}")
        print(f"🔍 Outlier: {n_outliers}")
        print(f"📈 Silhouette score: {self._calculate_silhouette_score(embeddings_norm):.3f}")
        
        return self.cluster_labels
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray) -> float:
        """Calcola il silhouette score per valutare la qualità del clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(self.cluster_labels)) > 1:
                # Esclude outlier dal calcolo
                mask = self.cluster_labels != -1
                if mask.sum() > 1:
                    return silhouette_score(embeddings[mask], self.cluster_labels[mask])
            return 0.0
        except:
            return 0.0
    
    def get_cluster_statistics(self) -> Dict:
        """
        Restituisce statistiche dettagliate sui cluster
        
        Returns:
            Dizionario con statistiche sui cluster
        """
        if self.cluster_labels is None:
            return {}
        
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        stats = {
            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'n_outliers': counts[unique_labels == -1][0] if -1 in unique_labels else 0,
            'cluster_sizes': {},
            'cluster_probabilities_avg': {},
            'total_samples': len(self.cluster_labels)
        }
        
        for label, count in zip(unique_labels, counts):
            if label != -1:  # Esclude outlier
                stats['cluster_sizes'][int(label)] = int(count)
                
                # Probabilità media per questo cluster
                mask = self.cluster_labels == label
                avg_prob = np.mean(self.cluster_probabilities[mask])
                stats['cluster_probabilities_avg'][int(label)] = float(avg_prob)
        
        return stats
    
    def get_cluster_representatives(self, 
                                 embeddings: np.ndarray, 
                                 session_data: Dict[str, Dict],
                                 n_representatives: Optional[int] = None) -> Dict[int, List[Dict]]:
        """
        Trova i rappresentanti più centrali per ogni cluster
        
        Args:
            embeddings: Array di embedding
            session_data: Dati delle sessioni con session_id come chiave
            n_representatives: Numero di rappresentanti per cluster (usa config se None)
            
        Returns:
            Dizionario con cluster_id -> lista di rappresentanti
        """
        if self.cluster_labels is None:
            return {}
        
        # Usa parametro da config se non specificato
        if n_representatives is None:
            n_representatives = self.n_representatives
        
        print(f"🎯 Ricerca rappresentanti per ogni cluster...")
        
        representatives = {}
        session_ids = list(session_data.keys())
        
        unique_clusters = set(self.cluster_labels)
        unique_clusters.discard(-1)  # Rimuove outlier
        
        for cluster_id in unique_clusters:
            # Trova tutti i punti di questo cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Calcola il centroide del cluster
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Trova i punti più vicini al centroide
            distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances_to_centroid)[:n_representatives]
            
            # Recupera i dati delle sessioni rappresentative
            cluster_representatives = []
            for idx in closest_indices:
                original_idx = cluster_indices[idx]
                session_id = session_ids[original_idx]
                session_info = session_data[session_id].copy()
                session_info['distance_to_centroid'] = float(distances_to_centroid[idx])
                session_info['cluster_probability'] = float(self.cluster_probabilities[original_idx])
                cluster_representatives.append(session_info)
            
            representatives[int(cluster_id)] = cluster_representatives
        
        print(f"✅ Trovati rappresentanti per {len(representatives)} cluster")
        return representatives
    
    def visualize_clusters(self, 
                          embeddings: np.ndarray, 
                          session_data: Dict[str, Dict],
                          method: str = 'tsne',
                          save_path: Optional[str] = None) -> None:
        """
        Visualizza i cluster in 2D
        
        Args:
            embeddings: Array di embedding
            session_data: Dati delle sessioni
            method: Metodo di riduzione dimensionale ('tsne' o 'pca')
            save_path: Percorso per salvare il grafico
        """
        if self.cluster_labels is None:
            print("❌ Eseguire prima il clustering")
            return
        
        print(f"📊 Visualizzazione cluster con {method.upper()}...")
        
        # Riduzione dimensionale
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Outlier in nero
                mask = self.cluster_labels == label
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c='black', marker='x', s=50, alpha=0.6, label='Outlier')
            else:
                mask = self.cluster_labels == label
                plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                          c=[color], s=60, alpha=0.8, label=f'Cluster {label}')
        
        plt.title(f'Visualizzazione Cluster ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Grafico salvato in: {save_path}")
        
        plt.show()
    
    def suggest_cluster_labels(self, representatives: Dict[int, List[Dict]]) -> Dict[int, str]:
        """
        Suggerisce etichette per i cluster basandosi sui rappresentanti
        
        Args:
            representatives: Rappresentanti per ogni cluster
            
        Returns:
            Dizionario con cluster_id -> etichetta suggerita
        """
        print(f"🏷️  Generazione etichette suggerite per i cluster...")
        
        suggested_labels = {}
        
        # Pattern per riconoscere categorie comuni
        patterns = {
            'accesso_portale': [
                'accesso', 'login', 'password', 'portale', 'app', 'non riesco', 'errore'
            ],
            'prenotazione_esami': [
                'prenotare', 'prenotazione', 'esame', 'visita', 'appuntamento'
            ],
            'ritiro_referti': [
                'referto', 'referti', 'ritirare', 'risultati', 'esito'
            ],
            'problemi_tecnici': [
                'errore', 'bug', 'non funziona', 'problema tecnico', 'malfunzionamento'
            ],
            'fatturazione': [
                'fattura', 'pagamento', 'costo', 'prezzo', 'ricevuta'
            ],
            'informazioni_generali': [
                'informazioni', 'orari', 'dove', 'come', 'quando'
            ]
        }
        
        for cluster_id, cluster_reps in representatives.items():
            # Combina tutti i testi dei rappresentanti
            combined_text = ' '.join([rep['testo_completo'].lower() for rep in cluster_reps])
            
            # Conta le occorrenze per ogni categoria
            category_scores = {}
            for category, keywords in patterns.items():
                score = sum(combined_text.count(keyword) for keyword in keywords)
                if score > 0:
                    category_scores[category] = score
            
            # Scegli la categoria con il punteggio più alto
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                suggested_labels[cluster_id] = best_category.replace('_', ' ').title()
            else:
                # Fallback: usa parole più frequenti
                words = combined_text.split()
                word_freq = Counter([w for w in words if len(w) > 3])
                if word_freq:
                    top_words = [word for word, _ in word_freq.most_common(3)]
                    suggested_labels[cluster_id] = f"Argomento: {', '.join(top_words)}"
                else:
                    suggested_labels[cluster_id] = f"Cluster {cluster_id}"
        
        print(f"✅ Etichette suggerite generate per {len(suggested_labels)} cluster")
        return suggested_labels

# Test del clustering
if __name__ == "__main__":
    print("=== TEST HDBSCAN CLUSTERING ===\n")
    
    # Inizializza componenti
    embedder = LaBSEEmbedder()
    aggregator = SessionAggregator(schema='humanitas')
    clusterer = HDBSCANClusterer(min_cluster_size=3, min_samples=2)
    
    try:
        # Estrai sessioni
        print("📊 Estrazione sessioni per clustering...")
        sessioni = aggregator.estrai_sessioni_aggregate(limit=200)  # Più dati per clustering
        sessioni_filtrate = aggregator.filtra_sessioni_vuote(sessioni)
        
        if len(sessioni_filtrate) < 10:
            print("⚠️  Troppo poche sessioni per clustering significativo")
            exit()
        
        # Genera embedding
        testi = [dati['testo_completo'] for dati in sessioni_filtrate.values()]
        print(f"🔍 Generazione embedding per {len(testi)} sessioni...")
        embeddings = embedder.encode(testi, show_progress_bar=True)
        
        # Clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Statistiche
        stats = clusterer.get_cluster_statistics()
        print(f"\n📊 STATISTICHE CLUSTERING:")
        print(f"  Cluster trovati: {stats['n_clusters']}")
        print(f"  Outlier: {stats['n_outliers']}")
        print(f"  Dimensioni cluster: {stats['cluster_sizes']}")
        
        # Rappresentanti
        representatives = clusterer.get_cluster_representatives(
            embeddings, sessioni_filtrate, n_representatives=2
        )
        
        # Etichette suggerite
        suggested_labels = clusterer.suggest_cluster_labels(representatives)
        
        # Mostra risultati
        print(f"\n🎯 CLUSTER E RAPPRESENTANTI:")
        print("=" * 80)
        
        for cluster_id, reps in representatives.items():
            label = suggested_labels.get(cluster_id, f"Cluster {cluster_id}")
            print(f"\n🏷️  CLUSTER {cluster_id}: {label}")
            print(f"📊 Dimensione: {stats['cluster_sizes'][cluster_id]} sessioni")
            print(f"🎯 Probabilità media: {stats['cluster_probabilities_avg'][cluster_id]:.3f}")
            
            for i, rep in enumerate(reps, 1):
                print(f"\n  📝 Rappresentante {i}:")
                print(f"    Sessione: {rep['session_id']}")
                print(f"    Distanza dal centro: {rep['distance_to_centroid']:.3f}")
                print(f"    Testo: {rep['testo_completo'][:150]}...")
                print(f"    {'─' * 40}")
        
    finally:
        aggregator.chiudi_connessione()
