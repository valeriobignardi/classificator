#!/usr/bin/env python3
"""
File: clustering_test_service_new.py
Autore: Assistant
Data creazione: 2025-08-25
Descrizione: Servizio per test rapidi di clustering HDBSCAN senza LLM utilizzando pipeline esistente
Storia aggiornamenti: 
2025-08-25 - Creazione iniziale utilizzando pipeline EndToEndPipeline
"""

import sys
import os
import yaml
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
import logging
import gc

# Importazioni per pulizia memoria GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Aggiunta percorsi per moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'EmbeddingEngine'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Clustering'))

from end_to_end_pipeline import EndToEndPipeline
from embedding_manager import embedding_manager
from hdbscan_clusterer import HDBSCANClusterer


class ClusteringTestService:
    """
    Servizio per test rapidi di clustering HDBSCAN senza coinvolgimento LLM
    
    Scopo: Permettere agli utenti di testare rapidamente parametri di clustering
    utilizzando la pipeline esistente senza dover eseguire l'intera pipeline di classificazione.
    
    Args:
        config_path: Percorso del file di configurazione dei tenant
        
    Methods:
        run_clustering_test: Esegue test clustering e restituisce metriche di qualità
        
    Data ultima modifica: 2025-08-25
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il servizio di test clustering utilizzando la pipeline esistente
        
        Args:
            config_path: Percorso del file config.yaml (opzionale)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        self.min_conversations_required = 50
        self._setup_logging()
        
        # Pipeline cache per evitare reinizializzazioni multiple
        self.pipeline_cache = {}
        
    def _setup_logging(self):
        """
        Configura logging per il servizio
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _get_pipeline(self, tenant_id: str) -> EndToEndPipeline:
        """
        Ottiene pipeline per il tenant specificato con embedder dinamico configurato
        
        AGGIORNAMENTO 2025-08-25: Usa EmbeddingManager con tenant_id UUID per consistenza
        
        Args:
            tenant_id: UUID del tenant (es: '16c222a9-f293-11ef-9315-96000228e7fe')
            
        Returns:
            Istanza EndToEndPipeline configurata per tenant
        """
        if tenant_id not in self.pipeline_cache:
            try:
                # SEMPRE USA tenant_id (UUID) per consistenza con embedder manager
                shared_embedder = embedding_manager.get_shared_embedder(tenant_id)
                
                # USA DIRETTAMENTE tenant_id (UUID) - NO CONVERSION!
                # Il tenant_slug verrà risolto internamente dalla pipeline quando necessario
                
                pipeline = EndToEndPipeline(
                    tenant_slug=tenant_id,  # PASSA UUID direttamente - pipeline risolverà slug quando necessario
                    confidence_threshold=0.7,
                    auto_mode=True,
                    shared_embedder=shared_embedder  # Passa embedder dinamico
                )
                self.pipeline_cache[tenant_id] = pipeline  # Cache con UUID come key
                logging.info(f"✅ Pipeline {tenant_id} inizializzata con embedder dinamico e cached")
            except Exception as e:
                logging.error(f"❌ Errore inizializzazione pipeline per UUID {tenant_id}: {e}")
                raise RuntimeError(f"Impossibile inizializzare pipeline per UUID {tenant_id}: {e}")
        
        return self.pipeline_cache[tenant_id]
    
    def _resolve_tenant_slug_from_uuid(self, tenant_uuid: str) -> str:
        """
        Risolve tenant UUID in slug usando database
        
        Args:
            tenant_uuid: UUID del tenant
            
        Returns:
            Slug del tenant o UUID se non trovato
        """
        try:
            from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            tag_connector = TagDatabaseConnector()
            tag_connector.connetti()
            
            query = "SELECT tenant_slug FROM tenants WHERE tenant_id = %s"
            result = tag_connector.esegui_query(query, (tenant_uuid,))
            
            if result and len(result) > 0:
                tenant_slug = result[0][0]
                tag_connector.disconnetti()
                return tenant_slug
            
            tag_connector.disconnetti()
            return tenant_uuid  # fallback
        except Exception as e:
            print(f"⚠️ Errore risoluzione tenant slug per UUID {tenant_uuid}: {e}")
            return tenant_uuid
    
    def load_tenant_clustering_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Carica la configurazione clustering specifica del tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dizionario con parametri clustering del tenant
        """
        try:
            # Cerca configurazione tenant-specifica
            tenant_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tenant_configs')
            tenant_config_file = os.path.join(tenant_config_dir, f'{tenant_id}_clustering.yaml')
            
            if os.path.exists(tenant_config_file):
                print(f"📁 Carico config tenant-specifica: {tenant_config_file}")
                with open(tenant_config_file, 'r', encoding='utf-8') as file:
                    tenant_config = yaml.safe_load(file)
                    return tenant_config.get('clustering_parameters', {})
            
            # Fallback: configurazione globale
            print(f"📁 Uso config globale per tenant {tenant_id}")
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config.get('clustering', {})
                
        except Exception as e:
            print(f"⚠️ Errore caricamento config clustering: {e}")
            # Parametri di default se tutto fallisce
            return {
                'min_cluster_size': 3,
                'min_samples': 2,
                'cluster_selection_epsilon': 0.15,
                'metric': 'cosine'
            }

    def get_sample_conversations(self, tenant: str = 'humanitas', limit: int = 1000) -> Dict[str, Any]:
        """
        Recupera un campione di conversazioni dal database per il tenant utilizzando pipeline
        
        Scopo: Estrarre conversazioni per il test di clustering
        Parametri di input:
            - tenant (str): Nome del tenant/schema database  
            - limit (int): Numero massimo di conversazioni da recuperare
        Valori di ritorno:
            - Dict[str, Any]: Dizionario con sessioni per il clustering
        Data ultima modifica: 2025-08-25
        """
        try:
            logging.info(f"📊 Estrazione {limit} conversazioni per tenant '{tenant}'")
            
            # Ottieni pipeline per il tenant
            pipeline = self._get_pipeline(tenant)
            
            # Estrai sessioni usando la funzione della pipeline
            sessioni = pipeline.estrai_sessioni(limit=limit)
            
            if not sessioni:
                logging.warning(f"❌ Nessuna sessione trovata per tenant '{tenant}'")
                return {}
            
            logging.info(f"✅ Estratte {len(sessioni)} sessioni valide per '{tenant}'")
            return sessioni
            
        except Exception as e:
            logging.error(f"❌ Errore estrazione conversazioni per {tenant}: {e}")
            return {}
    
    def _cleanup_gpu_memory(self):
        """
        Pulisce la memoria GPU per evitare accumulo di memoria tra test consecutivi
        
        Scopo: Libera la memoria GPU utilizzata dai modelli per evitare errori CUDA OOM
        
        Data ultima modifica: 2025-08-25
        """
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Forza la pulizia della cache CUDA
                torch.cuda.empty_cache()
                
                # Garbage collection per liberare oggetti Python
                gc.collect()
                
                # Log memoria GPU libera dopo pulizia
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                    print(f"🧹 Memoria GPU pulita - Allocata: {memory_allocated:.2f}GB, Riservata: {memory_reserved:.2f}GB")
                    
        except Exception as e:
            print(f"⚠️ Errore durante pulizia memoria GPU: {str(e)}")

    def _clear_pipeline_cache(self):
        """
        Pulisce la cache delle pipeline per liberare memoria
        
        Scopo: Evita accumulo di modelli in memoria tra test diversi
        
        Data ultima modifica: 2025-08-25
        """
        try:
            # Libera esplicitamente le pipeline dalla cache
            for tenant_slug, pipeline in self.pipeline_cache.items():
                try:
                    # Se la pipeline ha un embedder, prova a liberarne la memoria
                    if hasattr(pipeline, 'embedder') and hasattr(pipeline.embedder, 'model'):
                        del pipeline.embedder.model
                        print(f"🧹 Modello embedder per {tenant_slug} liberato dalla memoria")
                except Exception as e:
                    print(f"⚠️ Errore liberazione embedder per {tenant_slug}: {e}")
            
            # Pulisce completamente la cache
            self.pipeline_cache.clear()
            print(f"🧹 Cache pipeline pulita")
            
            # Garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ Errore durante pulizia cache pipeline: {str(e)}")

    def clear_pipeline_for_tenant(self, tenant_id: str):
        """
        Pulisce la pipeline specifica per un tenant dalla cache
        
        Scopo:
        Rimuove la pipeline cachata quando l'embedder viene ricaricato,
        evitando riferimenti a embedder con model=None
        
        Args:
            tenant_id: UUID del tenant da rimuovere dalla cache
            
        Data ultima modifica: 2025-01-27
        """
        try:
            if tenant_id in self.pipeline_cache:
                pipeline = self.pipeline_cache[tenant_id]
                
                # Cleanup esplicito del modello embedder se presente
                try:
                    if hasattr(pipeline, 'embedder') and hasattr(pipeline.embedder, 'model'):
                        if pipeline.embedder.model is not None:
                            del pipeline.embedder.model
                        pipeline.embedder.model = None
                        print(f"🧹 Embedder pipeline per tenant {tenant_id} pulito")
                except Exception as e:
                    print(f"⚠️ Errore cleanup embedder pipeline: {e}")
                
                # Rimuove dalla cache
                del self.pipeline_cache[tenant_id]
                print(f"✅ Pipeline per tenant {tenant_id} rimossa dalla cache")
                
                # Garbage collection localizzato
                gc.collect()
                
            else:
                print(f"ℹ️ Nessuna pipeline cachata per tenant {tenant_id}")
                
        except Exception as e:
            print(f"❌ Errore pulizia pipeline per tenant {tenant_id}: {e}")

    def run_clustering_test(self, 
                          tenant_id: str, 
                          custom_parameters: Optional[Dict] = None,
                          sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Esegue test clustering completo utilizzando la pipeline esistente
        
        Args:
            tenant_id: ID del tenant
            custom_parameters: Parametri clustering personalizzati (opzionale)
            sample_size: Dimensione campione (opzionale, default 100)
            
        Returns:
            Risultati completi del test clustering
        """
        start_time = time.time()
        print(f"🚀 Avvio test clustering per tenant {tenant_id}...")
        
        if sample_size is None:
            sample_size = 100
        
        try:
            # 1. Carica configurazione clustering
            if custom_parameters:
                clustering_config = custom_parameters
                print(f"🎛️ Uso parametri personalizzati: {custom_parameters}")
            else:
                clustering_config = self.load_tenant_clustering_config(tenant_id)
                print(f"🎛️ Uso parametri tenant: {clustering_config}")
            
            # 2. Recupera conversazioni campione usando la pipeline
            sessioni = self.get_sample_conversations(tenant_id, sample_size)
            
            if len(sessioni) < self.min_conversations_required:
                return {
                    'success': False,
                    'error': f'Troppe poche conversazioni trovate ({len(sessioni)}). Minimo richiesto: {self.min_conversations_required}',
                    'tenant_id': tenant_id,
                    'execution_time': time.time() - start_time
                }
            
            # 3. Prepara testi per embedding
            texts = []
            session_ids = []
            for session_id, session_data in sessioni.items():
                if 'testo_completo' in session_data and session_data['testo_completo']:
                    texts.append(session_data['testo_completo'])
                    session_ids.append(session_id)
            
            if len(texts) < self.min_conversations_required:
                return {
                    'success': False,
                    'error': f'Troppe poche conversazioni valide trovate ({len(texts)}). Minimo richiesto: {self.min_conversations_required}',
                    'tenant_id': tenant_id,
                    'execution_time': time.time() - start_time
                }
            
            # 4. Genera embeddings
            print(f"🔍 Generazione embeddings per {len(texts)} conversazioni...")
            try:
                # Ottieni pipeline e usa il suo embedder
                pipeline = self._get_pipeline(tenant_id)
                embeddings = pipeline.embedder.encode(texts, show_progress_bar=True)
                print(f"✅ Embeddings generati: {embeddings.shape}")
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Errore generazione embeddings: {str(e)}',
                    'tenant_id': tenant_id,
                    'execution_time': time.time() - start_time
                }
            
            # 5. Esegue clustering HDBSCAN
            print(f"🔗 Avvio clustering HDBSCAN...")
            try:
                clusterer = HDBSCANClusterer(
                    min_cluster_size=clustering_config.get('min_cluster_size', 3),
                    min_samples=clustering_config.get('min_samples', 2),
                    cluster_selection_epsilon=clustering_config.get('cluster_selection_epsilon', 0.15),
                    metric=clustering_config.get('metric', 'cosine')
                )
                
                cluster_labels = clusterer.fit_predict(embeddings)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Errore clustering HDBSCAN: {str(e)}',
                    'tenant_id': tenant_id,
                    'execution_time': time.time() - start_time
                }
            
            # 6. Analizza risultati
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1 = outliers
            n_outliers = int(np.sum(cluster_labels == -1))  # Converti numpy.int64 -> int
            n_clustered = len(texts) - n_outliers
            
            # 7. Calcola metriche di qualità
            quality_metrics = self._calculate_quality_metrics(embeddings, cluster_labels)
            
            # 8. Costruisce cluster dettagliati
            detailed_clusters = self._build_detailed_clusters(texts, session_ids, cluster_labels)
            
            # 9. Analizza outliers
            outlier_analysis = self._analyze_outliers(texts, session_ids, cluster_labels, embeddings)
            
            execution_time = time.time() - start_time
            
            print(f"✅ Clustering completato in {execution_time:.2f}s")
            print(f"📊 Risultati: {n_clusters} clusters, {n_outliers} outliers, {n_clustered} conversazioni clusterizzate")
            
            # 🧹 IMPORTANTE: Pulizia completa memoria dopo il test
            self._clear_pipeline_cache()  # Pulisce prima la cache delle pipeline
            self._cleanup_gpu_memory()    # Poi pulisce la memoria GPU
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'execution_time': execution_time,
                'statistics': {
                    'total_conversations': len(texts),
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'n_clustered': n_clustered,
                    'clustering_ratio': round(n_clustered / len(texts), 3),
                    'parameters_used': clustering_config
                },
                'quality_metrics': quality_metrics,
                'detailed_clusters': detailed_clusters,
                'outlier_analysis': outlier_analysis,
                'recommendations': self._generate_recommendations(
                    len(texts), n_clusters, n_outliers, quality_metrics
                )
            }
            
        except Exception as e:
            # 🧹 Pulizia memoria anche in caso di errore
            self._clear_pipeline_cache()
            self._cleanup_gpu_memory()
            
            return {
                'success': False,
                'error': f'Errore generale nel test clustering: {str(e)}',
                'tenant_id': tenant_id,
                'execution_time': time.time() - start_time
            }
    
    def _calculate_quality_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calcola metriche di qualità del clustering
        
        Args:
            embeddings: Array numpy con gli embeddings
            labels: Array numpy con le etichette dei cluster
            
        Returns:
            Dizionario con metriche di qualità
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Filtra outliers per metriche (non tutte supportano -1)
            mask = labels != -1
            if np.sum(mask) < 2:  # Serve almeno 2 punti non-outlier
                return {
                    'silhouette_score': 0.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': 0.0,
                    'note': 'Troppi pochi punti non-outlier per calcolare metriche'
                }
            
            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]
            
            metrics = {}
            
            # Silhouette Score (più alto = meglio, range [-1, 1])
            if len(np.unique(filtered_labels)) > 1:
                metrics['silhouette_score'] = float(silhouette_score(filtered_embeddings, filtered_labels))
            else:
                metrics['silhouette_score'] = 0.0
                
            # Calinski-Harabasz Index (più alto = meglio)
            if len(np.unique(filtered_labels)) > 1:
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(filtered_embeddings, filtered_labels))
            else:
                metrics['calinski_harabasz_score'] = 0.0
                
            # Davies-Bouldin Index (più basso = meglio)
            if len(np.unique(filtered_labels)) > 1:
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(filtered_embeddings, filtered_labels))
            else:
                metrics['davies_bouldin_score'] = 0.0
            
            return metrics
            
        except ImportError:
            return {
                'note': 'sklearn non disponibile per calcolo metriche di qualità'
            }
        except Exception as e:
            return {
                'error': f'Errore calcolo metriche: {str(e)}'
            }
    
    def _build_detailed_clusters(self, texts: List[str], session_ids: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Costruisce informazioni dettagliate sui cluster
        
        Args:
            texts: Lista dei testi
            session_ids: Lista degli ID sessione  
            labels: Array numpy con le etichette dei cluster
            
        Returns:
            Dizionario con dettagli dei cluster
        """
        cluster_groups = defaultdict(list)
        
        for i, label in enumerate(labels):
            if label != -1:  # Ignora outliers per i cluster
                cluster_groups[int(label)].append({
                    'session_id': session_ids[i],
                    'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                    'text_length': len(texts[i])
                })
        
        # Ordina cluster per dimensione (decrescente)
        sorted_clusters = []
        for cluster_id, conversations in sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True):
            sorted_clusters.append({
                'cluster_id': cluster_id,
                'size': len(conversations),
                'conversations': conversations[:10],  # Mostra solo i primi 10 per brevità
                'total_conversations': len(conversations)
            })
        
        return {
            'clusters': sorted_clusters,
            'total_clusters': len(cluster_groups)
        }
    
    def _analyze_outliers(self, texts: List[str], session_ids: List[str], 
                         labels: np.ndarray, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Analizza le conversazioni classificate come outliers
        
        Args:
            texts: Lista dei testi
            session_ids: Lista degli ID sessione
            labels: Array numpy con le etichette dei cluster
            embeddings: Array numpy con gli embeddings
            
        Returns:
            Dizionario con analisi degli outliers
        """
        outlier_indices = np.where(labels == -1)[0]
        
        if len(outlier_indices) == 0:
            return {
                'count': 0,
                'percentage': 0.0,
                'samples': [],
                'note': 'Nessun outlier trovato'
            }
        
        outliers = []
        for i in outlier_indices[:10]:  # Mostra solo i primi 10
            outliers.append({
                'session_id': session_ids[i],
                'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                'text_length': len(texts[i])
            })
        
        return {
            'count': len(outlier_indices),
            'percentage': round((len(outlier_indices) / len(texts)) * 100, 2),
            'samples': outliers,
            'total_outliers': len(outlier_indices)
        }
    
    def _generate_recommendations(self, n_conversations: int, n_clusters: int, 
                                n_outliers: int, quality_metrics: Dict) -> List[str]:
        """
        Genera raccomandazioni basate sui risultati del clustering
        
        Args:
            n_conversations: Numero totale conversazioni
            n_clusters: Numero di cluster trovati
            n_outliers: Numero di outliers
            quality_metrics: Metriche di qualità
            
        Returns:
            Lista di raccomandazioni testuali
        """
        recommendations = []
        
        # Analisi numero cluster
        cluster_ratio = n_clusters / n_conversations if n_conversations > 0 else 0
        if cluster_ratio > 0.5:
            recommendations.append("⚠️ Troppi cluster piccoli. Considera di aumentare min_cluster_size.")
        elif cluster_ratio < 0.05:
            recommendations.append("⚠️ Troppi pochi cluster. Considera di diminuire min_cluster_size.")
        else:
            recommendations.append("✅ Numero di cluster appropriato per il dataset.")
        
        # Analisi outliers
        outlier_ratio = n_outliers / n_conversations if n_conversations > 0 else 0
        if outlier_ratio > 0.3:
            recommendations.append("⚠️ Molti outliers (>30%). Considera di diminuire min_cluster_size o min_samples.")
        elif outlier_ratio < 0.05:
            recommendations.append("⚠️ Pochi outliers (<5%). Potresti avere cluster troppo inclusivi.")
        else:
            recommendations.append("✅ Percentuale di outliers bilanciata.")
        
        # Analisi qualità (se disponibili)
        if 'silhouette_score' in quality_metrics:
            silhouette = quality_metrics['silhouette_score']
            if silhouette > 0.5:
                recommendations.append("✅ Buona separazione dei cluster (silhouette > 0.5).")
            elif silhouette < 0.2:
                recommendations.append("⚠️ Cluster poco separati (silhouette < 0.2). Rivedi i parametri.")
            else:
                recommendations.append("⏸️ Qualità cluster moderata. Potresti ottimizzare i parametri.")
        
        return recommendations
