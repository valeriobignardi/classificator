"""
Autore: GitHub Copilot
Data di creazione: 26 Agosto 2025
Ultima modifica: 26 Agosto 2025

Sistema di visualizzazione avanzato per cluster con grafici 2D/3D interattivi
e statistiche complete per PARAMETRI CLUSTERING e STATISTICHE
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Any, Optional, Union
import json
import os
from datetime import datetime
from collections import Counter
import logging

class ClusterVisualizationManager:
    """
    Sistema completo per visualizzazione cluster con grafici 2D/3D interattivi
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inizializza il manager di visualizzazione
        
        Args:
            output_dir: Directory per salvare i grafici (default: ./cluster_visualizations)
        """
        self.output_dir = output_dir or "./cluster_visualizations"
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Crea directory se non esiste
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_clustering_parameters(self,
                                     embeddings: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     cluster_info: Dict[int, Dict[str, Any]],
                                     session_texts: List[str] = None,
                                     save_html: bool = True,
                                     show_console: bool = True) -> Dict[str, Any]:
        """
        Visualizzazione completa per PARAMETRI CLUSTERING (senza etichette finali)
        
        Args:
            embeddings: Array embeddings
            cluster_labels: Etichette cluster
            cluster_info: Info dettagliate sui cluster
            session_texts: Testi originali (opzionale)
            save_html: Se salvare grafici HTML
            show_console: Se mostrare output console
            
        Returns:
            Dizionario con metriche e path file generati
        """
        print(f"\n📊 ═══════════════════════════════════════════════")
        print(f"    VISUALIZZAZIONE CLUSTERING (PARAMETRI)")
        print(f"📊 ═══════════════════════════════════════════════")
        
        # STEP 1: Calcola metriche qualità
        quality_metrics = self._calculate_quality_metrics(
            embeddings, cluster_labels
        )
        
        # STEP 2: Statistiche dettagliate
        cluster_stats = self._calculate_cluster_statistics(
            cluster_labels, cluster_info, session_texts
        )
        
        if show_console:
            self._print_clustering_console_stats(quality_metrics, cluster_stats)
        
        results = {
            'quality_metrics': quality_metrics,
            'cluster_statistics': cluster_stats,
            'generated_files': []
        }
        
        if save_html:
            # STEP 3: Genera grafici interattivi
            html_files = self._generate_clustering_visualizations(
                embeddings, cluster_labels, cluster_info, 
                quality_metrics, cluster_stats, session_texts
            )
            results['generated_files'].extend(html_files)
        
        return results
    
    def visualize_classification_statistics(self,
                                         embeddings: np.ndarray,
                                         cluster_labels: np.ndarray,
                                         final_predictions: List[Dict[str, Any]],
                                         cluster_info: Dict[int, Dict[str, Any]] = None,
                                         session_texts: List[str] = None,
                                         save_html: bool = True,
                                         show_console: bool = True) -> Dict[str, Any]:
        """
        Visualizzazione completa per STATISTICHE (con etichette classificazione finali)
        
        Args:
            embeddings: Array embeddings
            cluster_labels: Etichette cluster originali
            final_predictions: Predizioni finali con etichette reali
            cluster_info: Info sui cluster (opzionale)
            session_texts: Testi originali (opzionale)
            save_html: Se salvare grafici HTML
            show_console: Se mostrare output console
            
        Returns:
            Dizionario con metriche e path file generati
        """
        print(f"\n📈 ═══════════════════════════════════════════════")
        print(f"    STATISTICHE CLASSIFICAZIONE COMPLETE")
        print(f"📈 ═══════════════════════════════════════════════")
        
        # STEP 1: Elabora predizioni finali
        final_labels = [pred.get('predicted_label', 'altro') for pred in final_predictions]
        confidences = [pred.get('confidence', 0.0) for pred in final_predictions]
        methods = [pred.get('method', 'unknown') for pred in final_predictions]
        
        # STEP 2: Calcola metriche qualità con etichette finali  
        quality_metrics = self._calculate_quality_metrics(
            embeddings, cluster_labels, final_labels=final_labels
        )
        
        # STEP 3: Statistiche avanzate con etichette reali
        classification_stats = self._calculate_classification_statistics(
            final_predictions, cluster_labels, confidences, methods
        )
        
        if show_console:
            self._print_classification_console_stats(
                quality_metrics, classification_stats
            )
        
        results = {
            'quality_metrics': quality_metrics,
            'classification_statistics': classification_stats,
            'generated_files': []
        }
        
        if save_html:
            # STEP 4: Genera grafici con etichette finali
            html_files = self._generate_classification_visualizations(
                embeddings, cluster_labels, final_labels, confidences,
                methods, quality_metrics, classification_stats, session_texts
            )
            results['generated_files'].extend(html_files)
        
        return results
    
    def _calculate_quality_metrics(self,
                                 embeddings: np.ndarray,
                                 cluster_labels: np.ndarray,
                                 final_labels: List[str] = None) -> Dict[str, float]:
        """
        Calcola metriche di qualità clustering
        
        Args:
            embeddings: Array embeddings
            cluster_labels: Etichette cluster
            final_labels: Etichette finali classificazione (opzionale)
            
        Returns:
            Dizionario metriche qualità
        """
        metrics = {}
        
        try:
            # Filtra outlier per il calcolo delle metriche
            non_outlier_mask = cluster_labels != -1
            
            if non_outlier_mask.sum() > 1:
                # Silhouette Score
                if len(set(cluster_labels[non_outlier_mask])) > 1:
                    metrics['silhouette_score'] = silhouette_score(
                        embeddings[non_outlier_mask],
                        cluster_labels[non_outlier_mask]
                    )
                else:
                    metrics['silhouette_score'] = 0.0
                
                # Calinski-Harabasz Score
                if len(set(cluster_labels[non_outlier_mask])) > 1:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                        embeddings[non_outlier_mask],
                        cluster_labels[non_outlier_mask]
                    )
                else:
                    metrics['calinski_harabasz_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                
        except Exception as e:
            self.logger.warning(f"Errore calcolo metriche qualità: {e}")
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
        
        # Metriche base
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_outliers = np.sum(cluster_labels == -1)
        
        metrics.update({
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'total_samples': len(cluster_labels),
            'outlier_percentage': (n_outliers / len(cluster_labels)) * 100,
            'cluster_percentage': ((len(cluster_labels) - n_outliers) / len(cluster_labels)) * 100
        })
        
        # Se abbiamo etichette finali, calcola anche purezza
        if final_labels:
            metrics.update(self._calculate_purity_metrics(cluster_labels, final_labels))
            
        return metrics
    
    def _calculate_purity_metrics(self,
                                cluster_labels: np.ndarray,
                                final_labels: List[str]) -> Dict[str, float]:
        """
        Calcola metriche di purezza cluster rispetto alle etichette finali
        
        Args:
            cluster_labels: Etichette cluster
            final_labels: Etichette finali
            
        Returns:
            Metriche di purezza
        """
        try:
            # Calcola purezza per ogni cluster
            cluster_purity = {}
            total_purity = 0
            valid_clusters = 0
            
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)  # Rimuovi outlier
            
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_final_labels = [final_labels[i] for i in range(len(final_labels)) if cluster_mask[i]]
                
                if cluster_final_labels:
                    # Trova etichetta più frequente in questo cluster
                    label_counts = Counter(cluster_final_labels)
                    most_common_count = label_counts.most_common(1)[0][1]
                    purity = most_common_count / len(cluster_final_labels)
                    
                    cluster_purity[cluster_id] = purity
                    total_purity += purity
                    valid_clusters += 1
            
            average_purity = total_purity / max(valid_clusters, 1)
            
            return {
                'average_cluster_purity': average_purity,
                'cluster_purity_details': cluster_purity,
                'valid_clusters_for_purity': valid_clusters
            }
            
        except Exception as e:
            self.logger.warning(f"Errore calcolo purezza: {e}")
            return {
                'average_cluster_purity': 0.0,
                'cluster_purity_details': {},
                'valid_clusters_for_purity': 0
            }
    
    def _calculate_cluster_statistics(self,
                                    cluster_labels: np.ndarray,
                                    cluster_info: Dict[int, Dict[str, Any]],
                                    session_texts: List[str] = None) -> Dict[str, Any]:
        """
        Calcola statistiche dettagliate sui cluster
        
        Args:
            cluster_labels: Etichette cluster
            cluster_info: Info cluster dal sistema
            session_texts: Testi originali (opzionale)
            
        Returns:
            Statistiche dettagliate
        """
        stats = {
            'cluster_sizes': {},
            'cluster_methods': {},
            'confidence_distribution': {},
            'text_statistics': {},
            'size_distribution': {
                'small': 0,    # < 5 sessioni
                'medium': 0,   # 5-20 sessioni  
                'large': 0     # > 20 sessioni
            }
        }
        
        # Analizza ogni cluster
        for cluster_id, info in cluster_info.items():
            size = info.get('size', 0)
            method = info.get('classification_method', 'unknown')
            confidence = info.get('average_confidence', 0.0)
            
            stats['cluster_sizes'][cluster_id] = size
            stats['cluster_methods'][cluster_id] = method
            stats['confidence_distribution'][cluster_id] = confidence
            
            # Distribuzione dimensioni
            if size < 5:
                stats['size_distribution']['small'] += 1
            elif size <= 20:
                stats['size_distribution']['medium'] += 1
            else:
                stats['size_distribution']['large'] += 1
        
        # Statistiche sui testi se disponibili
        if session_texts:
            stats['text_statistics'] = self._analyze_text_statistics(
                session_texts, cluster_labels, cluster_info
            )
        
        return stats
    
    def _calculate_classification_statistics(self,
                                           final_predictions: List[Dict[str, Any]],
                                           cluster_labels: np.ndarray,
                                           confidences: List[float],
                                           methods: List[str]) -> Dict[str, Any]:
        """
        Calcola statistiche complete sulle classificazioni finali
        
        Args:
            final_predictions: Predizioni finali
            cluster_labels: Etichette cluster originali
            confidences: Valori di confidenza
            methods: Metodi utilizzati
            
        Returns:
            Statistiche classificazione
        """
        stats = {
            'label_distribution': Counter([pred.get('predicted_label', 'altro') for pred in final_predictions]),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'method_distribution': Counter(methods),
            'confidence_by_method': {},
            'confidence_by_label': {},
            'cluster_to_label_mapping': {}
        }
        
        # Analisi per metodo
        method_confidences = {}
        for method, conf in zip(methods, confidences):
            if method not in method_confidences:
                method_confidences[method] = []
            method_confidences[method].append(conf)
        
        for method, confs in method_confidences.items():
            stats['confidence_by_method'][method] = {
                'mean': np.mean(confs),
                'count': len(confs)
            }
        
        # Analisi per etichetta
        label_confidences = {}
        for pred, conf in zip(final_predictions, confidences):
            label = pred.get('predicted_label', 'altro')
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)
        
        for label, confs in label_confidences.items():
            stats['confidence_by_label'][label] = {
                'mean': np.mean(confs),
                'count': len(confs)
            }
        
        # Mapping cluster -> etichetta finale più frequente
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:
                cluster_mask = cluster_labels == cluster_id
                cluster_predictions = [final_predictions[i] for i in range(len(final_predictions)) if cluster_mask[i]]
                cluster_labels_final = [pred.get('predicted_label', 'altro') for pred in cluster_predictions]
                
                if cluster_labels_final:
                    most_common_label = Counter(cluster_labels_final).most_common(1)[0][0]
                    stats['cluster_to_label_mapping'][cluster_id] = most_common_label
        
        return stats
    
    def _analyze_text_statistics(self,
                               session_texts: List[str],
                               cluster_labels: np.ndarray,
                               cluster_info: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analizza statistiche sui testi per cluster
        
        Args:
            session_texts: Lista testi
            cluster_labels: Etichette cluster
            cluster_info: Info cluster
            
        Returns:
            Statistiche sui testi
        """
        text_stats = {
            'avg_length_by_cluster': {},
            'vocabulary_overlap': {},
            'total_chars': sum(len(text) for text in session_texts),
            'total_words': sum(len(text.split()) for text in session_texts)
        }
        
        # Analisi per cluster
        for cluster_id in cluster_info.keys():
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [session_texts[i] for i in range(len(session_texts)) if cluster_mask[i]]
            
            if cluster_texts:
                avg_length = np.mean([len(text) for text in cluster_texts])
                text_stats['avg_length_by_cluster'][cluster_id] = avg_length
        
        return text_stats
    
    def _print_clustering_console_stats(self,
                                      quality_metrics: Dict[str, float],
                                      cluster_stats: Dict[str, Any]) -> None:
        """
        Stampa statistiche clustering sulla console
        """
        print(f"\n🎯 METRICHE QUALITÀ CLUSTERING")
        print(f"{'─' * 50}")
        print(f"   📊 Cluster trovati: {quality_metrics['n_clusters']}")
        print(f"   🔍 Outliers: {quality_metrics['n_outliers']} ({quality_metrics['outlier_percentage']:.1f}%)")
        print(f"   📈 Campioni clusterizzati: {quality_metrics['total_samples'] - quality_metrics['n_outliers']} ({quality_metrics['cluster_percentage']:.1f}%)")
        print(f"   🎯 Silhouette Score: {quality_metrics['silhouette_score']:.3f}")
        print(f"   📋 Calinski-Harabasz Score: {quality_metrics['calinski_harabasz_score']:.1f}")
        
        print(f"\n📊 DISTRIBUZIONE CLUSTER")
        print(f"{'─' * 50}")
        print(f"   🔸 Cluster piccoli (< 5): {cluster_stats['size_distribution']['small']}")
        print(f"   🔹 Cluster medi (5-20): {cluster_stats['size_distribution']['medium']}")  
        print(f"   🔶 Cluster grandi (> 20): {cluster_stats['size_distribution']['large']}")
        
        print(f"\n🔧 METODI CLUSTERING UTILIZZATI")
        print(f"{'─' * 50}")
        method_counts = Counter(cluster_stats['cluster_methods'].values())
        for method, count in method_counts.most_common():
            print(f"   • {method}: {count} cluster")
        
        print(f"\n🎯 TOP CLUSTER PER DIMENSIONE")
        print(f"{'─' * 50}")
        sorted_clusters = sorted(cluster_stats['cluster_sizes'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for cluster_id, size in sorted_clusters:
            confidence = cluster_stats['confidence_distribution'].get(cluster_id, 0.0)
            method = cluster_stats['cluster_methods'].get(cluster_id, 'unknown')
            print(f"   • Cluster {cluster_id}: {size} sessioni (conf: {confidence:.2f}, {method})")
    
    def _print_classification_console_stats(self,
                                          quality_metrics: Dict[str, float],
                                          classification_stats: Dict[str, Any]) -> None:
        """
        Stampa statistiche classificazione sulla console
        """
        self._print_clustering_console_stats(quality_metrics, {'size_distribution': {'small': 0, 'medium': 0, 'large': 0}, 'cluster_sizes': {}, 'cluster_methods': {}, 'confidence_distribution': {}})
        
        if 'average_cluster_purity' in quality_metrics:
            print(f"   🎯 Purezza media cluster: {quality_metrics['average_cluster_purity']:.3f}")
        
        print(f"\n🏷️  DISTRIBUZIONE ETICHETTE FINALI")
        print(f"{'─' * 50}")
        for label, count in classification_stats['label_distribution'].most_common():
            percentage = (count / sum(classification_stats['label_distribution'].values())) * 100
            avg_conf = classification_stats['confidence_by_label'].get(label, {}).get('mean', 0.0)
            print(f"   • {label}: {count} ({percentage:.1f}%, conf: {avg_conf:.2f})")
        
        print(f"\n📈 STATISTICHE CONFIDENZA")
        print(f"{'─' * 50}")
        conf_stats = classification_stats['confidence_stats']
        print(f"   📊 Media: {conf_stats['mean']:.3f}")
        print(f"   📊 Mediana: {conf_stats['median']:.3f}")
        print(f"   📊 Dev. Standard: {conf_stats['std']:.3f}")
        print(f"   📊 Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
        
        print(f"\n🔧 METODI CLASSIFICAZIONE UTILIZZATI")
        print(f"{'─' * 50}")
        for method, count in classification_stats['method_distribution'].most_common():
            avg_conf = classification_stats['confidence_by_method'].get(method, {}).get('mean', 0.0)
            percentage = (count / sum(classification_stats['method_distribution'].values())) * 100
            print(f"   • {method}: {count} ({percentage:.1f}%, conf: {avg_conf:.2f})")
    
    def _generate_clustering_visualizations(self,
                                          embeddings: np.ndarray,
                                          cluster_labels: np.ndarray,
                                          cluster_info: Dict[int, Dict[str, Any]],
                                          quality_metrics: Dict[str, float],
                                          cluster_stats: Dict[str, Any],
                                          session_texts: List[str] = None) -> List[str]:
        """
        Genera visualizzazioni HTML per il clustering
        
        Returns:
            Lista dei file HTML generati
        """
        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Grafico 2D con t-SNE
            tsne_file = self._create_2d_plot(
                embeddings, cluster_labels, cluster_info,
                method='tsne', filename=f"clustering_2d_tsne_{timestamp}.html",
                title="Clustering 2D - t-SNE"
            )
            generated_files.append(tsne_file)
            
            # 2. Grafico 2D con PCA
            pca_file = self._create_2d_plot(
                embeddings, cluster_labels, cluster_info,
                method='pca', filename=f"clustering_2d_pca_{timestamp}.html",
                title="Clustering 2D - PCA"
            )
            generated_files.append(pca_file)
            
            # 3. Grafico 3D con PCA
            pca_3d_file = self._create_3d_plot(
                embeddings, cluster_labels, cluster_info,
                method='pca', filename=f"clustering_3d_pca_{timestamp}.html",
                title="Clustering 3D - PCA"
            )
            generated_files.append(pca_3d_file)
            
            # 4. Dashboard statistiche
            dashboard_file = self._create_statistics_dashboard(
                quality_metrics, cluster_stats, cluster_info,
                filename=f"clustering_dashboard_{timestamp}.html",
                title="Dashboard Clustering"
            )
            generated_files.append(dashboard_file)
            
            print(f"\n💾 GRAFICI GENERATI:")
            for file in generated_files:
                print(f"   📄 {file}")
                
        except Exception as e:
            self.logger.error(f"Errore generazione visualizzazioni: {e}")
        
        return generated_files
    
    def _generate_classification_visualizations(self,
                                              embeddings: np.ndarray,
                                              cluster_labels: np.ndarray,
                                              final_labels: List[str],
                                              confidences: List[float],
                                              methods: List[str],
                                              quality_metrics: Dict[str, float],
                                              classification_stats: Dict[str, Any],
                                              session_texts: List[str] = None) -> List[str]:
        """
        Genera visualizzazioni HTML per le classificazioni finali
        
        Returns:
            Lista dei file HTML generati  
        """
        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Grafico 2D colorato per etichette finali
            labels_2d_file = self._create_2d_classification_plot(
                embeddings, final_labels, confidences,
                filename=f"classification_2d_{timestamp}.html",
                title="Classificazione Finale 2D"
            )
            generated_files.append(labels_2d_file)
            
            # 2. Grafico 3D colorato per etichette finali
            labels_3d_file = self._create_3d_classification_plot(
                embeddings, final_labels, confidences,
                filename=f"classification_3d_{timestamp}.html",
                title="Classificazione Finale 3D"
            )
            generated_files.append(labels_3d_file)
            
            # 3. Confronto cluster originali vs etichette finali
            comparison_file = self._create_cluster_label_comparison(
                embeddings, cluster_labels, final_labels,
                filename=f"cluster_vs_labels_{timestamp}.html",
                title="Cluster vs Etichette Finali"
            )
            generated_files.append(comparison_file)
            
            # 4. Dashboard classificazione completa
            class_dashboard_file = self._create_classification_dashboard(
                quality_metrics, classification_stats, 
                confidences, methods, final_labels,
                filename=f"classification_dashboard_{timestamp}.html",
                title="Dashboard Classificazione"
            )
            generated_files.append(class_dashboard_file)
            
            print(f"\n💾 GRAFICI CLASSIFICAZIONE GENERATI:")
            for file in generated_files:
                print(f"   📄 {file}")
                
        except Exception as e:
            self.logger.error(f"Errore generazione visualizzazioni classificazione: {e}")
        
        return generated_files
    
    def _create_2d_plot(self,
                       embeddings: np.ndarray,
                       cluster_labels: np.ndarray,
                       cluster_info: Dict[int, Dict[str, Any]],
                       method: str = 'tsne',
                       filename: str = None,
                       title: str = None) -> str:
        """
        Crea grafico 2D interattivo
        """
        # Riduzione dimensionale
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = MDS(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(embeddings)
        
        # Crea DataFrame
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': cluster_labels,
            'is_outlier': cluster_labels == -1
        })
        
        # Aggiungi info cluster
        df['cluster_info'] = df['cluster'].apply(
            lambda c: f"Outlier" if c == -1 else 
            f"Cluster {c}: {cluster_info.get(c, {}).get('intent_string', 'N/A')}"
        )
        
        # Crea grafico
        fig = px.scatter(
            df, x='x', y='y', 
            color='cluster',
            hover_data=['cluster_info'],
            title=title or f"Clustering 2D - {method.upper()}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            width=900, height=700,
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=1)
        )
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or f"cluster_2d_{method}.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_3d_plot(self,
                       embeddings: np.ndarray,
                       cluster_labels: np.ndarray,
                       cluster_info: Dict[int, Dict[str, Any]],
                       method: str = 'pca',
                       filename: str = None,
                       title: str = None) -> str:
        """
        Crea grafico 3D interattivo
        """
        # Riduzione dimensionale 3D
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
        else:
            reducer = MDS(n_components=3, random_state=42)
        
        coords_3d = reducer.fit_transform(embeddings)
        
        # Crea DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
            'cluster': cluster_labels
        })
        
        # Aggiungi info cluster
        df['cluster_info'] = df['cluster'].apply(
            lambda c: f"Outlier" if c == -1 else 
            f"Cluster {c}: {cluster_info.get(c, {}).get('intent_string', 'N/A')}"
        )
        
        # Crea grafico 3D
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            hover_data=['cluster_info'],
            title=title or f"Clustering 3D - {method.upper()}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(width=1000, height=800)
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or f"cluster_3d_{method}.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_2d_classification_plot(self,
                                     embeddings: np.ndarray,
                                     final_labels: List[str],
                                     confidences: List[float],
                                     filename: str = None,
                                     title: str = None) -> str:
        """
        Crea grafico 2D colorato per etichette finali
        """
        # t-SNE per visualizzazione
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords_2d = tsne.fit_transform(embeddings)
        
        # DataFrame
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'label': final_labels,
            'confidence': confidences
        })
        
        # Grafico con dimensione basata su confidenza
        fig = px.scatter(
            df, x='x', y='y', 
            color='label',
            size='confidence',
            hover_data=['label', 'confidence'],
            title=title or "Classificazione Finale 2D",
            size_max=15
        )
        
        fig.update_layout(
            width=900, height=700,
            showlegend=True,
            legend=dict(orientation="v", x=1.02, y=1)
        )
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or "classification_2d.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_3d_classification_plot(self,
                                     embeddings: np.ndarray,
                                     final_labels: List[str],
                                     confidences: List[float],
                                     filename: str = None,
                                     title: str = None) -> str:
        """
        Crea grafico 3D colorato per etichette finali
        """
        # PCA 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(embeddings)
        
        # DataFrame
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1], 
            'z': coords_3d[:, 2],
            'label': final_labels,
            'confidence': confidences
        })
        
        # Grafico 3D
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            size='confidence',
            hover_data=['label', 'confidence'],
            title=title or "Classificazione Finale 3D",
            size_max=15
        )
        
        fig.update_layout(width=1000, height=800)
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or "classification_3d.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_cluster_label_comparison(self,
                                       embeddings: np.ndarray,
                                       cluster_labels: np.ndarray,
                                       final_labels: List[str],
                                       filename: str = None,
                                       title: str = None) -> str:
        """
        Crea confronto side-by-side cluster vs etichette finali
        """
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords_2d = tsne.fit_transform(embeddings)
        
        # Subplot side by side
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Cluster Originali', 'Etichette Finali'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # DataFrame per entrambi i grafici
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': cluster_labels,
            'label': final_labels
        })
        
        # Plot 1: Cluster originali
        for cluster_id in set(cluster_labels):
            cluster_data = df[df['cluster'] == cluster_id]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    name=f'Cluster {cluster_id}' if cluster_id != -1 else 'Outliers',
                    legendgroup='clusters',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Plot 2: Etichette finali
        for label in set(final_labels):
            label_data = df[df['label'] == label]
            fig.add_trace(
                go.Scatter(
                    x=label_data['x'],
                    y=label_data['y'],
                    mode='markers',
                    name=f'{label}',
                    legendgroup='labels',
                    showlegend=True
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title or "Confronto Cluster vs Etichette Finali",
            width=1400, height=700
        )
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or "cluster_vs_labels.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_statistics_dashboard(self,
                                   quality_metrics: Dict[str, float],
                                   cluster_stats: Dict[str, Any],
                                   cluster_info: Dict[int, Dict[str, Any]],
                                   filename: str = None,
                                   title: str = None) -> str:
        """
        Crea dashboard con statistiche clustering
        """
        # Crea subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Distribuzione Dimensioni Cluster',
                'Metodi Clustering',
                'Distribuzione Confidenza',
                'Metriche Qualità'
            ],
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'histogram'}, {'type': 'indicator'}]]
        )
        
        # 1. Distribuzione dimensioni
        size_dist = cluster_stats['size_distribution']
        fig.add_trace(
            go.Bar(x=list(size_dist.keys()), y=list(size_dist.values()),
                   name='Dimensioni'),
            row=1, col=1
        )
        
        # 2. Metodi clustering
        method_counts = Counter(cluster_stats['cluster_methods'].values())
        fig.add_trace(
            go.Pie(labels=list(method_counts.keys()), values=list(method_counts.values()),
                   name='Metodi'),
            row=1, col=2
        )
        
        # 3. Distribuzione confidenza
        confidences = list(cluster_stats['confidence_distribution'].values())
        fig.add_trace(
            go.Histogram(x=confidences, name='Confidenza', nbinsx=20),
            row=2, col=1
        )
        
        # 4. Metriche qualità (indicatori)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=quality_metrics['silhouette_score'],
                title={"text": "Silhouette Score"},
                domain={'row': 1, 'column': 1}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title or "Dashboard Statistiche Clustering",
            width=1200, height=800,
            showlegend=False
        )
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or "clustering_dashboard.html")
        fig.write_html(filepath)
        
        return filepath
    
    def _create_classification_dashboard(self,
                                       quality_metrics: Dict[str, float],
                                       classification_stats: Dict[str, Any],
                                       confidences: List[float],
                                       methods: List[str],
                                       final_labels: List[str],
                                       filename: str = None,
                                       title: str = None) -> str:
        """
        Crea dashboard completo per classificazioni
        """
        # Crea subplot dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Distribuzione Etichette',
                'Metodi Classificazione',
                'Distribuzione Confidenza',
                'Confidenza per Metodo',
                'Confidenza per Etichetta',
                'Metriche Qualità'
            ],
            specs=[[{'type': 'pie'}, {'type': 'bar'}, {'type': 'histogram'}],
                   [{'type': 'box'}, {'type': 'box'}, {'type': 'indicator'}]]
        )
        
        # 1. Distribuzione etichette
        label_dist = classification_stats['label_distribution']
        fig.add_trace(
            go.Pie(labels=list(label_dist.keys()), values=list(label_dist.values())),
            row=1, col=1
        )
        
        # 2. Metodi classificazione
        method_dist = classification_stats['method_distribution']
        fig.add_trace(
            go.Bar(x=list(method_dist.keys()), y=list(method_dist.values())),
            row=1, col=2
        )
        
        # 3. Distribuzione confidenza generale
        fig.add_trace(
            go.Histogram(x=confidences, nbinsx=30),
            row=1, col=3
        )
        
        # 4. Box plot confidenza per metodo
        unique_methods = list(set(methods))
        for method in unique_methods:
            method_confidences = [conf for conf, m in zip(confidences, methods) if m == method]
            fig.add_trace(
                go.Box(y=method_confidences, name=method),
                row=2, col=1
            )
        
        # 5. Box plot confidenza per etichetta
        unique_labels = list(set(final_labels))[:10]  # Limite prime 10
        for label in unique_labels:
            label_confidences = [conf for conf, l in zip(confidences, final_labels) if l == label]
            fig.add_trace(
                go.Box(y=label_confidences, name=label),
                row=2, col=2
            )
        
        # 6. Indicatore qualità
        avg_conf = classification_stats['confidence_stats']['mean']
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=avg_conf,
                title={"text": "Confidenza Media"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkgreen"},
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.8}}
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=title or "Dashboard Classificazione Completo",
            width=1400, height=1000,
            showlegend=False
        )
        
        # Salva file
        filepath = os.path.join(self.output_dir, filename or "classification_dashboard.html")
        fig.write_html(filepath)
        
        return filepath
