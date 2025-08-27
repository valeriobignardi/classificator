#!/usr/bin/env python3
"""
File: hdbscan_tuning_guide.py
Autore: Sistema di Classificazione
Data: 26 Agosto 2025

Descrizione: Guida ottimizzazione parametri HDBSCAN per ridurre outliers

Storia aggiornamenti:
- 26 Agosto 2025: Creazione sistema suggerimenti parametri
"""

from typing import Dict, List, Tuple, Any
import numpy as np

class HDBSCANTuningGuide:
    """
    Guida per l'ottimizzazione dei parametri HDBSCAN
    
    Fornisce suggerimenti per ridurre outliers e migliorare la qualità clustering
    """
    
    @staticmethod
    def analyze_outlier_problem(n_points: int, n_outliers: int, 
                               current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza il problema degli outliers e suggerisce soluzioni
        
        Args:
            n_points: Numero totale punti
            n_outliers: Numero outliers attuali  
            current_params: Parametri attuali
            
        Returns:
            Dizionario con analisi e suggerimenti
        """
        outlier_percentage = (n_outliers / n_points) * 100
        
        analysis = {
            'outlier_percentage': outlier_percentage,
            'severity': HDBSCANTuningGuide._assess_outlier_severity(outlier_percentage),
            'current_params': current_params,
            'suggestions': []
        }
        
        # Suggerimenti basati sulla severità
        if outlier_percentage > 80:
            analysis['suggestions'].extend([
                {
                    'param': 'min_cluster_size',
                    'current': current_params.get('min_cluster_size', 5),
                    'suggested': max(3, current_params.get('min_cluster_size', 5) // 2),
                    'reason': 'Riduci min_cluster_size per permettere cluster più piccoli'
                },
                {
                    'param': 'cluster_selection_method',
                    'current': current_params.get('cluster_selection_method', 'eom'),
                    'suggested': 'leaf',
                    'reason': 'Usa "leaf" per cluster più dettagliati invece di "eom"'
                },
                {
                    'param': 'alpha',
                    'current': current_params.get('alpha', 1.0),
                    'suggested': 0.5,
                    'reason': 'Riduci alpha per essere meno severo sui noise points'
                }
            ])
        
        elif outlier_percentage > 50:
            analysis['suggestions'].extend([
                {
                    'param': 'cluster_selection_epsilon',
                    'current': current_params.get('cluster_selection_epsilon', 0.08),
                    'suggested': current_params.get('cluster_selection_epsilon', 0.08) * 1.5,
                    'reason': 'Aumenta epsilon per mergeare cluster vicini'
                },
                {
                    'param': 'min_samples',
                    'current': current_params.get('min_samples', 3),
                    'suggested': max(1, current_params.get('min_samples', 3) - 1),
                    'reason': 'Riduci min_samples per requisiti densità meno restrittivi'
                }
            ])
        
        return analysis
    
    @staticmethod
    def _assess_outlier_severity(percentage: float) -> str:
        """Valuta la severità del problema outliers"""
        if percentage > 80:
            return 'CRITICO'
        elif percentage > 60:
            return 'ALTO'
        elif percentage > 40:
            return 'MODERATO'
        elif percentage > 20:
            return 'BASSO'
        else:
            return 'OTTIMALE'
    
    @staticmethod
    def get_parameter_descriptions() -> Dict[str, Dict[str, str]]:
        """
        Restituisce descrizioni dettagliate parametri HDBSCAN
        
        Returns:
            Dizionario con descrizioni parametri per frontend
        """
        return {
            'min_cluster_size': {
                'description': 'Dimensione minima dei cluster',
                'effect_on_outliers': 'CRITICO: Valori più bassi = meno outliers',
                'range': '3-50',
                'default': '15',
                'tip': 'Prova 5-10 per dataset piccoli, 15-25 per grandi'
            },
            'min_samples': {
                'description': 'Densità minima necessaria per formare cluster',
                'effect_on_outliers': 'ALTO: Valori più bassi = meno outliers', 
                'range': '1-10',
                'default': '3',
                'tip': 'Usa 1-2 per dati sparsi, 3-5 per dati densi'
            },
            'cluster_selection_epsilon': {
                'description': 'Distanza massima per unire cluster',
                'effect_on_outliers': 'MODERATO: Valori più alti = meno outliers',
                'range': '0.0-1.0', 
                'default': '0.08',
                'tip': 'Aumenta per unire cluster vicini e ridurre outliers'
            },
            'cluster_selection_method': {
                'description': 'Strategia selezione cluster finale',
                'effect_on_outliers': 'MODERATO: "leaf" può ridurre outliers',
                'range': 'eom|leaf',
                'default': 'eom',
                'tip': 'leaf: più cluster piccoli; eom: cluster più stabili'
            },
            'alpha': {
                'description': 'Controllo rumore/outlier sensitivity',
                'effect_on_outliers': 'ALTO: Valori più bassi = meno outliers',
                'range': '0.1-2.0',
                'default': '1.0', 
                'tip': 'Riduci (0.5-0.8) per essere meno severo con outliers'
            },
            'max_cluster_size': {
                'description': 'Dimensione massima cluster (0=illimitato)',
                'effect_on_outliers': 'BASSO: Può dividere cluster grandi',
                'range': '0-1000',
                'default': '0',
                'tip': 'Imposta per dividere cluster troppo grandi'
            },
            'allow_single_cluster': {
                'description': 'Permetti un singolo cluster globale',
                'effect_on_outliers': 'ESTREMO: true può eliminare TUTTI gli outliers',
                'range': 'true|false',
                'default': 'false',
                'tip': 'Attiva solo se vuoi forzare tutto in un cluster'
            }
        }
    
    @staticmethod
    def get_quick_fix_presets() -> Dict[str, Dict[str, Any]]:
        """
        Preset parametri per problemi comuni
        
        Returns:
            Dizionario con preset ottimizzati
        """
        return {
            'reduce_outliers_aggressive': {
                'name': '🎯 Riduzione Outliers Aggressiva',
                'description': 'Parametri per minimizzare drasticamente gli outliers',
                'params': {
                    'min_cluster_size': 3,
                    'min_samples': 1,
                    'cluster_selection_epsilon': 0.15,
                    'cluster_selection_method': 'leaf',
                    'alpha': 0.5
                },
                'expected_outliers': '10-30%'
            },
            'reduce_outliers_moderate': {
                'name': '⚖️  Riduzione Outliers Moderata',
                'description': 'Equilibrio tra qualità cluster e riduzione outliers',
                'params': {
                    'min_cluster_size': 5,
                    'min_samples': 2, 
                    'cluster_selection_epsilon': 0.12,
                    'cluster_selection_method': 'eom',
                    'alpha': 0.8
                },
                'expected_outliers': '30-50%'
            },
            'quality_focused': {
                'name': '🏆 Focus Qualità Cluster',
                'description': 'Parametri conservativi per cluster di alta qualità',
                'params': {
                    'min_cluster_size': 10,
                    'min_samples': 3,
                    'cluster_selection_epsilon': 0.08,
                    'cluster_selection_method': 'eom', 
                    'alpha': 1.0
                },
                'expected_outliers': '50-80%'
            },
            'single_cluster_fallback': {
                'name': '🌐 Fallback Cluster Singolo',
                'description': 'Forza tutto in un cluster se altri metodi falliscono',
                'params': {
                    'min_cluster_size': 3,
                    'min_samples': 1,
                    'cluster_selection_epsilon': 0.2,
                    'allow_single_cluster': True,
                    'alpha': 0.3
                },
                'expected_outliers': '0-10%'
            }
        }
