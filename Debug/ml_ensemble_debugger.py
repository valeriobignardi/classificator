"""
ML Ensemble Debugger - Sistema di debugging per il classificatore ML ensemble
Fornisce debug dettagliato per training, predizione e ensemble voting
"""

import os
import json
import yaml
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import traceback
import sys

# Aggiungo il path per le utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Utils.numpy_serialization import convert_numpy_types

@dataclass
class MLDebugInfo:
    """Struttura per informazioni di debug ML"""
    timestamp: str
    session_id: str
    phase: str  # 'training', 'prediction', 'ensemble_voting'
    input_features: Optional[np.ndarray]
    feature_info: Dict[str, Any]
    ml_predictions: Dict[str, Any]
    llm_prediction: Optional[Dict[str, Any]]
    ensemble_result: Dict[str, Any]
    weights_used: Dict[str, float]
    processing_time: float
    error: Optional[str] = None
    context: Dict[str, Any] = None

class MLEnsembleDebugger:
    """
    Debugger avanzato per il sistema ML Ensemble
    Fornisce logging visualmente strutturato per training e predizione
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il debugger ML ensemble
        
        Args:
            config_path: Percorso del file di configurazione
        """
        # Carica configurazione
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Configurazione debug
        self.debug_config = self.config.get('debug', {})
        self.ml_debug_config = self.debug_config.get('ml_debug', {})
        
        # Check se debug è abilitato
        self.enabled = (self.debug_config.get('enabled', False) and 
                       self.ml_debug_config.get('enabled', False))
        
        if not self.enabled:
            return
        
        # Configurazione logging
        self.setup_logging()
        
        # Storage per debug info
        self.debug_history: List[MLDebugInfo] = []
        
        print("🔍 ML Ensemble Debugger inizializzato e ATTIVO")
        print(f"   📝 Debug file: {self.debug_file}")
    
    def setup_logging(self):
        """Setup del sistema di logging"""
        if not self.enabled:
            return
        
        # Crea directory debug
        debug_path = self.debug_config.get('debug_files', {}).get('base_path', './debug_logs')
        os.makedirs(debug_path, exist_ok=True)
        
        # File di debug ML
        ml_log_file = self.debug_config.get('debug_files', {}).get('ml_log', 'ml_debug.log')
        self.debug_file = os.path.join(debug_path, ml_log_file)
        
        # Setup logger
        self.logger = logging.getLogger('MLEnsembleDebugger')
        self.logger.setLevel(logging.DEBUG)
        
        # Handler per file
        file_handler = logging.FileHandler(self.debug_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug_training(self,
                      session_id: str,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      training_result: Dict[str, Any],
                      processing_time: float,
                      error: Optional[str] = None) -> None:
        """
        Debug del processo di training ML
        
        Args:
            session_id: ID della sessione di training
            X_train: Features di training
            y_train: Labels di training
            training_result: Risultato del training
            processing_time: Tempo di elaborazione
            error: Eventuale errore
        """
        if not self.enabled:
            return
        
        # Analizza features
        feature_info = self._analyze_features(X_train)
        
        # Analizza labels
        label_info = self._analyze_labels(y_train)
        
        debug_info = MLDebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            phase="training",
            input_features=X_train,
            feature_info={**feature_info, **label_info},
            ml_predictions={},
            llm_prediction=None,
            ensemble_result=training_result,
            weights_used={},
            processing_time=processing_time,
            error=error,
            context={
                'training_samples': len(X_train),
                'feature_dimensions': X_train.shape[1] if len(X_train) > 0 else 0,
                'unique_labels': len(np.unique(y_train)) if len(y_train) > 0 else 0
            }
        )
        
        self.debug_history.append(debug_info)
        
        if self.ml_debug_config.get('visual_formatting', True):
            self._print_training_debug(debug_info)
        
        self._log_to_file(debug_info)
    
    def debug_prediction(self,
                        session_id: str,
                        input_text: str,
                        features: np.ndarray,
                        ml_predictions: Dict[str, Any],
                        llm_prediction: Optional[Dict[str, Any]],
                        ensemble_result: Dict[str, Any],
                        weights_used: Dict[str, float],
                        processing_time: float,
                        error: Optional[str] = None) -> None:
        """
        Debug del processo di predizione ensemble
        
        Args:
            session_id: ID della sessione
            input_text: Testo di input
            features: Features estratte
            ml_predictions: Predizioni ML
            llm_prediction: Predizione LLM (se disponibile)
            ensemble_result: Risultato ensemble finale
            weights_used: Pesi utilizzati nell'ensemble
            processing_time: Tempo di elaborazione
            error: Eventuale errore
        """
        if not self.enabled:
            return
        
        # Analizza features
        feature_info = self._analyze_single_features(features)
        
        debug_info = MLDebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            phase="prediction",
            input_features=features,
            feature_info=feature_info,
            ml_predictions=ml_predictions,
            llm_prediction=llm_prediction,
            ensemble_result=ensemble_result,
            weights_used=weights_used,
            processing_time=processing_time,
            error=error,
            context={
                'input_text_length': len(input_text),
                'input_preview': input_text[:100] + '...' if len(input_text) > 100 else input_text,
                'feature_dimensions': len(features) if features is not None else 0,
                'ml_available': ml_predictions is not None,
                'llm_available': llm_prediction is not None
            }
        )
        
        self.debug_history.append(debug_info)
        
        if self.ml_debug_config.get('visual_formatting', True):
            self._print_prediction_debug(debug_info)
        
        self._log_to_file(debug_info)
    
    def debug_ensemble_voting(self,
                            session_id: str,
                            ml_pred: Dict[str, Any],
                            llm_pred: Dict[str, Any],
                            voting_process: Dict[str, Any],
                            final_result: Dict[str, Any],
                            processing_time: float) -> None:
        """
        Debug del processo di voting ensemble
        
        Args:
            session_id: ID della sessione
            ml_pred: Predizione ML
            llm_pred: Predizione LLM
            voting_process: Dettagli del processo di voting
            final_result: Risultato finale
            processing_time: Tempo di elaborazione
        """
        if not self.enabled:
            return
        
        debug_info = MLDebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            phase="ensemble_voting",
            input_features=None,
            feature_info={},
            ml_predictions=ml_pred,
            llm_prediction=llm_pred,
            ensemble_result=final_result,
            weights_used=voting_process.get('weights_used', {}),
            processing_time=processing_time,
            context={
                'agreement': voting_process.get('agreement', False),
                'combination_method': voting_process.get('combination_method', 'unknown'),
                'confidence_adjustment': voting_process.get('confidence_adjustment', 1.0)
            }
        )
        
        self.debug_history.append(debug_info)
        
        if self.ml_debug_config.get('visual_formatting', True):
            self._print_voting_debug(debug_info)
        
        self._log_to_file(debug_info)
    
    def _print_training_debug(self, debug_info: MLDebugInfo):
        """Stampa debug per il training"""
        print("\n" + "="*100)
        print(f"🎓 ML TRAINING DEBUG | {debug_info.timestamp}")
        print("="*100)
        
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"⏱️  Training Time: {debug_info.processing_time:.3f}s")
        if debug_info.error:
            print(f"❌ Error: {debug_info.error}")
        print("-"*100)
        
        # Training data info
        print("📊 TRAINING DATA ANALYSIS")
        print("-"*50)
        for key, value in debug_info.feature_info.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
            elif isinstance(value, list) and len(value) <= 10:
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {str(value)[:100]}...")
        print("-"*100)
        
        # Training results
        print("🎯 TRAINING RESULTS")
        print("-"*50)
        for key, value in debug_info.ensemble_result.items():
            print(f"   {key}: {value}")
        print("-"*100)
        
        print("✅ ML TRAINING DEBUG COMPLETED")
        print("="*100 + "\n")
    
    def _print_prediction_debug(self, debug_info: MLDebugInfo):
        """Stampa debug per la predizione"""
        print("\n" + "="*100)
        print(f"🔮 ML PREDICTION DEBUG | {debug_info.timestamp}")
        print("="*100)
        
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"⏱️  Prediction Time: {debug_info.processing_time:.3f}s")
        if debug_info.error:
            print(f"❌ Error: {debug_info.error}")
        print("-"*100)
        
        # Input info
        print("📥 INPUT ANALYSIS")
        print("-"*50)
        print(f"💬 Text: {debug_info.context.get('input_preview', 'N/A')}")
        print(f"🔢 Feature dims: {debug_info.context.get('feature_dimensions', 0)}")
        print("-"*100)
        
        # ML Predictions
        if debug_info.ml_predictions and self.ml_debug_config.get('log_prediction', True):
            print("🤖 ML PREDICTIONS")
            print("-"*50)
            for key, value in debug_info.ml_predictions.items():
                if key == 'probabilities' and isinstance(value, dict):
                    print(f"   {key}:")
                    for label, prob in sorted(value.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"     {label}: {prob:.3f}")
                else:
                    print(f"   {key}: {value}")
            print("-"*100)
        
        # LLM Predictions
        if debug_info.llm_prediction:
            print("🧠 LLM PREDICTIONS")
            print("-"*50)
            for key, value in debug_info.llm_prediction.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {value}")
            print("-"*100)
        
        # Ensemble result
        print("🎯 ENSEMBLE RESULT")
        print("-"*50)
        for key, value in debug_info.ensemble_result.items():
            print(f"   {key}: {value}")
        
        # Weights
        if debug_info.weights_used:
            print(f"\n⚖️  Weights used:")
            for component, weight in debug_info.weights_used.items():
                weight_bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
                print(f"   {component}: [{weight_bar}] {weight:.3f}")
        
        print("-"*100)
        print("✅ ML PREDICTION DEBUG COMPLETED")
        print("="*100 + "\n")
    
    def _print_voting_debug(self, debug_info: MLDebugInfo):
        """Stampa debug per il voting ensemble"""
        print("\n" + "="*100)
        print(f"🗳️  ENSEMBLE VOTING DEBUG | {debug_info.timestamp}")
        print("="*100)
        
        agreement = debug_info.context.get('agreement', False)
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"⏱️  Voting Time: {debug_info.processing_time:.3f}s")
        print(f"🤝 Agreement: {'✅ YES' if agreement else '❌ NO'}")
        print("-"*100)
        
        # ML vs LLM comparison
        print("⚔️  ML vs LLM COMPARISON")
        print("-"*50)
        if debug_info.ml_predictions:
            ml_label = debug_info.ml_predictions.get('predicted_label', 'N/A')
            ml_conf = debug_info.ml_predictions.get('confidence', 0.0)
            print(f"🤖 ML:  {ml_label} (conf: {ml_conf:.3f})")
        
        if debug_info.llm_prediction:
            llm_label = debug_info.llm_prediction.get('predicted_label', 'N/A')
            llm_conf = debug_info.llm_prediction.get('confidence', 0.0)
            print(f"🧠 LLM: {llm_label} (conf: {llm_conf:.3f})")
        
        # Final result
        final_label = debug_info.ensemble_result.get('predicted_label', 'N/A')
        final_conf = debug_info.ensemble_result.get('confidence', 0.0)
        print(f"🎯 FINAL: {final_label} (conf: {final_conf:.3f})")
        print("-"*100)
        
        # Voting process details
        print("🔍 VOTING PROCESS")
        print("-"*50)
        method = debug_info.context.get('combination_method', 'unknown')
        print(f"📋 Method: {method}")
        
        for key, value in debug_info.weights_used.items():
            print(f"⚖️  {key}: {value:.3f}")
        
        print("-"*100)
        print("✅ ENSEMBLE VOTING DEBUG COMPLETED")
        print("="*100 + "\n")
    
    def _analyze_features(self, X: np.ndarray) -> Dict[str, Any]:
        """Analizza le features di training"""
        if X is None or len(X) == 0:
            return {'feature_analysis': 'No features provided'}
        
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_mean': np.mean(X),
            'feature_std': np.std(X),
            'feature_min': np.min(X),
            'feature_max': np.max(X),
            'non_zero_features': np.count_nonzero(X),
            'feature_sparsity': 1 - (np.count_nonzero(X) / X.size)
        }
    
    def _analyze_single_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analizza features di una singola predizione"""
        if features is None or len(features) == 0:
            return {'feature_analysis': 'No features provided'}
        
        return {
            'n_features': len(features),
            'feature_norm': np.linalg.norm(features),
            'feature_mean': np.mean(features),
            'feature_std': np.std(features),
            'non_zero_count': np.count_nonzero(features),
            'sparsity': 1 - (np.count_nonzero(features) / len(features))
        }
    
    def _analyze_labels(self, y: np.ndarray) -> Dict[str, Any]:
        """Analizza le labels di training"""
        if y is None or len(y) == 0:
            return {'label_analysis': 'No labels provided'}
        
        unique_labels, counts = np.unique(y, return_counts=True)
        
        return {
            'n_labels': len(y),
            'unique_labels': len(unique_labels),
            'label_distribution': dict(zip(unique_labels, counts.tolist())),
            'most_frequent_label': unique_labels[np.argmax(counts)],
            'least_frequent_label': unique_labels[np.argmin(counts)],
            'label_balance_ratio': np.min(counts) / np.max(counts)
        }
    
    def _log_to_file(self, debug_info: MLDebugInfo):
        """Log strutturato su file"""
        log_data = {
            'timestamp': debug_info.timestamp,
            'session_id': debug_info.session_id,
            'phase': debug_info.phase,
            'feature_info': debug_info.feature_info,
            'ml_predictions': debug_info.ml_predictions,
            'llm_prediction': debug_info.llm_prediction,
            'ensemble_result': debug_info.ensemble_result,
            'weights_used': debug_info.weights_used,
            'processing_time': debug_info.processing_time,
            'error': debug_info.error,
            'context': debug_info.context
        }
        
        # Converte i tipi NumPy in tipi Python nativi prima della serializzazione JSON
        log_data_safe = convert_numpy_types(log_data)
        self.logger.debug(json.dumps(log_data_safe, ensure_ascii=False, indent=2))
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Ritorna summary delle informazioni di debug"""
        if not self.enabled or not self.debug_history:
            return {'debug_enabled': False, 'total_operations': 0}
        
        total_ops = len(self.debug_history)
        phases = {}
        avg_processing_time = 0.0
        errors = 0
        
        for debug_info in self.debug_history:
            phases[debug_info.phase] = phases.get(debug_info.phase, 0) + 1
            avg_processing_time += debug_info.processing_time
            if debug_info.error:
                errors += 1
        
        avg_processing_time /= total_ops
        
        return {
            'debug_enabled': True,
            'total_operations': total_ops,
            'phases': phases,
            'avg_processing_time': round(avg_processing_time, 3),
            'errors': errors,
            'error_rate': round(errors / total_ops, 3) if total_ops > 0 else 0
        }
