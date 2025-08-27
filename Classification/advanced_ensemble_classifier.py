#!/usr/bin/env python3
"""
Advanced Ensemble Classifier - Combinazione intelligente di LLM e ML
per massima accuratezza e robustezza nella classificazione delle conversazioni
"""

import numpy as np
import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Aggiungi il path per importare il modulo di supervisione umana
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from HumanSupervision.human_supervisor import HumanSupervision  # DEPRECATO: sostituito da QualityGateEngine

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Import della funzione centralizzata per conversione NumPy
from Utils.numpy_serialization import convert_numpy_types

# Import per debugging ML
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Debug'))
try:
    from ml_ensemble_debugger import MLEnsembleDebugger
    ML_DEBUGGER_AVAILABLE = True
except ImportError:
    MLEnsembleDebugger = None
    ML_DEBUGGER_AVAILABLE = False

class AdvancedEnsembleClassifier:
    """
    Advanced Ensemble Classifier che combina:
    1. LLM (Ollama/Mistral) - Per comprensione linguaggio naturale
    2. ML Classifiers (RF, SVM, LR) - Per pattern statistici
    3. Voting intelligente - Basato su confidenza e contesto
    4. Adaptive weighting - Pesi dinamici basati su performance
    """
    
    def __init__(self, 
                 llm_classifier=None,
                 confidence_threshold: float = 0.7,
                 adaptive_weights: bool = True,
                 performance_tracking: bool = True,
                 client_name: str = None,
                 config_path: str = None):
        """
        Inizializza l'ensemble classifier avanzato
        
        Args:
            llm_classifier: Classificatore LLM (IntelligentClassifier)
            confidence_threshold: Soglia minima di confidenza
            adaptive_weights: Se True, adatta i pesi basandosi sulle performance
            performance_tracking: Se True, traccia le performance per ottimizzazione
            client_name: Nome del cliente per fine-tuning automatico
            config_path: Percorso del file di configurazione per debug
        """
        self.llm_classifier = llm_classifier
        self.client_name = client_name
        
        # Inizializza ML Debugger
        self.ml_debugger = None
        if ML_DEBUGGER_AVAILABLE:
            try:
                self.ml_debugger = MLEnsembleDebugger(config_path=config_path)
                if self.ml_debugger.enabled:
                    print(f"🔍 ML Ensemble Debugger attivato")
            except Exception as e:
                print(f"⚠️ ML Ensemble Debugger non disponibile: {e}")
                self.ml_debugger = None
        
        # MODIFICA CRITICA: Usa LLMFactory per gestione dinamica modelli LLM
        # Se non viene passato un LLM classifier, creane uno tramite factory
        if self.llm_classifier is None:
            try:
                print("🔧 Tentativo di inizializzazione IntelligentClassifier tramite LLMFactory...")
                
                # Verifica memoria GPU prima di procedere
                try:
                    import torch
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        free = total - allocated
                        print(f"🧠 GPU Memory prima di LLM: {allocated:.1f}GB/{total:.1f}GB (free: {free:.1f}GB)")
                        
                        # Se poca memoria disponibile, salta LLM
                        if free < 1.5:  # Serve almeno 1.5GB per IntelligentClassifier
                            print(f"⚠️ GPU memory insufficiente ({free:.1f}GB < 1.5GB), skip LLM")
                            self.llm_classifier = None
                        else:
                            # CORREZIONE CRITICA: Usa LLMFactory per gestione tenant-aware
                            print("✅ Memoria GPU sufficiente, inizializzazione LLM tramite Factory...")
                            
                            try:
                                # Import LLMFactory per gestione dinamica
                                from llm_factory import llm_factory
                                
                                # Ottieni classifier tramite factory (tenant-aware)
                                if client_name:
                                    self.llm_classifier = llm_factory.get_llm_for_tenant(client_name)
                                    print(f"🏭 LLM classifier ottenuto tramite Factory per tenant {client_name}")
                                else:
                                    # Fallback per client generico
                                    self.llm_classifier = llm_factory.get_llm_for_tenant("default")
                                    print(f"🏭 LLM classifier ottenuto tramite Factory per tenant default")
                                
                                if self.llm_classifier and self.llm_classifier.is_available():
                                    current_model = getattr(self.llm_classifier, 'model_name', 'unknown')
                                    print(f"🤖 LLM classifier creato automaticamente nell'ensemble: {current_model}")
                                    if client_name and hasattr(self.llm_classifier, 'has_finetuned_model'):
                                        if self.llm_classifier.has_finetuned_model():
                                            print(f"🎯 Modello fine-tuned attivo per {client_name}")
                                        else:
                                            print(f"💡 Possibile fine-tuning per {client_name}")
                                else:
                                    print("⚠️ LLM da Factory non disponibile, ensemble userà solo ML")
                                    self.llm_classifier = None
                                    
                            except ImportError as factory_e:
                                print(f"⚠️ LLMFactory non disponibile ({factory_e}), fallback alla creazione diretta")
                                
                                # Fallback: creazione diretta (compatibilità)
                                from Classification.intelligent_classifier import IntelligentClassifier
                                self.llm_classifier = IntelligentClassifier(
                                    client_name=client_name,
                                    enable_finetuning=True
                                )
                                if self.llm_classifier.is_available():
                                    print("🤖 LLM classifier creato direttamente (fallback)")
                                else:
                                    print("⚠️ LLM fallback non disponibile, ensemble userà solo ML")
                                    self.llm_classifier = None
                    else:
                        print("⚠️ CUDA non disponibile, ensemble userà solo ML")
                        self.llm_classifier = None
                except Exception as gpu_e:
                    print(f"⚠️ Errore verifica GPU: {gpu_e}, ensemble userà solo ML")
                    self.llm_classifier = None
                
            except ImportError as e:
                print(f"⚠️ IntelligentClassifier non importabile ({e}), ensemble userà solo ML")
                self.llm_classifier = None
            except Exception as e:
                print(f"⚠️ Errore inizializzazione LLM ({e}), ensemble userà solo ML")
                self.llm_classifier = None
        self.confidence_threshold = confidence_threshold
        self.adaptive_weights = adaptive_weights
        self.performance_tracking = performance_tracking
        
        # Ensemble di classificatori ML
        self.ml_ensemble = None
        self.ml_classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Pesi dinamici per ogni classificatore
        self.weights = {
            'llm': 0.6,        # Peso iniziale LLM
            'ml_ensemble': 0.4  # Peso iniziale ML ensemble
        }
        
        # Tracking delle performance
        self.performance_history = []
        self.prediction_cache = {}
        
        # Metriche di performance per adaptive weighting
        self.performance_metrics = {
            'llm': {'accuracy': 0.85, 'confidence_reliability': 0.90},
            'ml_ensemble': {'accuracy': 0.80, 'confidence_reliability': 0.85}
        }
        
        # Integrazione BERTopic (feature augmentation)
        self.bertopic_provider = None
        self.bertopic_top_k = 15
        self.bertopic_return_one_hot = False
        
        # NOTA: HumanSupervision è stato sostituito da QualityGateEngine
        # per gestione asincrona dei disaccordi tramite REST API
        
        print("🔗 Advanced Ensemble Classifier inizializzato")
        print(f"   🧠 LLM weight: {self.weights['llm']:.2f}")
        print(f"   🤖 ML weight: {self.weights['ml_ensemble']:.2f}")
        print(f"   🎯 Confidence threshold: {confidence_threshold}")
        print(f"   🔄 Adaptive weights: {adaptive_weights}")
        print(f"   👤 Disaccordi gestiti da QualityGateEngine")
    
    def train_ml_ensemble_with_descriptions(self, X_train: np.ndarray, y_train: np.ndarray, 
                                           tag_descriptions: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Allena l'ensemble ML con supporto per descrizioni dei tag (feature enhancement)
        
        Args:
            X_train: Features di training (embeddings originali)
            y_train: Labels di training
            tag_descriptions: Dizionario tag_name -> description per arricchimento
            
        Returns:
            Metriche di training
        """
        print("🎓 Training ML ensemble con descrizioni dei tag...")
        start_time = datetime.now()
        
        try:
            # Se sono fornite descrizioni, arricchisce il training data
            if tag_descriptions:
                print("   📚 Arricchimento dati con descrizioni dei tag...")
                X_enhanced = self._enhance_features_with_descriptions(X_train, y_train, tag_descriptions)
                training_features = X_enhanced
                enhancement_info = "with_tag_descriptions"
            else:
                training_features = X_train
                enhancement_info = "standard"
            
            # Crea voting classifier con i migliori algoritmi
            self.ml_ensemble = VotingClassifier(
                estimators=[
                    ('rf', self.ml_classifiers['random_forest']),
                    ('svm', self.ml_classifiers['svm']),
                    ('lr', self.ml_classifiers['logistic_regression'])
                ],
                voting='soft'  # Usa le probabilità per il voting
            )
            
            # Training con features arricchite
            self.ml_ensemble.fit(training_features, y_train)
            
            # Calcola accuracy sul training set
            train_accuracy = self.ml_ensemble.score(training_features, y_train)
            
            # Aggiorna metriche
            self.performance_metrics['ml_ensemble']['accuracy'] = train_accuracy
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            training_result = {
                'training_accuracy': float(train_accuracy),
                'n_samples': int(len(training_features)),
                'n_features': int(training_features.shape[1]),
                'n_classes': int(len(np.unique(y_train))),
                'enhancement_method': enhancement_info,
                'processing_time': processing_time
            }
            
            # DEBUG: Log training details
            if self.ml_debugger and self.ml_debugger.enabled:
                session_id = f"training_enhanced_{int(start_time.timestamp())}"
                self.ml_debugger.debug_training(
                    session_id=session_id,
                    X_train=training_features,
                    y_train=y_train,
                    training_result=training_result,
                    processing_time=processing_time
                )
            
            print(f"✅ ML ensemble training completato con descrizioni!")
            print(f"   📊 Training accuracy: {train_accuracy:.3f}")
            print(f"   🎯 Features: {training_features.shape[1]} (enhancement: {enhancement_info})")
            print(f"   🤖 Algoritmi: Random Forest + SVM + Logistic Regression")
            
            return training_result
            
        except Exception as e:
            print(f"❌ Errore durante training ML ensemble: {e}")
            return self.train_ml_ensemble(X_train, y_train)  # Fallback al training standard

    def _enhance_features_with_descriptions(self, X_train: np.ndarray, y_train: np.ndarray, 
                                          tag_descriptions: Dict[str, str]) -> np.ndarray:
        """
        Arricchisce le features con embeddings delle descrizioni dei tag
        
        Args:
            X_train: Features originali (embeddings del testo)
            y_train: Labels
            tag_descriptions: Descrizioni dei tag
            
        Returns:
            Features arricchite
        """
        try:
            from EmbeddingEngine.labse_embedder import LaBSEEmbedder
            embedder = LaBSEEmbedder()
            
            # Crea embeddings delle descrizioni per ogni classe
            description_embeddings = {}
            for tag_name, description in tag_descriptions.items():
                if description:
                    desc_embedding = embedder.encode([description])[0]
                    description_embeddings[tag_name] = desc_embedding
            
            # Per ogni sample, concatena embedding originale con embedding della descrizione del tag
            enhanced_features = []
            for i, label in enumerate(y_train):
                original_features = X_train[i]
                
                if label in description_embeddings:
                    # Concatena features originali + descrizione del tag
                    desc_features = description_embeddings[label]
                    enhanced = np.concatenate([original_features, desc_features])
                else:
                    # Se non c'è descrizione, usa padding di zeri
                    desc_size = len(next(iter(description_embeddings.values())))
                    padding = np.zeros(desc_size)
                    enhanced = np.concatenate([original_features, padding])
                
                enhanced_features.append(enhanced)
            
            enhanced_array = np.array(enhanced_features)
            print(f"   🔧 Features arricchite: {X_train.shape[1]} -> {enhanced_array.shape[1]}")
            
            return enhanced_array
            
        except Exception as e:
            print(f"⚠️ Errore nell'arricchimento features: {e}, uso features standard")
            return X_train

    def train_ml_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Allena l'ensemble di classificatori ML con features fornite (già
        arricchite se necessario a monte della pipeline).
        
        Args:
            X_train: Features di training
            y_train: Labels di training
        
        Returns:
            Metriche di training
        """
        print("🎓 Training ML ensemble...")
        start_time = datetime.now()
        
        # Crea voting classifier con i migliori algoritmi
        self.ml_ensemble = VotingClassifier(
            estimators=[
                ('rf', self.ml_classifiers['random_forest']),
                ('svm', self.ml_classifiers['svm']),
                ('lr', self.ml_classifiers['logistic_regression'])
            ],
            voting='soft'  # Usa le probabilità per il voting
        )
        
        # Training
        self.ml_ensemble.fit(X_train, y_train)
        
        # Calcola accuracy sul training set
        train_accuracy = self.ml_ensemble.score(X_train, y_train)
        
        # Aggiorna metriche
        self.performance_metrics['ml_ensemble']['accuracy'] = train_accuracy
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        training_result = {
            'training_accuracy': float(train_accuracy),
            'n_samples': int(len(X_train)),
            'n_features': int(X_train.shape[1]),
            'n_classes': int(len(np.unique(y_train)))
        }
        
        # DEBUG: Log training details
        if self.ml_debugger and self.ml_debugger.enabled:
            session_id = f"training_{int(start_time.timestamp())}"
            self.ml_debugger.debug_training(
                session_id=session_id,
                X_train=X_train,
                y_train=y_train,
                training_result=training_result,
                processing_time=processing_time
            )
        
        print(f"✅ ML ensemble training completato!")
        print(f"   📊 Training accuracy: {train_accuracy:.3f}")
        print(f"   🎯 Algoritmi: Random Forest + SVM + Logistic Regression")
        
        return training_result
    
    def prepare_enhanced_training_data(self, texts: List[str], labels: List[str], 
                                     tag_descriptions: Optional[Dict[str, str]] = None) -> List[str]:
        """
        Prepara dati di training arricchiti con descrizioni dei tag
        
        Args:
            texts: Lista di testi originali
            labels: Lista di etichette corrispondenti
            tag_descriptions: Dizionario tag_name -> description (opzionale)
            
        Returns:
            Lista di testi arricchiti con contesto semantico
        """
        if tag_descriptions is None:
            # Tenta di recuperare descrizioni dal database
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TagDatabase'))
                from tag_database_connector import TagDatabaseConnector
                tag_db = TagDatabaseConnector()
                tags = tag_db.get_all_tags()
                tag_descriptions = {tag['tag_name']: tag['tag_description'] for tag in tags}
                print(f"🏷️ Recuperate {len(tag_descriptions)} descrizioni tag dal database per training ML")
            except Exception as e:
                print(f"⚠️ Impossibile recuperare descrizioni tag: {e}")
                tag_descriptions = {}
        
        enhanced_texts = []
        for text, label in zip(texts, labels):
            description = tag_descriptions.get(label, "")
            if description:
                # Arricchisce il testo con il contesto semantico del tag
                enhanced_text = f"{text} [CATEGORIA: {label} - {description}]"
            else:
                enhanced_text = text
            enhanced_texts.append(enhanced_text)
        
        print(f"🔍 Training ML arricchito: {len([t for t in enhanced_texts if '[CATEGORIA:' in t])}/{len(texts)} testi con descrizioni")
        return enhanced_texts
    
    def predict_with_ensemble(self, text: str, return_details: bool = False, embedder=None) -> Dict[str, Any]:
        """
        Predizione ensemble combinando LLM e ML
        
        Args:
            text: Testo da classificare
            return_details: Se True, ritorna dettagli di ogni classificatore
            embedder: Embedder già inizializzato da riutilizzare (evita CUDA OOM)
            
        Returns:
            Risultato della predizione ensemble
        """
        results = {
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        embedding = None  # Per il debug
        
        # 1. Predizione LLM
        llm_prediction = None
        llm_confidence = 0.0
        llm_available = False
        
        if self.llm_classifier and self.llm_classifier.is_available():
            try:
                llm_result = self.llm_classifier.classify_with_motivation(text)
                # Gestisci correttamente l'oggetto ClassificationResult
                llm_prediction = {
                    'predicted_label': llm_result.predicted_label,
                    'confidence': llm_result.confidence,
                    'motivation': llm_result.motivation
                }
                llm_confidence = llm_result.confidence
                llm_available = True
            except Exception as e:
                print(f"⚠️ Errore LLM: {e}")
                llm_prediction = None
        
        # 2. Predizione ML Ensemble
        ml_prediction = None
        ml_confidence = 0.0
        ml_available = False
        
        if self.ml_ensemble is not None:
            try:
                # Usa l'embedder passato come parametro o creane uno nuovo solo se necessario
                if embedder is not None:
                    embedding = embedder.encode([text])
                else:
                    print("⚠️ Nessun embedder fornito, creo uno nuovo (potenziale rischio CUDA OOM)")
                    from EmbeddingEngine.labse_embedder import LaBSEEmbedder
                    temp_embedder = LaBSEEmbedder()
                    embedding = temp_embedder.encode([text])
                
                # Applica feature augmentation BERTopic se disponibile
                ml_features = embedding
                if self.bertopic_provider is not None:
                    try:
                        topic_feats = self.bertopic_provider.transform(
                            [text], embeddings=embedding,
                            top_k=self.bertopic_top_k,
                            return_one_hot=self.bertopic_return_one_hot
                        )
                        parts = [embedding, topic_feats.get('topic_probas')]
                        if self.bertopic_return_one_hot and 'one_hot' in topic_feats:
                            parts.append(topic_feats['one_hot'])
                        ml_features = np.concatenate([p for p in parts if p is not None], axis=1)
                    except Exception as be:
                        print(f"⚠️ BERTopic transform fallita: {be}. Uso solo embedding base.")
                        ml_features = embedding
                
                # Predizione
                ml_proba = self.ml_ensemble.predict_proba(ml_features)[0]
                ml_label_idx = np.argmax(ml_proba)
                ml_label = self.ml_ensemble.classes_[ml_label_idx]
                ml_confidence = ml_proba[ml_label_idx]
                
                # CONVERSIONE NUMPY -> PYTHON per evitare errori JSON serialization
                ml_prediction = {
                    'predicted_label': convert_numpy_types(ml_label),
                    'confidence': convert_numpy_types(ml_confidence),
                    'probabilities': convert_numpy_types(dict(zip(self.ml_ensemble.classes_, ml_proba)))
                }
                ml_available = True
                
            except Exception as e:
                print(f"⚠️ Errore ML Ensemble: {e}")
                ml_prediction = None
        
        # 3. Combinazione Ensemble
        if llm_available and ml_available:
            # Entrambi disponibili - usa voting ponderato con supervisione umana
            ensemble_prediction = self._combine_predictions(llm_prediction, ml_prediction, text)
            method = 'ENSEMBLE'
        elif llm_available:
            # Solo LLM disponibile
            ensemble_prediction = llm_prediction.copy()
            ensemble_prediction['ensemble_confidence'] = llm_confidence
            method = 'LLM'
        elif ml_available:
            # Solo ML disponibile
            ensemble_prediction = ml_prediction.copy()
            ensemble_prediction['ensemble_confidence'] = ml_confidence
            method = 'ML'
        else:
            # Nessuno disponibile - errore
            raise ValueError("Nessun classificatore disponibile")
        
        # 4. Risultato finale
        final_confidence = ensemble_prediction.get('ensemble_confidence', 0.0)
        is_high_confidence = final_confidence >= self.confidence_threshold
        
        # CONVERSIONE NUMPY -> PYTHON per risultati finali
        results.update({
            'predicted_label': convert_numpy_types(ensemble_prediction['predicted_label']),
            'confidence': convert_numpy_types(final_confidence),
            'is_high_confidence': is_high_confidence,
            'method': method,
            'ensemble_confidence': convert_numpy_types(final_confidence)
        })
        
        # Aggiungi dettagli se richiesti
        if return_details:
            results.update({
                'llm_prediction': convert_numpy_types(llm_prediction),
                'ml_prediction': convert_numpy_types(ml_prediction),
                'llm_available': llm_available,
                'ml_available': ml_available,
                'weights_used': convert_numpy_types(self.weights.copy()),
                'performance_metrics': convert_numpy_types(self.performance_metrics.copy())
            })
        
        # Tracking performance
        if self.performance_tracking:
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'confidence': final_confidence,
                'llm_available': llm_available,
                'ml_available': ml_available
            })
        
        # DEBUG: Log predizione completa
        processing_time = (datetime.now() - start_time).total_seconds()
        if self.ml_debugger and self.ml_debugger.enabled:
            session_id = f"predict_{int(start_time.timestamp())}"
            self.ml_debugger.debug_prediction(
                session_id=session_id,
                input_text=text,
                features=embedding[0] if embedding is not None else None,
                ml_predictions=ml_prediction,
                llm_prediction=llm_prediction,
                ensemble_result=results,
                weights_used=self.weights,
                processing_time=processing_time
            )
        
        return results
    
    def _combine_predictions(self, llm_pred: Dict, ml_pred: Dict, text: str = "") -> Dict[str, Any]:
        """
        Combina le predizioni LLM e ML usando voting intelligente con supervisione umana
        
        Args:
            llm_pred: Predizione LLM
            ml_pred: Predizione ML
            text: Testo originale per supervisione umana
            
        Returns:
            Predizione combinata
        """
        llm_label = llm_pred['predicted_label']
        llm_conf = llm_pred['confidence']
        ml_label = ml_pred['predicted_label']
        ml_conf = ml_pred['confidence']
        
        # Calcola pesi adattivi basati su confidenza
        llm_weight = self.weights['llm']
        ml_weight = self.weights['ml_ensemble']
        
        # Bonus per alta confidenza
        if llm_conf > 0.9:
            llm_weight *= 1.2
        if ml_conf > 0.9:
            ml_weight *= 1.2
        
        # Normalizza pesi
        total_weight = llm_weight + ml_weight
        llm_weight /= total_weight
        ml_weight /= total_weight
        
        # Decisione finale
        if llm_label == ml_label:
            # Accordo - usa confidenza ponderata
            final_label = llm_label
            final_confidence = (llm_conf * llm_weight + ml_conf * ml_weight)
            agreement = True
            human_intervention = False
            method = 'weighted_voting'
        else:
            # Disaccordo - usa weighted voting automatico
            # I disaccordi saranno gestiti da QualityGateEngine per review asincrona
            agreement = False
            
            # Calcola pesi aggiustati per confidence
            adjusted_llm_weight = llm_weight * llm_conf
            adjusted_ml_weight = ml_weight * ml_conf
            
            if adjusted_llm_weight > adjusted_ml_weight:
                final_label = llm_label
                final_confidence = llm_conf * 0.8  # Penalizza per disaccordo
                method = 'LLM_DISAGREEMENT'
            else:
                final_label = ml_label
                final_confidence = ml_conf * 0.8  # Penalizza per disaccordo
                method = 'ML_DISAGREEMENT'
            
            human_intervention = False
        
        # Prepara risultato per debug
        voting_result = {
            'predicted_label': convert_numpy_types(final_label),
            'ensemble_confidence': convert_numpy_types(final_confidence),
            'agreement': agreement,
            'human_intervention': human_intervention,
            'llm_weight_used': convert_numpy_types(llm_weight),
            'ml_weight_used': convert_numpy_types(ml_weight),
            'combination_method': method
        }
        
        # DEBUG: Log processo di voting
        if self.ml_debugger and self.ml_debugger.enabled:
            session_id = f"voting_{int(datetime.now().timestamp())}"
            voting_process = {
                'agreement': agreement,
                'combination_method': method,
                'weights_used': {'llm': llm_weight, 'ml': ml_weight},
                'confidence_adjustment': 0.8 if not agreement else 1.0
            }
            self.ml_debugger.debug_ensemble_voting(
                session_id=session_id,
                ml_pred=ml_pred,
                llm_pred=llm_pred,
                voting_process=voting_process,
                final_result=voting_result,
                processing_time=0.001  # Voting è veloce
            )
        
        return voting_result
    
    def update_adaptive_weights(self, feedback_data: List[Dict]) -> None:
        """
        Aggiorna i pesi in base ai feedback di performance
        
        Args:
            feedback_data: Lista di feedback con ground truth
        """
        if not self.adaptive_weights:
            return
        
        print("🔄 Aggiornamento pesi adattivi...")
        
        llm_correct = 0
        ml_correct = 0
        total_samples = len(feedback_data)
        
        for feedback in feedback_data:
            if feedback.get('llm_correct', False):
                llm_correct += 1
            if feedback.get('ml_correct', False):
                ml_correct += 1
        
        # Calcola nuove accuracies
        llm_accuracy = llm_correct / total_samples if total_samples > 0 else 0.5
        ml_accuracy = ml_correct / total_samples if total_samples > 0 else 0.5
        
        # Aggiorna pesi basandosi su performance relative
        total_acc = llm_accuracy + ml_accuracy
        if total_acc > 0:
            self.weights['llm'] = llm_accuracy / total_acc
            self.weights['ml_ensemble'] = ml_accuracy / total_acc
        
        # Aggiorna metriche
        self.performance_metrics['llm']['accuracy'] = llm_accuracy
        self.performance_metrics['ml_ensemble']['accuracy'] = ml_accuracy
        
        print(f"✅ Pesi aggiornati:")
        print(f"   🧠 LLM: {self.weights['llm']:.3f} (acc: {llm_accuracy:.3f})")
        print(f"   🤖 ML: {self.weights['ml_ensemble']:.3f} (acc: {ml_accuracy:.3f})")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """
        Ritorna statistiche complete dell'ensemble
        
        Returns:
            Statistiche dell'ensemble
        """
        stats = {
            'configuration': {
                'weights': self.weights.copy(),
                'confidence_threshold': self.confidence_threshold,
                'adaptive_weights': self.adaptive_weights,
                'performance_tracking': self.performance_tracking
            },
            'performance_metrics': self.performance_metrics.copy(),
            'prediction_history': {
                'total_predictions': len(self.performance_history),
                'methods_used': {}
            }
        }
        
        # Analizza metodi usati
        if self.performance_history:
            methods = [p['method'] for p in self.performance_history]
            for method in set(methods):
                stats['prediction_history']['methods_used'][method] = methods.count(method)
        
        return stats
    
    def save_ensemble_model(self, model_path: str) -> None:
        """
        Salva il modello ensemble
        
        Args:
            model_path: Percorso dove salvare il modello
        """
        print(f"💾 Salvataggio ensemble model...")
        
        # Crea directory se non esiste
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Salva ML ensemble
        if self.ml_ensemble:
            joblib.dump(self.ml_ensemble, f"{model_path}_ml_ensemble.pkl")
        
        # Salva configurazione e statistiche
        config_data = {
            'weights': self.weights,
            'confidence_threshold': self.confidence_threshold,
            'adaptive_weights': self.adaptive_weights,
            'performance_metrics': self.performance_metrics,
            'performance_history': self.performance_history[-100:]  # Ultimi 100
        }
        
        with open(f"{model_path}_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Ensemble model salvato in {model_path}")
    
    def load_ensemble_model(self, model_path: str) -> None:
        """
        Carica il modello ensemble
        
        Args:
            model_path: Percorso del modello da caricare
        """
        print(f"📥 Caricamento ensemble model...")
        
        # Carica ML ensemble
        ml_path = f"{model_path}_ml_ensemble.pkl"
        if os.path.exists(ml_path):
            self.ml_ensemble = joblib.load(ml_path)
            print(f"✅ ML ensemble caricato")
        
        # Carica configurazione
        config_path = f"{model_path}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.weights = config_data.get('weights', self.weights)
            self.performance_metrics = config_data.get('performance_metrics', self.performance_metrics)
            self.performance_history = config_data.get('performance_history', [])
            
            print(f"✅ Configurazione caricata")
        
        print(f"🔗 Ensemble model caricato da {model_path}")
    
    def batch_predict(self, texts: List[str], batch_size: int = 32, embedder=None) -> List[Dict[str, Any]]:
        """
        Predizione batch per efficienza
        
        Args:
            texts: Lista di testi da classificare
            batch_size: Dimensione del batch
            embedder: Embedder da riutilizzare per evitare CUDA OOM
            
        Returns:
            Lista di predizioni
        """
        print(f"📦 Batch prediction di {len(texts)} testi...")
        
        # Crea embedder se non fornito
        if embedder is None:
            from EmbeddingEngine.labse_embedder import LaBSEEmbedder
            embedder = LaBSEEmbedder()
            print("✅ Embedder creato per il batch")
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                try:
                    # Passa l'embedder riutilizzabile
                    prediction = self.predict_with_ensemble(text, return_details=True, embedder=embedder)
                    batch_results.append(prediction)
                except Exception as e:
                    print(f"⚠️ Errore nella predizione: {e}")
                    batch_results.append({
                        'predicted_label': 'altro',
                        'confidence': 0.1,
                        'is_high_confidence': False,
                        'method': 'ERROR_FALLBACK',
                        'llm_prediction': None,
                        'ml_prediction': None,
                        'ensemble_confidence': 0.1,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
            
            # Progress feedback
            progress = min(i + batch_size, len(texts))
            print(f"   Progresso: {progress}/{len(texts)} (riutilizzando embedder)")
        
        print(f"✅ Batch prediction completata con embedder riutilizzabile")
        return results
    
    def predict_with_llm_only(self, text: str, return_details: bool = False) -> Dict[str, Any]:
        """
        Esegue classificazione usando SOLO il LLM (per nuovi clienti senza modello ML).
        
        Args:
            text: Testo da classificare
            return_details: Se True, restituisce dettagli aggiuntivi
            
        Returns:
            Risultato della classificazione LLM
        """
        if not self.llm_classifier or not self.llm_classifier.is_available():
            raise RuntimeError("LLM classifier non disponibile per classificazione standalone")
        
        try:
            llm_result = self.llm_classifier.classify_with_motivation(text)
            
            result = {
                'predicted_label': convert_numpy_types(llm_result.predicted_label),
                'confidence': convert_numpy_types(llm_result.confidence),
                'method': 'llm_standalone'
            }
            
            if return_details:
                result.update({
                    'motivation': convert_numpy_types(llm_result.motivation),
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'timestamp': datetime.now().isoformat(),
                    'classifier_type': 'llm_only'
                })
            
            return result
            
        except Exception as e:
            print(f"❌ Errore nella classificazione LLM standalone: {e}")
            return {
                'predicted_label': '',
                'confidence': 0.0,
                'error': str(e),
                'method': 'llm_standalone_failed'
            }
    
    def get_tags_with_descriptions_for_training(self) -> Dict[str, str]:
        """
        Recupera tag e descrizioni dal database per arricchire il training ML
        
        Returns:
            Dizionario tag_name -> description
        """
        try:
            # Import locale per evitare problemi di dipendenze circolari
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Database'))
            from Database.schema_manager import ClassificationSchemaManager
            
            schema_manager = ClassificationSchemaManager()
            tags = schema_manager.get_all_tags()
            
            tags_dict = {}
            for tag in tags:
                tag_name = tag.get('tag_name', '')
                tag_description = tag.get('tag_description', '')
                if tag_name:
                    tags_dict[tag_name] = tag_description or f"Categoria {tag_name}"
            
            print(f"📋 Recuperati {len(tags_dict)} tag con descrizioni per training ML")
            return tags_dict
            
        except Exception as e:
            print(f"⚠️ Errore recupero tag dal database: {e}")
            # Fallback ai tag hardcoded
            return {
                'ritiro_cartella_clinica_referti': 'Richieste di ritiro documentazione medica, cartelle cliniche e referti',
                'prenotazione_esami': 'Prenotazioni e appuntamenti per esami diagnostici, visite specialistiche',
                'info_contatti': 'Richieste di informazioni su contatti, numeri di telefono, reparti',
                'problema_accesso_portale': 'Problemi tecnici di accesso al portale online, credenziali',
                'info_esami': 'Informazioni generali su procedure di esami, modalità, tempi',
                'cambio_anagrafica': 'Modifiche dati anagrafici, aggiornamento informazioni personali',
                'norme_di_preparazione': 'Istruzioni per preparazione esami, digiuno, farmaci',
                'problema_amministrativo': 'Questioni amministrative, pagamenti, ticket, rimborsi',
                'info_ricovero': 'Informazioni su ricoveri, degenze, procedure ospedaliere',
                'altro': 'Richieste non classificabili nelle categorie principali'
            }
    
    def reload_llm_configuration(self, tenant_id: str = None) -> Dict[str, Any]:
        """
        Ricarica configurazione LLM per il tenant corrente
        
        FUNZIONE CRITICA: Implementa reload dinamico del modello LLM
        usando LLMFactory per sincronizzare con le modifiche da configurazione React.
        
        Args:
            tenant_id: ID del tenant (usa self.client_name se None)
            
        Returns:
            Risultato del reload con dettagli
            
        Ultima modifica: 26 Agosto 2025
        """
        effective_tenant = tenant_id or self.client_name or "default"
        
        try:
            print(f"🔄 RELOAD LLM CONFIGURATION per tenant {effective_tenant}")
            
            # Import LLMFactory per gestione dinamica
            try:
                from llm_factory import llm_factory
            except ImportError:
                return {
                    'success': False,
                    'error': 'LLMFactory non disponibile'
                }
            
            # Forza reload tramite factory
            old_model = getattr(self.llm_classifier, 'model_name', 'unknown') if self.llm_classifier else 'none'
            
            try:
                # Ottieni nuovo classifier con configurazione aggiornata
                new_classifier = llm_factory.get_llm_for_tenant(effective_tenant, force_reload=True)
                
                if new_classifier and new_classifier.is_available():
                    new_model = getattr(new_classifier, 'model_name', 'unknown')
                    
                    # Sostituisci classifier corrente
                    self.llm_classifier = new_classifier
                    
                    print(f"✅ LLM Classifier ricaricato: {old_model} -> {new_model}")
                    
                    return {
                        'success': True,
                        'old_model': old_model,
                        'new_model': new_model,
                        'tenant_id': effective_tenant,
                        'classifier_available': True
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Nuovo classifier non disponibile',
                        'tenant_id': effective_tenant
                    }
                    
            except Exception as factory_e:
                print(f"❌ Errore LLMFactory reload: {factory_e}")
                return {
                    'success': False,
                    'error': f'Errore factory: {str(factory_e)}',
                    'tenant_id': effective_tenant
                }
                
        except Exception as e:
            print(f"❌ Errore reload LLM configuration: {e}")
            return {
                'success': False,
                'error': f'Errore reload: {str(e)}',
                'tenant_id': effective_tenant
            }
    
    def get_current_llm_info(self) -> Dict[str, Any]:
        """
        Ottiene informazioni sul classificatore LLM corrente
        
        Returns:
            Informazioni dettagliate su LLM classifier
        """
        if not self.llm_classifier:
            return {
                'llm_available': False,
                'error': 'Nessun LLM classifier configurato'
            }
        
        try:
            return {
                'llm_available': self.llm_classifier.is_available(),
                'model_name': getattr(self.llm_classifier, 'model_name', 'unknown'),
                'base_model': getattr(self.llm_classifier, 'base_model_name', 'unknown'),
                'client_name': getattr(self.llm_classifier, 'client_name', 'unknown'),
                'ollama_url': getattr(self.llm_classifier, 'ollama_url', 'unknown'),
                'temperature': getattr(self.llm_classifier, 'temperature', 'unknown'),
                'max_tokens': getattr(self.llm_classifier, 'max_tokens', 'unknown'),
                'cache_enabled': getattr(self.llm_classifier, 'enable_cache', False),
                'finetuning_enabled': getattr(self.llm_classifier, 'enable_finetuning', False)
            }
        except Exception as e:
            return {
                'llm_available': False,
                'error': f'Errore informazioni LLM: {str(e)}'
            }

    def set_bertopic_provider(self, provider, top_k: int = 15, return_one_hot: bool = False) -> None:
        """
        Imposta il provider BERTopic per feature augmentation in predizione.
        
        Args:
            provider: Istanza di BERTopicFeatureProvider già addestrata
            top_k: Numero massimo di topic da mantenere
            return_one_hot: Se includere one-hot del topic principale
        """
        self.bertopic_provider = provider
        self.bertopic_top_k = top_k
        self.bertopic_return_one_hot = return_one_hot
        print(f"🔗 BERTopic provider impostato (top_k={top_k}, one_hot={return_one_hot})")
