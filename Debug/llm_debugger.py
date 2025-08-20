"""
LLM Debugger - Sistema di debugging avanzato per il classificatore LLM
Fornisce debug dettagliato e visivamente strutturato di input, output e processing LLM
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import traceback
import sys

# Aggiungo il path per le utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Utils.numpy_serialization import convert_numpy_types

@dataclass
class LLMDebugInfo:
    """Struttura per informazioni di debug LLM"""
    timestamp: str
    session_id: str
    phase: str  # 'training', 'classification', 'validation'
    input_text: str
    prompt: str
    model_params: Dict[str, Any]
    raw_response: str
    parsed_response: Dict[str, Any]
    confidence: float
    reasoning: str
    processing_time: float
    error: Optional[str] = None
    context: Dict[str, Any] = None

class LLMDebugger:
    """
    Debugger avanzato per il sistema LLM
    Fornisce logging visivamente strutturato e analisi dettagliata
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il debugger LLM
        
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
        self.llm_debug_config = self.debug_config.get('llm_debug', {})
        
        # Check se debug è abilitato
        self.enabled = (self.debug_config.get('enabled', False) and 
                       self.llm_debug_config.get('enabled', False))
        
        if not self.enabled:
            return
        
        # Configurazione logging
        self.setup_logging()
        
        # Storage per debug info
        self.debug_history: List[LLMDebugInfo] = []
        
        print("🔍 LLM Debugger inizializzato e ATTIVO")
        print(f"   📝 Debug file: {self.debug_file}")
        print(f"   🎨 Visual formatting: {self.llm_debug_config.get('visual_formatting', True)}")
    
    def setup_logging(self):
        """Setup del sistema di logging"""
        if not self.enabled:
            return
        
        # Crea directory debug
        debug_path = self.debug_config.get('debug_files', {}).get('base_path', './debug_logs')
        os.makedirs(debug_path, exist_ok=True)
        
        # File di debug LLM
        llm_log_file = self.debug_config.get('debug_files', {}).get('llm_log', 'llm_debug.log')
        self.debug_file = os.path.join(debug_path, llm_log_file)
        
        # Setup logger
        self.logger = logging.getLogger('LLMDebugger')
        self.logger.setLevel(logging.DEBUG)
        
        # Handler per file
        if self.llm_debug_config.get('save_to_file', True):
            file_handler = logging.FileHandler(self.debug_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug_llm_call(self,
                      session_id: str,
                      phase: str,
                      input_text: str,
                      prompt: str,
                      model_params: Dict[str, Any],
                      raw_response: str,
                      parsed_response: Dict[str, Any],
                      confidence: float,
                      reasoning: str,
                      processing_time: float,
                      error: Optional[str] = None,
                      context: Dict[str, Any] = None) -> None:
        """
        Debug completo di una chiamata LLM
        
        Args:
            session_id: ID della sessione
            phase: Fase (training/classification/validation)
            input_text: Testo di input originale
            prompt: Prompt inviato al LLM
            model_params: Parametri del modello
            raw_response: Risposta grezza dal LLM
            parsed_response: Risposta parsata/strutturata
            confidence: Confidenza calcolata
            reasoning: Reasoning/motivazione
            processing_time: Tempo di elaborazione
            error: Eventuale errore
            context: Contesto aggiuntivo
        """
        if not self.enabled:
            return
        
        # Crea debug info
        debug_info = LLMDebugInfo(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            phase=phase,
            input_text=input_text,
            prompt=prompt,
            model_params=model_params,
            raw_response=raw_response,
            parsed_response=parsed_response,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            error=error,
            context=context or {}
        )
        
        # Aggiungi alla storia
        self.debug_history.append(debug_info)
        
        # Output visuale
        if self.llm_debug_config.get('visual_formatting', True):
            self._print_visual_debug(debug_info)
        
        # Log su file
        if self.llm_debug_config.get('save_to_file', True):
            self._log_to_file(debug_info)
    
    def _print_visual_debug(self, debug_info: LLMDebugInfo):
        """Stampa debug con formattazione visuale avanzata"""
        
        print("\n" + "="*100)
        print(f"🤖 LLM DEBUG - {debug_info.phase.upper()} | {debug_info.timestamp}")
        print("="*100)
        
        # Header informazioni
        print(f"📋 Session ID: {debug_info.session_id}")
        print(f"⏱️  Processing Time: {debug_info.processing_time:.3f}s")
        if debug_info.error:
            print(f"❌ Error: {debug_info.error}")
        print("-"*100)
        
        # INPUT SECTION
        if self.llm_debug_config.get('log_input', True):
            print("📥 INPUT SECTION")
            print("-"*50)
            print(f"💬 Original Text ({len(debug_info.input_text)} chars):")
            print(f"   {self._truncate_text(debug_info.input_text, 200)}")
            print()
            print(f"🎯 Prompt Template:")
            print(f"   {self._format_prompt(debug_info.prompt)}")
            print()
            print(f"⚙️  Model Parameters:")
            for key, value in debug_info.model_params.items():
                print(f"   {key}: {value}")
            print("-"*100)
        
        # OUTPUT SECTION
        if self.llm_debug_config.get('log_output', True):
            print("📤 OUTPUT SECTION")
            print("-"*50)
            print(f"🔍 Raw Response ({len(debug_info.raw_response)} chars):")
            print(f"   {self._truncate_text(debug_info.raw_response, 300)}")
            print()
            print(f"📊 Parsed Response:")
            for key, value in debug_info.parsed_response.items():
                print(f"   {key}: {value}")
            print("-"*100)
        
        # ANALYSIS SECTION
        if self.llm_debug_config.get('log_confidence', True):
            print("📈 ANALYSIS SECTION")
            print("-"*50)
            print(f"🎯 Confidence Score: {debug_info.confidence:.3f}")
            confidence_bar = "█" * int(debug_info.confidence * 20) + "░" * (20 - int(debug_info.confidence * 20))
            print(f"   [{confidence_bar}] {debug_info.confidence:.1%}")
            print()
            
        if self.llm_debug_config.get('log_reasoning', True) and debug_info.reasoning:
            print(f"🧠 Reasoning/Motivation:")
            print(f"   {self._truncate_text(debug_info.reasoning, 400)}")
            print("-"*100)
        
        # CONTEXT SECTION
        if debug_info.context:
            print("🔍 CONTEXT & METADATA")
            print("-"*50)
            for key, value in debug_info.context.items():
                print(f"   {key}: {value}")
            print("-"*100)
        
        print("✅ LLM DEBUG COMPLETED")
        print("="*100 + "\n")
    
    def _format_prompt(self, prompt: str) -> str:
        """Formatta il prompt per visualizzazione"""
        if len(prompt) <= 150:
            return prompt.replace('\n', '\\n')
        
        lines = prompt.split('\n')
        formatted_lines = []
        for i, line in enumerate(lines):
            if i < 3:  # Prime 3 righe
                formatted_lines.append(f"     Line {i+1}: {line[:100]}")
            elif i == len(lines) - 1:  # Ultima riga
                formatted_lines.append(f"     Line {i+1}: {line[:100]}")
        
        if len(lines) > 4:
            formatted_lines.insert(-1, f"     ... [{len(lines)-4} more lines] ...")
        
        return '\n' + '\n'.join(formatted_lines)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Tronca testo con ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _log_to_file(self, debug_info: LLMDebugInfo):
        """Log strutturato su file"""
        log_data = {
            'timestamp': debug_info.timestamp,
            'session_id': debug_info.session_id,
            'phase': debug_info.phase,
            'input_length': len(debug_info.input_text),
            'prompt_length': len(debug_info.prompt),
            'model_params': debug_info.model_params,
            'raw_response_length': len(debug_info.raw_response),
            'parsed_response': debug_info.parsed_response,
            'confidence': debug_info.confidence,
            'reasoning_length': len(debug_info.reasoning) if debug_info.reasoning else 0,
            'processing_time': debug_info.processing_time,
            'error': debug_info.error,
            'context': debug_info.context
        }
        
        # Log dettagliato solo se richiesto
        if self.llm_debug_config.get('log_input', True):
            log_data['input_text'] = debug_info.input_text[:500]  # Primi 500 chars
            log_data['prompt'] = debug_info.prompt[:1000]  # Primi 1000 chars
        
        if self.llm_debug_config.get('log_output', True):
            log_data['raw_response'] = debug_info.raw_response[:500]  # Primi 500 chars
        
        if self.llm_debug_config.get('log_reasoning', True):
            log_data['reasoning'] = debug_info.reasoning[:500] if debug_info.reasoning else None
        
        # Converte i tipi NumPy in tipi Python nativi prima della serializzazione JSON
        log_data_safe = convert_numpy_types(log_data)
        self.logger.debug(json.dumps(log_data_safe, ensure_ascii=False, indent=2))
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Ritorna un summary delle informazioni di debug"""
        if not self.enabled or not self.debug_history:
            return {'debug_enabled': False, 'total_calls': 0}
        
        total_calls = len(self.debug_history)
        phases = {}
        avg_confidence = 0.0
        avg_processing_time = 0.0
        errors = 0
        
        for debug_info in self.debug_history:
            # Count per phase
            phases[debug_info.phase] = phases.get(debug_info.phase, 0) + 1
            
            # Average confidence
            avg_confidence += debug_info.confidence
            
            # Average processing time
            avg_processing_time += debug_info.processing_time
            
            # Count errors
            if debug_info.error:
                errors += 1
        
        avg_confidence /= total_calls
        avg_processing_time /= total_calls
        
        return {
            'debug_enabled': True,
            'total_calls': total_calls,
            'phases': phases,
            'avg_confidence': round(avg_confidence, 3),
            'avg_processing_time': round(avg_processing_time, 3),
            'errors': errors,
            'error_rate': round(errors / total_calls, 3) if total_calls > 0 else 0
        }
    
    def analyze_llm_performance(self) -> Dict[str, Any]:
        """Analizza le performance del LLM basandosi sui debug data"""
        if not self.enabled or not self.debug_history:
            return {'analysis_available': False}
        
        # Analisi confidenza per fase
        confidence_by_phase = {}
        predictions_by_phase = {}
        
        for debug_info in self.debug_history:
            phase = debug_info.phase
            
            if phase not in confidence_by_phase:
                confidence_by_phase[phase] = []
                predictions_by_phase[phase] = []
            
            confidence_by_phase[phase].append(debug_info.confidence)
            
            # Estrai predizione
            predicted_label = debug_info.parsed_response.get('predicted_label', 'unknown')
            predictions_by_phase[phase].append(predicted_label)
        
        # Calcola statistiche
        analysis = {
            'analysis_available': True,
            'phases_analyzed': list(confidence_by_phase.keys()),
            'confidence_stats': {},
            'prediction_stats': {}
        }
        
        for phase in confidence_by_phase:
            confidences = confidence_by_phase[phase]
            predictions = predictions_by_phase[phase]
            
            analysis['confidence_stats'][phase] = {
                'count': len(confidences),
                'avg': round(sum(confidences) / len(confidences), 3),
                'min': round(min(confidences), 3),
                'max': round(max(confidences), 3),
                'below_05': sum(1 for c in confidences if c < 0.5),
                'below_03': sum(1 for c in confidences if c < 0.3)
            }
            
            # Conta predizioni uniche
            unique_predictions = {}
            for pred in predictions:
                unique_predictions[pred] = unique_predictions.get(pred, 0) + 1
            
            analysis['prediction_stats'][phase] = {
                'unique_labels': len(unique_predictions),
                'label_distribution': unique_predictions,
                'most_frequent': max(unique_predictions.items(), key=lambda x: x[1]) if unique_predictions else None
            }
        
        return analysis

    def export_debug_data(self, output_file: str = None) -> str:
        """Esporta tutti i dati di debug in un file JSON"""
        if not self.enabled:
            return None
        
        if output_file is None:
            output_file = f"llm_debug_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(self.debug_history),
            'debug_config': self.llm_debug_config,
            'summary': self.get_debug_summary(),
            'performance_analysis': self.analyze_llm_performance(),
            'debug_records': [
                {
                    'timestamp': debug.timestamp,
                    'session_id': debug.session_id,
                    'phase': debug.phase,
                    'input_text': debug.input_text,
                    'prompt': debug.prompt,
                    'model_params': debug.model_params,
                    'raw_response': debug.raw_response,
                    'parsed_response': debug.parsed_response,
                    'confidence': debug.confidence,
                    'reasoning': debug.reasoning,
                    'processing_time': debug.processing_time,
                    'error': debug.error,
                    'context': debug.context
                }
                for debug in self.debug_history
            ]
        }
        
        # Converte i tipi NumPy prima dell'export
        export_data_safe = convert_numpy_types(export_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data_safe, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Debug data esportati in: {output_file}")
        return output_file
