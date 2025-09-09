#!/usr/bin/env python3
"""
Script di verifica per controllare che il tracing sia stato aggiunto correttamente

Autore: Valerio Bignardi  
Data: 2025-09-09
"""

import re
import os
from typing import Dict, List, Tuple

class TracingVerifier:
    """Verifica che il tracing sia stato aggiunto correttamente"""
    
    def __init__(self):
        self.classification_files = [
            '/home/ubuntu/classificatore/Classification/intelligent_classifier.py',
            '/home/ubuntu/classificatore/Classification/advanced_ensemble_classifier.py',
            '/home/ubuntu/classificatore/Clustering/intelligent_intent_clusterer.py'
        ]
        
    def analyze_tracing_coverage(self, file_path: str) -> Dict[str, any]:
        """
        Analizza la copertura del tracing in un file
        
        Returns:
            Dict con statistiche del tracing
        """
        
        if not os.path.exists(file_path):
            return {'error': f'File non trovato: {file_path}'}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Trova tutte le funzioni
        function_pattern = r'def\s+(\w+)\s*\('
        functions = re.findall(function_pattern, content)
        
        # Classifica le funzioni per tipo
        classification_functions = []
        private_functions = []
        public_functions = []
        
        for func in functions:
            if any(keyword in func.lower() for keyword in 
                   ['classify', 'predict', 'cluster', 'parse', 'validate', 'resolve', 'evaluate', 'match']):
                classification_functions.append(func)
            elif func.startswith('_'):
                private_functions.append(func)
            else:
                public_functions.append(func)
        
        # Conta i trace_all
        trace_calls = len(re.findall(r'trace_all\(', content))
        enter_traces = len(re.findall(r'trace_all\([^)]*"ENTER"', content))
        exit_traces = len(re.findall(r'trace_all\([^)]*"EXIT"', content))
        error_traces = len(re.findall(r'trace_all\([^)]*"ERROR"', content))
        
        # Controlla la completezza per ogni funzione di classificazione
        functions_with_tracing = {}
        for func in classification_functions:
            func_pattern = rf'def\s+{re.escape(func)}\s*\([^)]*\):'
            match = re.search(func_pattern, content)
            if match:
                # Trova il contenuto della funzione (approssimativo)
                start = match.end()
                # Cerca la prossima definizione di funzione o la fine del file
                next_func_match = re.search(r'\ndef\s+\w+\s*\(', content[start:])
                if next_func_match:
                    end = start + next_func_match.start()
                else:
                    end = len(content)
                
                func_content = content[start:end]
                
                has_enter = 'trace_all(' in func_content and '"ENTER"' in func_content
                has_exit = 'trace_all(' in func_content and '"EXIT"' in func_content
                has_error = 'trace_all(' in func_content and '"ERROR"' in func_content
                
                functions_with_tracing[func] = {
                    'has_enter': has_enter,
                    'has_exit': has_exit, 
                    'has_error': has_error,
                    'complete': has_enter and has_exit and has_error
                }
        
        return {
            'file': os.path.basename(file_path),
            'total_functions': len(functions),
            'classification_functions': len(classification_functions),
            'private_functions': len(private_functions),
            'public_functions': len(public_functions),
            'trace_calls': trace_calls,
            'enter_traces': enter_traces,
            'exit_traces': exit_traces,
            'error_traces': error_traces,
            'functions_with_tracing': functions_with_tracing,
            'classification_function_names': classification_functions
        }
    
    def generate_report(self) -> str:
        """Genera un report completo del tracing"""
        
        report = []
        report.append("📊 REPORT VERIFICA TRACING CLASSIFICAZIONE")
        report.append("=" * 50)
        report.append("")
        
        total_classification_functions = 0
        total_complete_tracing = 0
        total_trace_calls = 0
        
        for file_path in self.classification_files:
            analysis = self.analyze_tracing_coverage(file_path)
            
            if 'error' in analysis:
                report.append(f"❌ {analysis['error']}")
                continue
            
            report.append(f"📄 {analysis['file']}")
            report.append("-" * 30)
            report.append(f"  Funzioni totali: {analysis['total_functions']}")
            report.append(f"  Funzioni di classificazione: {analysis['classification_functions']}")
            report.append(f"  Chiamate trace_all: {analysis['trace_calls']}")
            report.append(f"  ENTER traces: {analysis['enter_traces']}")
            report.append(f"  EXIT traces: {analysis['exit_traces']}")
            report.append(f"  ERROR traces: {analysis['error_traces']}")
            report.append("")
            
            # Analisi dettagliata funzioni di classificazione
            if analysis['functions_with_tracing']:
                report.append("  🎯 Funzioni di Classificazione:")
                
                complete_count = 0
                for func_name, tracing_info in analysis['functions_with_tracing'].items():
                    status_icons = []
                    if tracing_info['has_enter']:
                        status_icons.append("📥")
                    if tracing_info['has_exit']:
                        status_icons.append("📤")
                    if tracing_info['has_error']:
                        status_icons.append("🚨")
                    
                    if tracing_info['complete']:
                        complete_count += 1
                        report.append(f"    ✅ {func_name} {''.join(status_icons)}")
                    else:
                        missing = []
                        if not tracing_info['has_enter']:
                            missing.append("ENTER")
                        if not tracing_info['has_exit']:
                            missing.append("EXIT")
                        if not tracing_info['has_error']:
                            missing.append("ERROR")
                        
                        report.append(f"    ⚠️  {func_name} - Manca: {', '.join(missing)}")
                
                completion_rate = (complete_count / len(analysis['functions_with_tracing'])) * 100
                report.append(f"  📈 Copertura: {complete_count}/{len(analysis['functions_with_tracing'])} ({completion_rate:.1f}%)")
                
                total_classification_functions += analysis['classification_functions']
                total_complete_tracing += complete_count
                total_trace_calls += analysis['trace_calls']
            
            report.append("")
        
        # Summary finale
        report.append("🏆 SUMMARY GLOBALE")
        report.append("=" * 30)
        report.append(f"📊 Funzioni di classificazione totali: {total_classification_functions}")
        report.append(f"✅ Funzioni con tracing completo: {total_complete_tracing}")
        report.append(f"🔍 Chiamate trace_all totali: {total_trace_calls}")
        
        if total_classification_functions > 0:
            global_completion_rate = (total_complete_tracing / total_classification_functions) * 100
            report.append(f"📈 Copertura globale: {global_completion_rate:.1f}%")
            
            if global_completion_rate >= 95:
                report.append("🎉 ECCELLENTE! Tracing quasi completo!")
            elif global_completion_rate >= 80:
                report.append("👍 BUONO! La maggior parte delle funzioni ha tracing")
            elif global_completion_rate >= 60:
                report.append("⚠️ DISCRETO. Alcune funzioni necessitano ancora tracing")
            else:
                report.append("❌ INSUFFICIENTE. Molte funzioni mancano di tracing")
        
        return '\n'.join(report)
    
    def run(self):
        """Esegue la verifica e stampa il report"""
        report = self.generate_report()
        print(report)
        
        # Salva anche su file
        report_file = '/home/ubuntu/classificatore/tracing_verification_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Report salvato in: {report_file}")

def main():
    """Funzione main"""
    verifier = TracingVerifier()
    verifier.run()

if __name__ == "__main__":
    main()
