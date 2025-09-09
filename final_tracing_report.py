#!/usr/bin/env python3
"""
Rapporto finale del completamento tracing per tutte le funzioni di classificazione

Autore: Valerio Bignardi  
Data: 2025-09-09
"""

import re
from typing import Dict, List

def count_tracing_in_file(file_path: str, function_names: List[str]) -> Dict[str, Dict]:
    """Conta il tracing specifico per le funzioni indicate"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return {}
    
    results = {}
    
    for func_name in function_names:
        # Cerca tutte le occorrenze di trace_all per questa funzione
        enter_pattern = rf'trace_all\([^)]*"{re.escape(func_name)}"[^)]*"ENTER"'
        exit_pattern = rf'trace_all\([^)]*"{re.escape(func_name)}"[^)]*"EXIT"'
        error_pattern = rf'trace_all\([^)]*"{re.escape(func_name)}"[^)]*"ERROR"'
        
        enter_count = len(re.findall(enter_pattern, content))
        exit_count = len(re.findall(exit_pattern, content))
        error_count = len(re.findall(error_pattern, content))
        
        if enter_count > 0 or exit_count > 0 or error_count > 0:
            results[func_name] = {
                'enter': enter_count,
                'exit': exit_count, 
                'error': error_count,
                'complete': enter_count > 0 and exit_count > 0 and error_count > 0
            }
    
    return results

def main():
    """Genera rapporto finale"""
    
    print("🎯 RAPPORTO FINALE: TRACING FUNZIONI CLASSIFICAZIONE")
    print("=" * 60)
    print()
    
    # Funzioni principali di classificazione
    key_functions = {
        '/home/ubuntu/classificatore/Classification/intelligent_classifier.py': [
            'classify_with_motivation',
            'classify_batch', 
            'classify_multiple_conversations_optimized',
            '_call_openai_api_structured',
            '_call_ollama_api_structured',
            '_call_llm_api_structured',
            '_parse_llm_response',
            '_fallback_parse_response',
            '_intelligent_semantic_fallback',
            '_validate_llm_classification_with_embedding',
            '_semantic_label_resolution',
            '_resolve_label_semantically',
            '_find_best_match_via_examples',
            '_evaluate_new_category_with_bertopic'
        ],
        '/home/ubuntu/classificatore/Classification/advanced_ensemble_classifier.py': [
            'predict_with_ensemble',
            'predict_with_llm_only', 
            '_combine_predictions'
        ],
        '/home/ubuntu/classificatore/Clustering/intelligent_intent_clusterer.py': [
            'cluster_intelligently'
        ]
    }
    
    total_functions = 0
    total_with_complete_tracing = 0
    
    for file_path, functions in key_functions.items():
        file_name = file_path.split('/')[-1]
        print(f"📄 {file_name}")
        print("-" * 40)
        
        tracing_results = count_tracing_in_file(file_path, functions)
        
        file_complete = 0
        for func_name in functions:
            total_functions += 1
            
            if func_name in tracing_results:
                result = tracing_results[func_name]
                status = "✅" if result['complete'] else "⚠️"
                
                if result['complete']:
                    file_complete += 1
                    total_with_complete_tracing += 1
                
                print(f"  {status} {func_name}")
                print(f"     📥 ENTER: {result['enter']}")
                print(f"     📤 EXIT: {result['exit']}")  
                print(f"     🚨 ERROR: {result['error']}")
            else:
                print(f"  ❌ {func_name} - Nessun tracing trovato")
        
        completion_rate = (file_complete / len(functions)) * 100 if functions else 0
        print(f"\n  📊 Completamento: {file_complete}/{len(functions)} ({completion_rate:.1f}%)")
        print()
    
    # Statistiche globali
    global_completion = (total_with_complete_tracing / total_functions) * 100 if total_functions else 0
    
    print("🏆 STATISTICHE GLOBALI")
    print("=" * 30)
    print(f"📊 Funzioni analizzate: {total_functions}")
    print(f"✅ Con tracing completo: {total_with_complete_tracing}")
    print(f"📈 Completamento: {global_completion:.1f}%")
    print()
    
    # Valutazione finale
    if global_completion >= 95:
        print("🎉 ECCELLENTE! Tracing completo implementato!")
        print("   Tutte le funzioni di classificazione hanno:")
        print("   - 📥 Tracing ENTER (all'inizio)")
        print("   - 📤 Tracing EXIT (ai return)")
        print("   - 🚨 Tracing ERROR (nei catch)")
    elif global_completion >= 80:
        print("👍 OTTIMO! La maggior parte delle funzioni ha tracing completo")
    elif global_completion >= 60:
        print("⚠️ BUONO. Alcune funzioni necessitano ancora work")
    else:
        print("❌ Da completare. Molte funzioni mancano di tracing")
    
    print()
    print("🔍 BENEFICI DEL TRACING IMPLEMENTATO:")
    print("   - Debugging dettagliato della pipeline di classificazione")
    print("   - Monitoring delle performance per ogni funzione")
    print("   - Tracciamento completo del flusso di esecuzione")
    print("   - Gestione avanzata degli errori con contesto")
    print("   - Analisi dei tempi di esecuzione")
    print()
    print("💡 Usa trace_all per monitorare il sistema in produzione!")

if __name__ == "__main__":
    main()
