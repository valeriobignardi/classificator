#!/usr/bin/env python3
"""
Script per analizzare i disaccordi tra ML e LLM
Analizza le performance storiche per ottimizzare i pesi
Autore: Valerio Bignardi
Data: 28 Agosto 2025
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

# Aggiunge il percorso per importare i moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analizza_performance_ensemble(log_file_path: str = None) -> Dict[str, Any]:
    """
    Analizza le performance dell'ensemble ML vs LLM
    
    Args:
        log_file_path: Percorso al file di log delle predizioni
        
    Returns:
        Analisi delle performance e raccomandazioni sui pesi
    """
    
    # Simulazione dati (da sostituire con dati reali dal database)
    print("🔍 Simulazione analisi performance ensemble...")
    
    # Categorie più frequenti (da config.yaml o database)
    categorie_frequenti = [
        "assistenza_tecnica", "informazioni_prodotto", "reclami",
        "vendite", "supporto", "altro"
    ]
    
    # Simulazione performance per categoria
    performance_by_category = {}
    for categoria in categorie_frequenti:
        # Simula performance diverse per categoria
        if categoria in ["assistenza_tecnica", "supporto"]:
            # ML migliore su pattern ricorrenti
            ml_acc = 0.89
            llm_acc = 0.83
        elif categoria in ["altro", "reclami"]:
            # LLM migliore su casi complessi
            ml_acc = 0.75
            llm_acc = 0.88
        else:
            # Performance simili
            ml_acc = 0.82
            llm_acc = 0.84
            
        performance_by_category[categoria] = {
            'ml_accuracy': ml_acc,
            'llm_accuracy': llm_acc,
            'ml_better': ml_acc > llm_acc,
            'difference': abs(ml_acc - llm_acc)
        }
    
    # Calcola pesi ottimali
    total_ml_acc = sum([p['ml_accuracy'] for p in performance_by_category.values()])
    total_llm_acc = sum([p['llm_accuracy'] for p in performance_by_category.values()])
    
    optimal_ml_weight = total_ml_acc / (total_ml_acc + total_llm_acc)
    optimal_llm_weight = total_llm_acc / (total_ml_acc + total_llm_acc)
    
    # Analisi disaccordi
    disaccordi_analysis = {
        'percentage_disagreements': 0.15,  # 15% dei casi
        'avg_confidence_when_disagree': 0.72,
        'most_problematic_categories': ["altro", "informazioni_prodotto"]
    }
    
    # Genera raccomandazioni
    raccomandazioni = []
    
    if optimal_llm_weight > 0.55:
        raccomandazioni.append(
            f"💡 Aumenta peso LLM a {optimal_llm_weight:.2f} (attualmente 0.60)"
        )
    if optimal_ml_weight > 0.45:
        raccomandazioni.append(
            f"💡 Aumenta peso ML a {optimal_ml_weight:.2f} (attualmente 0.40)"
        )
    
    # Strategia per disaccordi
    if disaccordi_analysis['avg_confidence_when_disagree'] < 0.7:
        raccomandazioni.append(
            "⚠️ Quando disaccordo + confidence < 0.7 → Usa tag 'ALTRO'"
        )
    
    raccomandazioni.append(
        "🎯 Implementa pesi adattivi per categoria (ML meglio su pattern ricorrenti)"
    )
    
    return {
        'timestamp': datetime.now().isoformat(),
        'current_weights': {'llm': 0.6, 'ml': 0.4},
        'optimal_weights': {
            'llm': optimal_llm_weight,
            'ml': optimal_ml_weight
        },
        'performance_by_category': performance_by_category,
        'disagreement_analysis': disaccordi_analysis,
        'raccomandazioni': raccomandazioni,
        'next_steps': [
            "1. Implementa pesi adattivi per categoria",
            "2. Usa soglia confidence 0.7 per etichetta ALTRO",
            "3. Monitora performance dopo modifiche",
            "4. Considera active learning per casi dubbi"
        ]
    }

def genera_configurazione_ottimale(analisi: Dict[str, Any]) -> str:
    """
    Genera configurazione ottimale per ensemble
    
    Args:
        analisi: Risultato dell'analisi performance
        
    Returns:
        Codice di configurazione Python
    """
    optimal_weights = analisi['optimal_weights']
    
    config_code = f'''
# Configurazione ottimale ensemble (generata il {analisi['timestamp']})
OPTIMAL_ENSEMBLE_CONFIG = {{
    'weights': {{
        'llm': {optimal_weights['llm']:.3f},
        'ml_ensemble': {optimal_weights['ml']:.3f}
    }},
    'confidence_threshold_for_altro': 0.70,
    'disagreement_penalty': 0.8,  # Penalità per disaccordi
    'adaptive_weights_by_category': {{
        'assistenza_tecnica': {{'llm': 0.45, 'ml': 0.55}},  # ML migliore
        'supporto': {{'llm': 0.45, 'ml': 0.55}},
        'altro': {{'llm': 0.75, 'ml': 0.25}},  # LLM migliore
        'reclami': {{'llm': 0.70, 'ml': 0.30}},
        'default': {{'llm': {optimal_weights['llm']:.3f}, 'ml': {optimal_weights['ml']:.3f}}}
    }}
}}

# Strategia per disaccordi:
# 1. Se confidence_max < 0.7 → Tag "ALTRO" 
# 2. Altrimenti usa pesi adattivi per categoria
# 3. Applica penalità 0.8 alla confidence finale
'''
    
    return config_code

if __name__ == "__main__":
    print("📊 ANALISI PERFORMANCE ENSEMBLE ML vs LLM")
    print("=" * 60)
    
    # Esegui analisi
    analisi = analizza_performance_ensemble()
    
    # Stampa risultati
    print("\n🎯 PESI ATTUALI vs OTTIMALI:")
    print(f"   LLM: {analisi['current_weights']['llm']:.2f} → {analisi['optimal_weights']['llm']:.2f}")
    print(f"   ML:  {analisi['current_weights']['ml']:.2f} → {analisi['optimal_weights']['ml']:.2f}")
    
    print("\n📋 RACCOMANDAZIONI:")
    for i, rec in enumerate(analisi['raccomandazioni'], 1):
        print(f"   {i}. {rec}")
    
    print("\n🔧 PROSSIMI STEP:")
    for step in analisi['next_steps']:
        print(f"   • {step}")
    
    # Genera configurazione
    config_code = genera_configurazione_ottimale(analisi)
    
    print("\n💾 CONFIGURAZIONE GENERATA:")
    print(config_code)
    
    # Salva analisi
    output_file = f"analisi_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analisi, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analisi salvata in: {output_file}")
