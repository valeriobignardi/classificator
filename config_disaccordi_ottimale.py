"""
Configurazione ottimizzata per gestione disaccordi ML vs LLM
Basata su analisi performance del 28 Agosto 2025
Autore: Valerio Bignardi
"""

# STRATEGIA OTTIMALE PER DISACCORDI ML vs LLM

# 1. PESI ADATTIVI PER CATEGORIA
ADAPTIVE_WEIGHTS = {
    'assistenza_tecnica': {'llm': 0.45, 'ml': 0.55},  # ML è migliore
    'supporto': {'llm': 0.45, 'ml': 0.55},           # ML è migliore  
    'informazioni_prodotto': {'llm': 0.50, 'ml': 0.50}, # Equivalenti
    'vendite': {'llm': 0.50, 'ml': 0.50},            # Equivalenti
    'altro': {'llm': 0.75, 'ml': 0.25},              # LLM è migliore
    'reclami': {'llm': 0.70, 'ml': 0.30},            # LLM è migliore
    'default': {'llm': 0.51, 'ml': 0.49}             # Pesi bilanciati
}

# 2. SOGLIE DI CONFIDENCE
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 0.85,    # Classificazione automatica sicura
    'medium_confidence': 0.70,  # Usa ensemble con penalità
    'low_confidence': 0.50,     # Etichetta come "ALTRO" o "INCERTO"
}

# 3. STRATEGIA COMPLETA
DISAGREEMENT_STRATEGY = {
    'step_1': 'Calcola confidence massima tra ML e LLM',
    'step_2': 'Se max_confidence < 0.7 → Tag "ALTRO"',
    'step_3': 'Altrimenti usa pesi adattivi per categoria',
    'step_4': 'Applica penalità 0.8 per disaccordo',
    'step_5': 'Registra caso per analisi future'
}

# 4. CODICE DI IMPLEMENTAZIONE
IMPLEMENTATION_LOGIC = '''
def resolve_disagreement(llm_pred, ml_pred, category=None):
    max_confidence = max(llm_pred['confidence'], ml_pred['confidence'])
    
    # STEP 1: Confidence troppo bassa → ALTRO
    if max_confidence < 0.70:
        return {
            'predicted_label': 'ALTRO',
            'confidence': max_confidence * 0.6,  # Penalità maggiore
            'method': 'LOW_CONFIDENCE_ALTRO',
            'reason': 'Disaccordo + confidence bassa'
        }
    
    # STEP 2: Usa pesi adattivi per categoria
    weights = ADAPTIVE_WEIGHTS.get(category, ADAPTIVE_WEIGHTS['default'])
    
    # STEP 3: Calcola pesi aggiustati
    adjusted_llm = weights['llm'] * llm_pred['confidence']
    adjusted_ml = weights['ml'] * ml_pred['confidence']
    
    # STEP 4: Scegli vincitore e applica penalità
    if adjusted_llm > adjusted_ml:
        return {
            'predicted_label': llm_pred['predicted_label'],
            'confidence': llm_pred['confidence'] * 0.8,  # Penalità disaccordo
            'method': 'LLM_DISAGREEMENT_ADAPTIVE',
            'reason': f'LLM vince con pesi adattivi ({weights})'
        }
    else:
        return {
            'predicted_label': ml_pred['predicted_label'],
            'confidence': ml_pred['confidence'] * 0.8,  # Penalità disaccordo
            'method': 'ML_DISAGREEMENT_ADAPTIVE',
            'reason': f'ML vince con pesi adattivi ({weights})'
        }
'''
