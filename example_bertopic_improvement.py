"""
Esempio concreto: Come BERTopic migliora i modelli ML
Author: AI Assistant
Date: 2025-08-21
"""

# ESEMPIO CONCRETO - CLASSIFICAZIONE "prenotazione_esami"

# === CASO 1: SOLO EMBEDDINGS ===
input_text = "vorrei prenotare una risonanza magnetica per domani"
embeddings_only = [0.23, -0.45, 0.12, ..., 0.67]  # 768 dim

# Il modello ML deve imparare pattern complessi:
# - Correlazioni tra centinaia di dimensioni
# - Pattern non lineari nascosti negli embeddings
# - Difficile generalizzazione

# === CASO 2: EMBEDDINGS + BERTOPIC FEATURES ===
input_text = "vorrei prenotare una risonanza magnetica per domani"

# BERTopic scopre topic semantici:
discovered_topics = {
    "Topic_0": ["prenotazione", "appuntamento", "riservare"],     # PRENOTAZIONI
    "Topic_1": ["risonanza", "tac", "ecografia", "esame"],       # ESAMI MEDICI  
    "Topic_2": ["domani", "urgente", "presto", "disponibilità"], # TEMPISTICA
    "Topic_3": ["dottore", "medico", "specialista"],             # PERSONALE
    "Topic_4": ["costo", "prezzo", "quanto", "pagamento"]        # FATTURAZIONE
}

# BERTopic calcola probabilità per ogni topic:
topic_probabilities = [
    0.85,  # Topic_0 (PRENOTAZIONI) - ALTA probabilità ✅
    0.78,  # Topic_1 (ESAMI MEDICI) - ALTA probabilità ✅  
    0.62,  # Topic_2 (TEMPISTICA) - MEDIA probabilità
    0.15,  # Topic_3 (PERSONALE) - BASSA probabilità
    0.08   # Topic_4 (FATTURAZIONE) - BASSA probabilità
]

# Features finali per ML:
final_features = [
    # Embeddings originali (768 dim)
    0.23, -0.45, 0.12, ..., 0.67,
    # Topic probabilities (5 dim) - NUOVO!
    0.85, 0.78, 0.62, 0.15, 0.08,
    # One-hot top topic (5 dim) - NUOVO!  
    1, 0, 0, 0, 0  # Top topic = Topic_0 (PRENOTAZIONI)
]

# === VANTAGGI PER MODELLI ML ===

# 1. INTERPRETABILITÀ
# Il modello ML può imparare regole come:
# IF topic_prenotazioni > 0.7 AND topic_esami > 0.6 
#    THEN classe = "prenotazione_esami"

# 2. GENERALIZZAZIONE
# Pattern più robusti: anche se cambiano parole specifiche,
# i topic rimangono stabili

# 3. RIDUZIONE DIMENSIONALITÀ SEMANTICA  
# Invece di 768 dimensioni complesse, ha 5-20 topic interpretabili

# === ESEMPIO PRATICO DI MIGLIORAMENTO ===

def compare_performance():
    """Simulazione dell'effetto BERTopic sui modelli ML"""
    
    # Esempio testi simili ma con parole diverse
    texts = [
        "vorrei prenotare una risonanza magnetica",
        "devo fissare un appuntamento per la RM", 
        "posso riservare una slot per risonanza?"
    ]
    
    # SOLO EMBEDDINGS - parole diverse = embeddings molto diversi
    embeddings = [
        [0.2, -0.4, 0.1, ...],  # "prenotare" "risonanza"  
        [0.8, -0.1, 0.7, ...],  # "fissare" "RM"
        [0.1,  0.5, -0.3, ...]  # "riservare" "slot"
    ]
    # ❌ ML tradizionale fatica a capire che sono simili
    
    # CON BERTOPIC - stesso pattern semantico
    topic_probs = [
        [0.85, 0.78, 0.15, 0.10, 0.05],  # Tutti e 3 i testi
        [0.87, 0.76, 0.18, 0.12, 0.07],  # hanno ALTA prob per
        [0.83, 0.79, 0.14, 0.09, 0.06]   # Topic_0 + Topic_1!
    ]
    # ✅ ML tradizionale vede chiaramente il pattern comune!
    
    return "BERTopic rende espliciti i pattern semantici nascosti"

# === METRICHE DI MIGLIORAMENTO ATTESE ===
expected_improvements = {
    "accuracy": "+5-15%",           # Migliore classificazione
    "precision": "+10-20%",         # Meno falsi positivi  
    "recall": "+8-18%",             # Meno falsi negativi
    "generalization": "+20-30%",    # Su testi nuovi/diversi
    "interpretability": "+300%"      # Comprensione decisioni
}

print("BERTopic trasforma features complesse in pattern interpretabili!")
