# 📊 GUIDA CONFIGURAZIONE PARALLELISMO BATCH PROCESSING

## 🎯 PARAMETRI CHIAVE

### 1. classification_batch_size
- **Cosa controlla**: Conversazioni per chunk
- **Valore attuale**: 32
- **Effetto**: Più alto = meno chunk, più conversazioni per chiamata batch
- **Raccomandazione**: 32-64 per OpenAI (limite API)

### 2. max_parallel_calls  
- **Cosa controlla**: Chiamate simultanee totali
- **Valore attuale**: 200
- **Effetto**: Più alto = più parallelismo (limitato da rate limits OpenAI)
- **Raccomandazione**: 100-300 (dipende dal tier OpenAI)

## 📈 CALCOLO PERFORMANCE

Con configurazione attuale:
- **Batch size**: 32 conversazioni/chunk
- **Max parallel**: 200 chiamate simultanee
- **Per 200 conversazioni**: 7 chunk in parallelo
- **Speedup teorico**: ~7x rispetto a processamento sequenziale

## 🛠️ COME MODIFICARE

### Opzione 1: config.yaml (Globale)
```yaml
pipeline:
  classification_batch_size: 64    # Aumenta chunk size
  
llm:
  openai:
    max_parallel_calls: 300        # Aumenta parallelismo
```

### Opzione 2: Database per tenant specifico
```python
# Imposta parametri per tenant specifico
tenant_config = {
    'pipeline': {
        'classification_batch_size': 48
    },
    'llm': {
        'openai': {
            'max_parallel_calls': 250
        }
    }
}
```

### Opzione 3: Runtime dinamico
```python
# Passa parametri durante classificazione
classifier.classify_multiple_conversations_optimized(
    conversations=conversations,
    batch_size=64,          # Override batch size
    max_concurrent=250      # Override parallelismo
)
```

## 💡 RACCOMANDAZIONI TUNING

### Per dataset piccoli (< 100 conversazioni):
```yaml
classification_batch_size: 20
max_parallel_calls: 100
```

### Per dataset medi (100-500 conversazioni):
```yaml  
classification_batch_size: 32
max_parallel_calls: 200
```

### Per dataset grandi (> 500 conversazioni):
```yaml
classification_batch_size: 64
max_parallel_calls: 300
```

## ⚠️ LIMITI E CONSIDERAZIONI

1. **Rate Limits OpenAI**: Tier 1 = 500 RPM, Tier 2 = 5000 RPM
2. **Memoria**: Più batch simultanei = più memoria utilizzata
3. **Timeout**: Chiamate parallele possono causare timeout
4. **Costi**: Più parallelismo = più velocità ma stesso costo totale

## 🧪 TEST PERFORMANCE

Per testare configurazioni:
```bash
python test_batch_parallel_performance.py
```

Confronta tempi con diverse configurazioni per trovare l'ottimale.
