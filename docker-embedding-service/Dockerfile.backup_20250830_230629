# LaBSE Embedding Service - Versione Corretta con Best Practices
# Autore: Valerio Bignardi  
# Data: 2025-08-29
FROM python:3.11-slim

# Metadata
LABEL maintainer="Valerio Bignardi"
LABEL description="LaBSE Embedding Service - Implementazione corretta con volume cache"
LABEL version="2.0.0"

# Variabili d'ambiente per HuggingFace
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_NAME="sentence-transformers/LaBSE"
ENV MAX_BATCH_SIZE=32
ENV MAX_SEQUENCE_LENGTH=512
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HUB_CACHE=/cache

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Directory di lavoro
WORKDIR /app

# Installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia applicazione
COPY app/ .

# SOLUZIONE CHIAVE: Usa volume per cache invece di directory locale
# La cache sarà gestita come volume Docker esterno
VOLUME ["/cache"]

# Espone porta
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando di avvio
CMD ["python", "main.py"]

# Espone porta
EXPOSE 8080

# Health check ottimizzato
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Comando di avvio
CMD ["python", "main.py"]
