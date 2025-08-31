# Dockerfile per LaBSE Embedding Service con GPU Support
# Autore: Valerio Bignardi
# Data: 2025-08-31
# Basato su documentazione ufficiale PyTorch e NVIDIA CUDA

# Utilizzo immagine base Python ufficiale (più stabile e leggera)
FROM python:3.11-slim AS builder

# Installa dipendenze sistema necessarie
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Installa PyTorch con supporto CUDA 12.8 (comando ufficiale PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt

# Fase 2: Immagine finale runtime
FROM python:3.11-slim

WORKDIR /app

# Installa curl per health check
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copia l'ambiente Python dalla fase di build
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia l'applicazione
COPY app/ .

# Variabili d'ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/cache
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HUB_CACHE=/cache
VOLUME /cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "main.py"]
