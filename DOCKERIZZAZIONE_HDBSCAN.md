# PIANO DETTAGLIATO: DOCKERIZZAZIONE SERVIZIO HDBSCAN+UMAP

**Autore:** Valerio Bignardi  
**Data:** 2025-01-27  
**Obiettivo:** Staccare HDBSCAN e UMAP dal codice principale e creare un servizio Docker dedicato

---

## 📋 EXECUTIVE SUMMARY

Il piano prevede l'estrazione delle componenti HDBSCAN e UMAP dal sistema attuale per creare un **servizio microservizi dockerizzato** che offra API REST per:
- Clustering con HDBSCAN (CPU/GPU)
- Riduzione dimensionale con UMAP
- Predizione di nuovi punti
- Gestione stato modelli

### 🎯 BENEFICI ATTESI:
- **Scalabilità:** Servizio dedicato scalabile independente
- **Isolamento:** Separazione dipendenze pesanti (cuML, UMAP)
- **Performance:** Ottimizzazione specifica GPU/CPU
- **Maintenance:** Deployment e aggiornamenti indipendenti
- **Resource Management:** Controllo risorse GPU dedicato

---

## 🔍 FASE 1: ANALISI DELL'ESISTENTE

### 📊 COMPONENTI DA ESTRARRE:

#### 1.1 **File Principali:**
```
📁 /Clustering/
├── hdbscan_clusterer.py          # 🎯 CORE - HDBSCANClusterer class
├── hdbscan_tuning_guide.py       # 🔧 Tuning parameters
├── clustering_test_service.py    # 🧪 Testing utilities
└── clustering_test_service_new.py # 🧪 Extended testing
```

#### 1.2 **Dipendenze Critiche:**
```python
# Core clustering
import hdbscan                     # HDBSCAN algorithm
import umap.umap_ as umap         # UMAP dimensionality reduction
import cuml                       # GPU clustering (RAPIDS cuML)
import cupy as cp                 # GPU arrays
from cuml.cluster import HDBSCAN as cumlHDBSCAN

# Scientific computing
import numpy as np
import pandas as pd
import scipy.sparse

# Machine learning
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Utilities
import time
import pickle
import joblib
```

#### 1.3 **Punti di Integrazione Attuali:**
```python
# Pipeline usage
from hdbscan_clusterer import HDBSCANClusterer

# In Pipeline/end_to_end_pipeline.py line 33
# In Clustering/intelligent_intent_clusterer.py line 24
# In TopicModeling/bertopic_feature_provider.py line 226
```

---

## 🏗️ FASE 2: ARCHITETTURA DEL SERVIZIO

### 2.1 **Struttura Proposta:**
```
📁 docker-clustering-service/
├── 📄 Dockerfile
├── 📄 docker-compose.yml
├── 📄 requirements.txt
├── 📁 app/
│   ├── 📄 main.py                 # FastAPI app entry point
│   ├── 📄 clustering_service.py   # Core clustering logic
│   ├── 📄 umap_service.py         # UMAP dimensionality reduction
│   ├── 📄 models.py               # Pydantic models
│   ├── 📄 config.py               # Configuration management
│   └── 📁 utils/
│       ├── 📄 gpu_utils.py        # GPU detection & management
│       ├── 📄 validation.py       # Input validation
│       └── 📄 cache.py            # Model state caching
├── 📁 tests/
│   ├── 📄 test_clustering.py
│   ├── 📄 test_umap.py
│   └── 📄 test_api.py
└── 📁 scripts/
    ├── 📄 health_check.py
    └── 📄 benchmark.py
```

### 2.2 **API Endpoints Design:**
```python
# Clustering Operations
POST /api/v1/cluster/fit          # Train HDBSCAN model
POST /api/v1/cluster/predict      # Predict new points
POST /api/v1/cluster/fit_predict  # Train and predict in one call

# UMAP Operations  
POST /api/v1/umap/fit             # Train UMAP reducer
POST /api/v1/umap/transform       # Transform new data
POST /api/v1/umap/fit_transform   # Fit and transform

# Model Management
GET  /api/v1/models/status        # Model state info
POST /api/v1/models/save          # Save model state
POST /api/v1/models/load          # Load model state
DELETE /api/v1/models/clear       # Clear model cache

# Health & Monitoring
GET  /api/v1/health               # Service health
GET  /api/v1/metrics              # Performance metrics
GET  /api/v1/gpu/status           # GPU availability
```

---

## 🔧 FASE 3: IMPLEMENTAZIONE DETTAGLIATA

### 3.1 **Core Clustering Service (`app/clustering_service.py`):**
```python
"""
Servizio clustering basato su HDBSCANClusterer esistente
Estrae e adatta la logica da /Clustering/hdbscan_clusterer.py
"""

class ClusteringService:
    def __init__(self):
        self.clusterer = None
        self.model_state = {}
        self.gpu_available = self._check_gpu()
    
    # Metodi estratti da HDBSCANClusterer:
    async def fit_predict(self, embeddings, params)
    async def predict_new_points(self, new_embeddings)
    async def save_model_state(self, model_id)
    async def load_model_state(self, model_id)
    
    # GPU management estratto dal codice esistente
    def _check_gpu_memory(self, embeddings_size_mb)
    def _should_use_gpu(self, embeddings_size_mb)
```

### 3.2 **UMAP Service (`app/umap_service.py`):**
```python
"""
Servizio UMAP basato su _apply_umap_reduction() esistente
Estrae logica da HDBSCANClusterer._apply_umap_reduction()
"""

class UMAPService:
    def __init__(self):
        self.umap_reducer = None
        self.reduction_cache = {}
    
    # Metodi estratti da _apply_umap_reduction():
    async def fit_transform(self, embeddings, params)
    async def transform(self, embeddings)
    async def get_debug_info(self)
```

### 3.3 **FastAPI Application (`app/main.py`):**
```python
"""
FastAPI application con routing e middleware
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="HDBSCAN+UMAP Clustering Service")

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Services
clustering_service = ClusteringService()
umap_service = UMAPService()

# Endpoints implementation...
@app.post("/api/v1/cluster/fit_predict")
async def cluster_fit_predict(request: ClusterRequest):
    # Implementation extracted from HDBSCANClusterer.fit_predict()
    pass
```

### 3.4 **Pydantic Models (`app/models.py`):**
```python
"""
Data models per API requests/responses
Basati sui parametri attuali di HDBSCANClusterer
"""

class ClusterRequest(BaseModel):
    embeddings: List[List[float]]
    min_cluster_size: int = 5
    min_samples: int = 1
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0
    
    # UMAP parameters
    use_umap: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 50
    umap_metric: str = "euclidean"
    
    # GPU parameters
    gpu_enabled: bool = True
    gpu_memory_limit: float = 0.8

class ClusterResponse(BaseModel):
    cluster_labels: List[int]
    outlier_scores: List[float]
    cluster_probabilities: List[float]
    execution_info: Dict
    performance_metrics: Dict
```

---

## 🐳 FASE 4: DOCKERIZZAZIONE

### 4.1 **Dockerfile Ottimizzato:**
```dockerfile
# Multi-stage build per ottimizzazione dimensioni
FROM nvidia/cuda:12.8-runtime-ubuntu22.04 AS base

# GPU runtime per RAPIDS cuML
FROM base AS gpu-builder
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# CPU-only version
FROM python:3.11-slim AS cpu-builder  
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Final stage - runtime
FROM base AS runtime
WORKDIR /app

# Copy dependencies from appropriate builder
ARG GPU_SUPPORT=true
COPY --from=${GPU_SUPPORT:+gpu-}${GPU_SUPPORT:-cpu-}builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application
COPY app/ .

# Configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CLUSTERING_SERVICE_PORT=8081
ENV CLUSTERING_SERVICE_WORKERS=1

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python health_check.py || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
```

### 4.2 **Docker Compose Configuration:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  clustering-service:
    build: 
      context: .
      args:
        GPU_SUPPORT: true
    ports:
      - "8081:8081"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - GPU_MEMORY_FRACTION=0.8
      - MODEL_CACHE_SIZE=1000MB
    volumes:
      - clustering_models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "health_check.py"]
      interval: 30s
      timeout: 10s
      retries: 3

  clustering-cpu:
    build:
      context: .
      args:
        GPU_SUPPORT: false
    ports:
      - "8082:8081"
    environment:
      - FORCE_CPU_ONLY=true
    volumes:
      - clustering_models:/app/models
    profiles: ["cpu-only"]

volumes:
  clustering_models:
```

### 4.3 **Requirements Split:**
```text
# requirements-gpu.txt
torch>=2.0.0
cupy-cuda12x>=12.0.0
cuml>=23.08.0
rapids-cudf>=23.08.0
hdbscan>=0.8.29
umap-learn>=0.5.5
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
numpy>=1.21.0
scikit-learn>=1.0.0
pydantic>=2.0.0

# requirements-cpu.txt  
torch>=2.0.0
hdbscan>=0.8.29
umap-learn>=0.5.5
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
numpy>=1.21.0
scikit-learn>=1.0.0
pydantic>=2.0.0
```

---

## 🔌 FASE 5: INTEGRAZIONE CON SISTEMA ESISTENTE

### 5.1 **Client Adapter (`Clustering/hdbscan_client.py`):**
```python
"""
Adapter per sostituire HDBSCANClusterer con chiamate HTTP
Mantiene la stessa interfaccia per backward compatibility
"""

import requests
import numpy as np
from typing import Tuple, Dict, Any

class HDBSCANClustererClient:
    """
    Drop-in replacement per HDBSCANClusterer che usa HTTP API
    """
    
    def __init__(self, service_url: str = "http://localhost:8081", **kwargs):
        self.service_url = service_url
        self.params = kwargs
        self.cluster_labels = None
        self.outlier_scores = None
        
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Mantiene la stessa signature di HDBSCANClusterer.fit_predict()
        """
        response = requests.post(
            f"{self.service_url}/api/v1/cluster/fit_predict",
            json={
                "embeddings": embeddings.tolist(),
                **self.params
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            self.cluster_labels = np.array(result["cluster_labels"])
            self.outlier_scores = np.array(result["outlier_scores"])
            return self.cluster_labels
        else:
            raise Exception(f"Clustering service error: {response.text}")
    
    def predict_new_points(self, new_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mantiene la stessa signature per predizioni incrementali
        """
        # Implementation...
        pass
```

### 5.2 **Migration Strategy:**

#### **Step 1: Backward Compatibility**
```python
# In existing files, add feature flag
USE_CLUSTERING_SERVICE = os.getenv("USE_CLUSTERING_SERVICE", "false").lower() == "true"

if USE_CLUSTERING_SERVICE:
    from Clustering.hdbscan_client import HDBSCANClustererClient as HDBSCANClusterer
else:
    from Clustering.hdbscan_clusterer import HDBSCANClusterer
```

#### **Step 2: Gradual Migration**
```python
# Environment-based switching
CLUSTERING_SERVICE_URL = os.getenv("CLUSTERING_SERVICE_URL", "http://localhost:8081")

# Factory pattern for clusterer creation
def create_clusterer(**kwargs):
    if USE_CLUSTERING_SERVICE:
        return HDBSCANClustererClient(CLUSTERING_SERVICE_URL, **kwargs)
    else:
        return HDBSCANClusterer(**kwargs)
```

---

## 📊 FASE 6: TESTING E VALIDAZIONE

### 6.1 **Unit Tests (`tests/`):**
```python
# tests/test_clustering.py
async def test_clustering_fit_predict():
    """Test clustering functionality"""
    
# tests/test_umap.py  
async def test_umap_reduction():
    """Test UMAP dimensionality reduction"""
    
# tests/test_api.py
async def test_api_endpoints():
    """Test FastAPI endpoints"""

# tests/test_compatibility.py
def test_backward_compatibility():
    """Ensure client matches original interface"""
```

### 6.2 **Performance Benchmarks:**
```python
# scripts/benchmark.py
def benchmark_clustering_performance():
    """
    Confronta performance:
    - Original HDBSCANClusterer vs Service HTTP
    - GPU vs CPU performance  
    - Memory usage comparison
    - Latency measurements
    """
```

### 6.3 **Integration Tests:**
```python
# tests/test_integration.py
def test_pipeline_integration():
    """
    Test completo su Pipeline esistente:
    - end_to_end_pipeline.py con nuovo servizio
    - Risultati identici a implementazione originale
    - Performance acceptable
    """
```

---

## 🚀 FASE 7: DEPLOYMENT E MONITORING

### 7.1 **Health Monitoring:**
```python
# scripts/health_check.py
def check_service_health():
    """
    - GPU availability
    - Memory usage
    - Model cache status
    - Response times
    """

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

clustering_requests = Counter('clustering_requests_total')
clustering_duration = Histogram('clustering_duration_seconds')
gpu_memory_usage = Gauge('gpu_memory_usage_bytes')
```

### 7.2 **Logging & Observability:**
```python
# Structured logging
import structlog

logger = structlog.get_logger()

# Request tracing
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info("request_processed",
        method=request.method,
        url=str(request.url),
        duration=duration,
        status_code=response.status_code
    )
    return response
```

### 7.3 **Production Deployment:**
```bash
# Production deployment script
#!/bin/bash
# deploy.sh

# Build images
docker build -t clustering-service:latest .

# Deploy with GPU support
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl -f http://localhost:8081/api/v1/health

# Load balancer configuration (nginx/traefik)
# Auto-scaling based on GPU memory usage
```

---

## 📈 FASE 8: OTTIMIZZAZIONI AVANZATE

### 8.1 **Caching Strategy:**
```python
# Redis-based model caching
import redis
import pickle

class ModelCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379)
    
    def cache_model(self, model_id: str, model_state: dict):
        """Cache trained models for reuse"""
        
    def get_cached_model(self, model_id: str):
        """Retrieve cached model if available"""
```

### 8.2 **Batch Processing:**
```python
# Batch request optimization
@app.post("/api/v1/cluster/batch")
async def cluster_batch(requests: List[ClusterRequest]):
    """Process multiple clustering requests efficiently"""
```

### 8.3 **Auto-scaling Configuration:**
```yaml
# kubernetes/hpa.yaml (if using K8s)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: clustering-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: clustering-service
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 🎯 FASE 9: MIGRATION PLAN & TIMELINE

### 9.1 **Milestone Timeline:**

| **Week** | **Milestone** | **Deliverables** |
|----------|---------------|------------------|
| W1 | Analysis & Design | Architecture document, API specs |
| W2 | Core Service Development | FastAPI app, clustering logic |
| W3 | UMAP Integration | UMAP service, dimensionality reduction |
| W4 | Dockerization | Dockerfile, docker-compose, build scripts |
| W5 | Client Adapter | HTTP client, backward compatibility |
| W6 | Testing & Validation | Unit tests, integration tests, benchmarks |
| W7 | Documentation | API docs, deployment guides |
| W8 | Production Deployment | Monitoring, health checks, scaling |

### 9.2 **Risk Mitigation:**

| **Risk** | **Impact** | **Mitigation** |
|----------|------------|----------------|
| Performance degradation | HIGH | Benchmarking, caching, optimization |
| GPU compatibility issues | MEDIUM | Fallback to CPU, multi-stage Docker |
| Network latency | MEDIUM | Local deployment, batch processing |
| Memory leaks | HIGH | Comprehensive testing, monitoring |
| Backward compatibility | HIGH | Extensive integration testing |

### 9.3 **Rollback Strategy:**
```python
# Feature flag for instant rollback
ENABLE_CLUSTERING_SERVICE = os.getenv("ENABLE_CLUSTERING_SERVICE", "false")

if ENABLE_CLUSTERING_SERVICE == "true":
    # Use HTTP service
    clusterer = HDBSCANClustererClient()
else:
    # Fallback to original implementation
    clusterer = HDBSCANClusterer()
```

---

## 📚 FASE 10: DOCUMENTAZIONE E MAINTENANCE

### 10.1 **API Documentation:**
- OpenAPI/Swagger specs automatiche
- Esempi di utilizzo per ogni endpoint
- Performance characteristics e limiti
- Troubleshooting guides

### 10.2 **Operations Runbook:**
- Deployment procedures
- Monitoring e alerting setup
- Common issues e soluzioni
- Scaling guidelines
- Backup e recovery procedures

### 10.3 **Migration Guide:**
```markdown
# Migration Guide: HDBSCANClusterer → Clustering Service

## Prerequisites
- Docker & docker-compose installed
- GPU drivers (if using GPU)
- Network connectivity to service

## Step-by-step Migration
1. Deploy clustering service
2. Set environment variables
3. Test compatibility
4. Monitor performance
5. Full migration
```

---

## ✅ CONCLUSIONI E NEXT STEPS

### 🎯 **Benefici Attesi:**
1. **Scalabilità**: Servizio dedicato scalabile independente
2. **Isolamento**: Separazione dipendenze pesanti (cuML, UMAP, CUDA)
3. **Performance**: Ottimizzazione GPU/CPU specifica
4. **Maintenance**: Deployment e updates indipendenti
5. **Resource Control**: Gestione dedicata risorse GPU

### 🚀 **Immediate Actions:**
1. **Approvazione architettura** da stakeholders
2. **Setup development environment** per servizio
3. **Extraction primo componente** (HDBSCANClusterer core)
4. **API design validation** con team

### 📊 **Success Metrics:**
- Performance parity con implementazione originale
- <100ms overhead latency per chiamate HTTP
- Successful backward compatibility
- Zero downtime deployment capability
- GPU memory optimization >20%

### 🔮 **Future Enhancements:**
- Support per clustering algorithms aggiuntivi (K-means, DBSCAN)
- Real-time streaming clustering
- Advanced model versioning
- Multi-tenant isolation
- GraphQL API support

---

**Status:** ✅ Ready for Implementation  
**Next:** Team review e approval per iniziare sviluppo
