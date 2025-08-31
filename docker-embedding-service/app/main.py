#!/usr/bin/env python3
"""
LaBSE Embedding Service - Servizio Docker per embedding multilingue
Autore: Valerio Bignardi
Data: 2025-08-29
Descrizione: Servizio REST dedicato per embedding LaBSE con gestione ottimizzata della memoria
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import numpy as np
import time
import gc
import psutil
import os
from contextlib import asynccontextmanager
import logging
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelli supportati
SUPPORTED_MODELS = {
    "labse": "sentence-transformers/LaBSE",
    "labse-en-de": "sentence-transformers/LaBSE"
}

# Configurazione globale
class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
    MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    CUDA_DEVICE = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "5000"))
    
config = Config()

# Variabili globali per il modello
model = None
device = None
model_stats = {
    "requests_count": 0,
    "total_texts_processed": 0,
    "total_processing_time": 0.0,
    "memory_peak": 0,
    "model_loaded_at": None
}

class EmbeddingRequest(BaseModel):
    """Request model per embedding"""
    texts: List[str] = Field(..., description="Lista di testi da processare")
    normalize_embeddings: bool = Field(True, description="Normalizza embeddings")
    batch_size: Optional[int] = Field(None, description="Batch size (max 32)")
    session_ids: Optional[List[str]] = Field(None, description="ID sessioni opzionali")
    
class EmbeddingResponse(BaseModel):
    """Response model per embedding"""
    embeddings: List[List[float]]
    shape: List[int]
    processing_time: float
    texts_count: int
    batch_size_used: int
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Response model per health check"""
    status: str
    model_loaded: bool
    device: str
    memory_usage: Dict[str, float]
    stats: Dict[str, Any]
    uptime: float

def get_memory_usage():
    """Ottieni utilizzo memoria sistema"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "process_rss_mb": memory_info.rss / 1024 / 1024,
        "process_vms_mb": memory_info.vms / 1024 / 1024,
        "system_available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "system_used_percent": psutil.virtual_memory().percent
    }

def get_gpu_memory():
    """Ottieni informazioni memoria GPU"""
    if torch.cuda.is_available():
        try:
            return {
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        except:
            return {"gpu_info": "unavailable"}
    return {"gpu_info": "no_cuda"}

async def load_model():
    """Carica il modello LaBSE con gestione ottimizzata"""
    global model, device
    
    logger.info("🚀 Inizializzazione servizio embedding LaBSE...")
    
    # Verifica disponibilità CUDA dettagliata
    logger.info(f"🔍 Verifica CUDA:")
    logger.info(f"   - CUDA disponibile: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   - CUDA version: {torch.version.cuda}")
        logger.info(f"   - GPU count: {torch.cuda.device_count()}")
        logger.info(f"   - GPU names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        logger.info(f"   - Current GPU: {torch.cuda.current_device()}")
    
    # Auto-detect device con logging dettagliato
    if torch.cuda.is_available():
        device = f"cuda:{config.CUDA_DEVICE}" if config.CUDA_DEVICE.isdigit() else "cuda"
        logger.info(f"🎯 GPU selezionata: {device}")
        logger.info(f"🔧 GPU properties: {torch.cuda.get_device_properties(device)}")
    else:
        device = "cpu"
        logger.info("💻 Fallback su CPU - CUDA non disponibile")
    
    try:
        # Caricamento modello con gestione memoria ottimizzata
        logger.info(f"📥 Caricamento modello: {config.MODEL_NAME}")
        
        if device.startswith("cuda"):
            # Caricamento ottimizzato per GPU
            model = SentenceTransformer(
                config.MODEL_NAME,
                device=device,
                trust_remote_code=True
            )
            
            # Verifica memoria GPU dopo caricamento
            gpu_mem = get_gpu_memory()
            logger.info(f"💾 Memoria GPU utilizzata: {gpu_mem.get('gpu_allocated_mb', 0):.1f}MB")
            
        else:
            # Caricamento su CPU
            model = SentenceTransformer(config.MODEL_NAME, device=device)
        
        # Test del modello
        test_embedding = model.encode(["Test di inizializzazione"], show_progress_bar=False)
        logger.info(f"✅ Modello caricato con successo - Shape embedding: {test_embedding.shape}")
        
        # Aggiorna statistiche
        model_stats["model_loaded_at"] = time.time()
        model_stats["memory_peak"] = max(model_stats["memory_peak"], 
                                       get_memory_usage()["process_rss_mb"])
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore caricamento modello: {e}")
        return False

def cleanup_memory():
    """Pulizia memoria dopo processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestione ciclo di vita applicazione"""
    # Startup
    logger.info("🔄 Avvio servizio embedding...")
    success = await load_model()
    if not success:
        logger.error("❌ Impossibile avviare il servizio")
        raise RuntimeError("Model loading failed")
    
    logger.info("🎉 Servizio embedding pronto!")
    yield
    
    # Shutdown
    logger.info("🛑 Spegnimento servizio embedding...")
    cleanup_memory()

# Inizializza FastAPI
app = FastAPI(
    title="LaBSE Embedding Service",
    description="Servizio REST per embedding multilingue con LaBSE",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model_stats
    
    uptime = time.time() - model_stats.get("model_loaded_at", time.time())
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        memory_usage=get_memory_usage(),
        stats=model_stats,
        uptime=uptime
    )

@app.get("/info")
async def service_info():
    """Informazioni dettagliate sul servizio"""
    gpu_info = get_gpu_memory() if torch.cuda.is_available() else {}
    
    return {
        "service": "LaBSE Embedding Service",
        "version": "1.0.0",
        "model": config.MODEL_NAME,
        "device": str(device),
        "config": {
            "max_batch_size": config.MAX_BATCH_SIZE,
            "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
            "cache_size": config.CACHE_SIZE
        },
        "memory": get_memory_usage(),
        "gpu": gpu_info,
        "stats": model_stats
    }

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest, 
    background_tasks: BackgroundTasks
):
    """
    Genera embeddings per i testi forniti
    
    Args:
        request: Richiesta con testi e parametri
        
    Returns:
        Response con embeddings generati
    """
    global model_stats
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # LIMITE RIMOSSO: Il servizio può gestire qualsiasi quantità di testi
    # suddividendoli automaticamente in batch ottimali
    
    start_time = time.time()
    
    try:
        # Determina batch size ottimale
        batch_size = min(
            request.batch_size or config.MAX_BATCH_SIZE,
            config.MAX_BATCH_SIZE,
            len(request.texts)
        )
        
        logger.info(f"🔄 Processing {len(request.texts)} texts with batch_size={batch_size}")
        
        # Pre-processamento testi
        processed_texts = []
        for text in request.texts:
            if not text or not text.strip():
                processed_texts.append("testo vuoto")
            else:
                # Tronca se troppo lungo
                if len(text) > config.MAX_SEQUENCE_LENGTH:
                    text = text[:config.MAX_SEQUENCE_LENGTH] + "..."
                processed_texts.append(text.strip())
        
        # Genera embeddings con ottimizzazioni device-specific
        if device.startswith("cuda"):
            # Ottimizzazioni GPU
            with torch.amp.autocast('cuda'):
                embeddings = model.encode(
                    processed_texts,
                    normalize_embeddings=request.normalize_embeddings,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
        else:
            # Processing CPU
            embeddings = model.encode(
                processed_texts,
                normalize_embeddings=request.normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        # Converti a lista per JSON serialization
        embeddings_list = embeddings.tolist()
        
        processing_time = time.time() - start_time
        
        # Aggiorna statistiche
        model_stats["requests_count"] += 1
        model_stats["total_texts_processed"] += len(request.texts)
        model_stats["total_processing_time"] += processing_time
        model_stats["memory_peak"] = max(
            model_stats["memory_peak"],
            get_memory_usage()["process_rss_mb"]
        )
        
        logger.info(f"✅ Processed {len(request.texts)} texts in {processing_time:.3f}s")
        
        # Cleanup in background
        background_tasks.add_task(cleanup_memory)
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape),
            processing_time=processing_time,
            texts_count=len(request.texts),
            batch_size_used=batch_size,
            model_info={
                "model_name": config.MODEL_NAME,
                "device": str(device),
                "normalize_embeddings": request.normalize_embeddings
            }
        )
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"💾 GPU Out of Memory: {e}")
        cleanup_memory()
        raise HTTPException(status_code=507, detail="GPU out of memory")
        
    except Exception as e:
        logger.error(f"❌ Error during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/batch_similarity")
async def batch_similarity(
    texts1: List[str],
    texts2: List[str],
    normalize: bool = True
):
    """Calcola similarità batch tra due set di testi"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        embeddings1 = model.encode(texts1, normalize_embeddings=normalize)
        embeddings2 = model.encode(texts2, normalize_embeddings=normalize)
        
        # Calcola similarità coseno
        similarities = torch.cosine_similarity(
            torch.tensor(embeddings1).unsqueeze(1),
            torch.tensor(embeddings2).unsqueeze(0),
            dim=-1
        ).tolist()
        
        return {
            "similarities": similarities,
            "shape": [len(texts1), len(texts2)]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    """Pulisce cache e memoria"""
    cleanup_memory()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1  # Single worker per gestione memoria ottimizzata
    )
