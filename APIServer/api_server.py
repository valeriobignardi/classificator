#!/usr/bin/env python3
"""
FastAPI REST API per il sistema di classificazione conversazioni 
Endpoints completi per integrazione con sistemi esterni
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
from datetime import datetime
import json
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Aggiunge path per importare i moduli
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pipeline'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HumanReview'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))

# Inizializza FastAPI
app = FastAPI(
    title="Humanitas Conversation Classification API",
    description="API REST per classificazione automatica delle conversazioni",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, specificare domini specifici
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Modelli Pydantic per request/response
class ConversationText(BaseModel):
    text: str = Field(..., description="Testo della conversazione da classificare")
    session_id: Optional[str] = Field(None, description="ID sessione (opzionale)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadati aggiuntivi")

class BatchConversationRequest(BaseModel):
    conversations: List[ConversationText] = Field(..., description="Lista conversazioni da classificare")
    batch_id: Optional[str] = Field(None, description="ID batch (opzionale)")
    use_ensemble: bool = Field(True, description="Usa ensemble classifier")
    save_to_database: bool = Field(False, description="Salva risultati nel database")

class ClassificationResult(BaseModel):
    predicted_label: str = Field(..., description="Etichetta predetta")
    confidence: float = Field(..., description="Confidenza della predizione")
    is_high_confidence: bool = Field(..., description="Se la confidenza è alta")
    method: str = Field(..., description="Metodo di classificazione usato")
    processing_time_ms: float = Field(..., description="Tempo di elaborazione in ms")
    timestamp: str = Field(..., description="Timestamp della classificazione")
    session_id: Optional[str] = Field(None, description="ID sessione")

class BatchClassificationResult(BaseModel):
    batch_id: str = Field(..., description="ID del batch")
    total_conversations: int = Field(..., description="Numero totale conversazioni")
    successful_classifications: int = Field(..., description="Classificazioni riuscite")
    failed_classifications: int = Field(..., description="Classificazioni fallite")
    results: List[ClassificationResult] = Field(..., description="Risultati individuali")
    processing_time_total_ms: float = Field(..., description="Tempo totale elaborazione")
    statistics: Dict[str, Any] = Field(..., description="Statistiche del batch")

class SystemStatus(BaseModel):
    status: str = Field(..., description="Stato del sistema")
    llm_available: bool = Field(..., description="LLM disponibile")
    ml_model_loaded: bool = Field(..., description="Modello ML caricato")
    ensemble_active: bool = Field(..., description="Ensemble attivo")
    total_classifications_today: int = Field(..., description="Classificazioni totali oggi")
    average_confidence: float = Field(..., description="Confidenza media")
    uptime_seconds: int = Field(..., description="Uptime in secondi")

class TrainingRequest(BaseModel):
    training_data: List[Dict[str, Any]] = Field(..., description="Dati di training")
    validation_split: float = Field(0.2, description="Split per validazione")
    retrain_ensemble: bool = Field(True, description="Riallena ensemble")

class OptimizationRequest(BaseModel):
    optimization_type: str = Field(..., description="Tipo ottimizzazione (threshold/weights/clustering)")
    parameters: Dict[str, Any] = Field({}, description="Parametri per l'ottimizzazione")

# === MODELLI PYDANTIC PER PROMPT MANAGEMENT ===
class PromptModel(BaseModel):
    id: Optional[int] = Field(None, description="ID del prompt")
    tenant_id: str = Field(..., description="ID del tenant")
    engine: str = Field(..., description="Engine (LLM, ML, FINETUNING)")
    prompt_type: str = Field(..., description="Tipo prompt (SYSTEM, TEMPLATE, SPECIALIZED)")
    prompt_name: str = Field(..., description="Nome identificativo del prompt")
    content: str = Field(..., description="Contenuto del prompt")
    variables: Optional[Dict[str, Any]] = Field({}, description="Variabili dinamiche")
    is_active: bool = Field(True, description="Se il prompt è attivo")
    created_at: Optional[str] = Field(None, description="Data creazione")
    updated_at: Optional[str] = Field(None, description="Data ultimo aggiornamento")

class CreatePromptRequest(BaseModel):
    tenant_id: str = Field(..., description="ID del tenant")
    engine: str = Field(..., description="Engine (LLM, ML, FINETUNING)")
    prompt_type: str = Field(..., description="Tipo prompt (SYSTEM, TEMPLATE, SPECIALIZED)")
    prompt_name: str = Field(..., description="Nome identificativo del prompt")
    content: str = Field(..., description="Contenuto del prompt")
    variables: Optional[Dict[str, Any]] = Field({}, description="Variabili dinamiche")
    is_active: bool = Field(True, description="Se il prompt è attivo")

class UpdatePromptRequest(BaseModel):
    content: Optional[str] = Field(None, description="Nuovo contenuto del prompt")
    variables: Optional[Dict[str, Any]] = Field(None, description="Nuove variabili dinamiche")
    is_active: Optional[bool] = Field(None, description="Nuovo stato attivo")

class PromptListResponse(BaseModel):
    prompts: List[PromptModel] = Field(..., description="Lista prompt del tenant")
    total_count: int = Field(..., description="Numero totale prompt")

class PromptPreviewResponse(BaseModel):
    prompt_id: int = Field(..., description="ID del prompt")
    content: str = Field(..., description="Contenuto processato con variabili")
    variables_resolved: Dict[str, Any] = Field(..., description="Variabili risolte")

# Variabili globali per la pipeline
pipeline = None
ensemble_classifier = None
threshold_optimizer = None
prompt_manager = None
start_time = datetime.now()
classification_stats = {
    'total_today': 0,
    'total_all_time': 0,
    'confidence_sum': 0.0,
    'high_confidence_count': 0
}

async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Validazione API key (in produzione usare sistema più robusto)
    """
    # Per demo, accetta qualsiasi token
    # In produzione: validare token JWT o API key dal database
    if credentials.credentials:
        return credentials.credentials
    raise HTTPException(status_code=401, detail="API key richiesta")

@app.on_event("startup")
async def startup_event():
    """
    Inizializzazione dell'applicazione
    """
    global pipeline, ensemble_classifier, threshold_optimizer, prompt_manager
    
    logger.info("🚀 Inizializzazione API Humanitas...")
    
    try:
        # Inizializza pipeline
        from end_to_end_pipeline import EndToEndPipeline
        pipeline = EndToEndPipeline(
            tenant_slug="humanitas",
            confidence_threshold=0.7,
            auto_mode=True
            # auto_retrain ora gestito da config.yaml
        )
        
        # Inizializza ensemble classifier avanzato
        from advanced_ensemble_classifier import AdvancedEnsembleClassifier
        ensemble_classifier = AdvancedEnsembleClassifier(
            llm_classifier=pipeline.llm_classifier,
            confidence_threshold=0.7
        )
        
        # Inizializza ottimizzatore
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'OptimizationEngine'))
        from advanced_threshold_optimizer import AdvancedThresholdOptimizer
        threshold_optimizer = AdvancedThresholdOptimizer()
        
        # Inizializza PromptManager per gestione prompt database-driven
        try:
            from prompt_manager import PromptManager
            prompt_manager = PromptManager()
            logger.info("✅ PromptManager inizializzato")
        except Exception as e:
            logger.warning(f"⚠️ PromptManager non disponibile: {e}")
            prompt_manager = None
        
        logger.info("✅ API inizializzata con successo")
        
    except Exception as e:
        logger.error(f"❌ Errore nell'inizializzazione: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Endpoint root con informazioni API
    """
    return {
        "service": "Humanitas Conversation Classification API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """
    Health check del sistema
    """
    global pipeline, ensemble_classifier, start_time, classification_stats
    
    uptime = (datetime.now() - start_time).total_seconds()
    avg_confidence = (classification_stats['confidence_sum'] / 
                     max(1, classification_stats['total_today']))
    
    return SystemStatus(
        status="healthy" if pipeline else "degraded",
        llm_available=pipeline.llm_classifier.is_available() if pipeline else False,
        ml_model_loaded=pipeline.classifier.classifier is not None if pipeline else False,
        ensemble_active=ensemble_classifier is not None,
        total_classifications_today=classification_stats['total_today'],
        average_confidence=avg_confidence,
        uptime_seconds=int(uptime)
    )

@app.post("/classify", response_model=ClassificationResult)
async def classify_conversation(
    conversation: ConversationText,
    api_key: str = Depends(get_api_key)
):
    """
    Classifica una singola conversazione
    """
    global ensemble_classifier, classification_stats
    
    if not ensemble_classifier:
        raise HTTPException(status_code=503, detail="Ensemble classifier non disponibile")
    
    start_time_ms = datetime.now()
    
    try:
        # Classificazione
        result = ensemble_classifier.predict_with_ensemble(conversation.text)
        
        # Calcola tempo di elaborazione
        processing_time = (datetime.now() - start_time_ms).total_seconds() * 1000
        
        # Aggiorna statistiche
        classification_stats['total_today'] += 1
        classification_stats['total_all_time'] += 1
        classification_stats['confidence_sum'] += result['confidence']
        if result['is_high_confidence']:
            classification_stats['high_confidence_count'] += 1
        
        return ClassificationResult(
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            is_high_confidence=result['is_high_confidence'],
            method=result['method'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            session_id=conversation.session_id
        )
        
    except Exception as e:
        logger.error(f"Errore nella classificazione: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nella classificazione: {str(e)}")

@app.post("/classify/batch", response_model=BatchClassificationResult)
async def classify_batch(
    request: BatchConversationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Classifica un batch di conversazioni
    """
    global ensemble_classifier, pipeline
    
    if not ensemble_classifier:
        raise HTTPException(status_code=503, detail="Ensemble classifier non disponibile")
    
    batch_id = request.batch_id or f"batch_{int(datetime.now().timestamp())}"
    start_time_batch = datetime.now()
    
    logger.info(f"🔄 Processo batch {batch_id} con {len(request.conversations)} conversazioni")
    
    try:
        results = []
        failed_count = 0
        label_counts = {}
        confidence_scores = []
        
        # Estrai testi
        texts = [conv.text for conv in request.conversations]
        
        # Batch prediction
        batch_predictions = ensemble_classifier.batch_predict(texts)
        
        # Processa risultati
        for i, (conv, prediction) in enumerate(zip(request.conversations, batch_predictions)):
            try:
                result = ClassificationResult(
                    predicted_label=prediction['predicted_label'],
                    confidence=prediction['confidence'],
                    is_high_confidence=prediction['is_high_confidence'],
                    method=prediction['method'],
                    processing_time_ms=0,  # Calcolato per l'intero batch
                    timestamp=datetime.now().isoformat(),
                    session_id=conv.session_id
                )
                
                results.append(result)
                
                # Statistiche
                label = prediction['predicted_label']
                label_counts[label] = label_counts.get(label, 0) + 1
                confidence_scores.append(prediction['confidence'])
                
            except Exception as e:
                logger.error(f"Errore nella conversazione {i}: {e}")
                failed_count += 1
        
        # Tempo totale
        total_processing_time = (datetime.now() - start_time_batch).total_seconds() * 1000
        
        # Statistiche batch
        statistics = {
            'label_distribution': label_counts,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'high_confidence_rate': sum(1 for r in results if r.is_high_confidence) / len(results) if results else 0.0,
            'processing_rate_per_second': len(results) / (total_processing_time / 1000) if total_processing_time > 0 else 0.0
        }
        
        # Salva nel database se richiesto
        if request.save_to_database and pipeline:
            background_tasks.add_task(save_batch_to_database, request.conversations, results)
        
        # Aggiorna statistiche globali
        classification_stats['total_today'] += len(results)
        classification_stats['total_all_time'] += len(results)
        classification_stats['confidence_sum'] += sum(confidence_scores)
        classification_stats['high_confidence_count'] += sum(1 for r in results if r.is_high_confidence)
        
        logger.info(f"✅ Batch {batch_id} completato: {len(results)} successi, {failed_count} errori")
        
        return BatchClassificationResult(
            batch_id=batch_id,
            total_conversations=len(request.conversations),
            successful_classifications=len(results),
            failed_classifications=failed_count,
            results=results,
            processing_time_total_ms=total_processing_time,
            statistics=statistics
        )
        
    except Exception as e:
        logger.error(f"Errore nel batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel batch processing: {str(e)}")

@app.post("/train")
async def retrain_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Riallena il modello con nuovi dati
    """
    global pipeline, ensemble_classifier
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline non disponibile")
    
    # Avvia training in background
    background_tasks.add_task(perform_retraining, request)
    
    return {
        "message": "Training avviato in background",
        "training_data_size": len(request.training_data),
        "validation_split": request.validation_split,
        "status": "started"
    }

@app.post("/optimize")
async def optimize_system(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Ottimizza parametri del sistema
    """
    global threshold_optimizer, pipeline, ensemble_classifier
    
    if not threshold_optimizer:
        raise HTTPException(status_code=503, detail="Ottimizzatore non disponibile")
    
    # Avvia ottimizzazione in background
    background_tasks.add_task(perform_optimization, request)
    
    return {
        "message": f"Ottimizzazione {request.optimization_type} avviata",
        "parameters": request.parameters,
        "status": "started"
    }

@app.get("/statistics")
async def get_statistics(api_key: str = Depends(get_api_key)):
    """
    Statistiche del sistema
    """
    global classification_stats, pipeline, ensemble_classifier
    
    # Statistiche ensemble
    ensemble_stats = {}
    if ensemble_classifier:
        ensemble_stats = ensemble_classifier.get_ensemble_statistics()
    
    # Statistiche pipeline
    pipeline_stats = {}
    if pipeline:
        try:
            pipeline_stats = pipeline.get_statistiche_database()
        except:
            pipeline_stats = {"error": "Impossibile recuperare statistiche database"}
    
    return {
        "classification_stats": classification_stats,
        "ensemble_stats": ensemble_stats,
        "pipeline_stats": pipeline_stats,
        "uptime_seconds": int((datetime.now() - start_time).total_seconds())
    }

@app.get("/models/status")
async def get_models_status(api_key: str = Depends(get_api_key)):
    """
    Stato dei modelli caricati
    """
    global pipeline, ensemble_classifier
    
    return {
        "pipeline_loaded": pipeline is not None,
        "llm_available": pipeline.llm_classifier.is_available() if pipeline else False,
        "ml_model_loaded": pipeline.classifier.classifier is not None if pipeline else False,
        "ensemble_loaded": ensemble_classifier is not None,
        "semantic_memory_loaded": pipeline.semantic_memory is not None if pipeline else False
    }

# Background tasks
async def save_batch_to_database(conversations: List[ConversationText], results: List[ClassificationResult]):
    """
    Salva batch nel database (background task)
    """
    global pipeline
    
    try:
        logger.info("💾 Salvataggio batch nel database...")
        
        # Simula salvataggio (implementare logica reale)
        await asyncio.sleep(1)  # Simula operazione DB
        
        logger.info("✅ Batch salvato nel database")
        
    except Exception as e:
        logger.error(f"❌ Errore nel salvataggio batch: {e}")

async def perform_retraining(request: TrainingRequest):
    """
    Esegue il retraining (background task)
    """
    global pipeline, ensemble_classifier
    
    try:
        logger.info("🎓 Avvio retraining del modello...")
        
        # Simula retraining (implementare logica reale)
        await asyncio.sleep(10)  # Simula training
        
        logger.info("✅ Retraining completato")
        
    except Exception as e:
        logger.error(f"❌ Errore nel retraining: {e}")

async def perform_optimization(request: OptimizationRequest):
    """
    Esegue l'ottimizzazione (background task)
    """
    global threshold_optimizer, pipeline, ensemble_classifier
    
    try:
        logger.info(f"🔧 Avvio ottimizzazione {request.optimization_type}...")
        
        # Simula ottimizzazione (implementare logica reale)
        await asyncio.sleep(5)  # Simula ottimizzazione
        
        logger.info("✅ Ottimizzazione completata")
        
    except Exception as e:
        logger.error(f"❌ Errore nell'ottimizzazione: {e}")

# =====================================================================
# PROMPT MANAGEMENT ENDPOINTS - CRUD OPERATIONS
# =====================================================================

@app.get("/api/prompts/{tenant_id}", response_model=PromptListResponse)
async def get_tenant_prompts(
    tenant_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Recupera tutti i prompt di un tenant specifico
    """
    global prompt_manager
    
    try:
        if not prompt_manager:
            raise HTTPException(
                status_code=503, 
                detail="PromptManager non disponibile"
            )
        
        # Recupera prompt dal database
        prompt_data = prompt_manager.get_all_prompts_for_tenant(tenant_id)
        
        # Converte in modelli Pydantic
        prompts = []
        for p in prompt_data:
            prompts.append(PromptModel(**p))
        
        logger.info(f"📋 Recuperati {len(prompts)} prompt per tenant {tenant_id}")
        
        return PromptListResponse(
            prompts=prompts,
            total_count=len(prompts)
        )
        
    except Exception as e:
        logger.error(f"❌ Errore recupero prompt per tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts", response_model=PromptModel)
async def create_prompt(
    request: CreatePromptRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Crea un nuovo prompt per un tenant
    """
    global prompt_manager
    
    try:
        if not prompt_manager:
            raise HTTPException(
                status_code=503,
                detail="PromptManager non disponibile"
            )
        
        # Crea prompt nel database
        prompt_id = prompt_manager.create_prompt(
            tenant_id=request.tenant_id,
            engine=request.engine,
            prompt_type=request.prompt_type,
            prompt_name=request.prompt_name,
            content=request.content,
            variables=request.variables,
            is_active=request.is_active
        )
        
        if not prompt_id:
            raise HTTPException(
                status_code=500,
                detail="Errore creazione prompt nel database"
            )
        
        # Recupera il prompt appena creato
        prompt_data = prompt_manager.get_prompt_by_id(prompt_id)
        if not prompt_data:
            raise HTTPException(
                status_code=500,
                detail="Errore recupero prompt creato"
            )
        
        logger.info(f"➕ Creato nuovo prompt {request.prompt_name} per tenant {request.tenant_id}")
        
        return PromptModel(**prompt_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Errore creazione prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/prompts/{prompt_id}", response_model=PromptModel)
async def update_prompt(
    prompt_id: int,
    request: UpdatePromptRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Aggiorna un prompt esistente
    """
    global prompt_manager
    
    try:
        if not prompt_manager:
            raise HTTPException(
                status_code=503,
                detail="PromptManager non disponibile"
            )
        
        # Aggiorna prompt nel database
        success = prompt_manager.update_prompt(
            prompt_id=prompt_id,
            content=request.content,
            variables=request.variables,
            is_active=request.is_active
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt {prompt_id} non trovato o errore aggiornamento"
            )
        
        # Recupera prompt aggiornato
        prompt_data = prompt_manager.get_prompt_by_id(prompt_id)
        if not prompt_data:
            raise HTTPException(
                status_code=500,
                detail="Errore recupero prompt aggiornato"
            )
        
        logger.info(f"✏️ Aggiornato prompt {prompt_id}")
        
        return PromptModel(**prompt_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Errore aggiornamento prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: int,
    api_key: str = Depends(get_api_key)
):
    """
    Elimina un prompt
    """
    global prompt_manager
    
    try:
        if not prompt_manager:
            raise HTTPException(
                status_code=503,
                detail="PromptManager non disponibile"
            )
        
        # Elimina (disattiva) prompt nel database
        success = prompt_manager.delete_prompt(prompt_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt {prompt_id} non trovato o già eliminato"
            )
        
        logger.info(f"🗑️ Eliminato prompt {prompt_id}")
        
        return {"message": f"Prompt {prompt_id} eliminato con successo"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Errore eliminazione prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/{prompt_id}/preview", response_model=PromptPreviewResponse)
async def preview_prompt(
    prompt_id: int,
    api_key: str = Depends(get_api_key)
):
    """
    Genera anteprima di un prompt con variabili risolte
    """
    global prompt_manager
    
    try:
        if not prompt_manager:
            raise HTTPException(
                status_code=503,
                detail="PromptManager non disponibile"
            )
        
        # Genera anteprima con variabili risolte
        preview_data = prompt_manager.preview_prompt_with_variables(prompt_id)
        
        if not preview_data:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt {prompt_id} non trovato"
            )
        
        logger.info(f"👀 Anteprima generata per prompt {prompt_id}")
        
        return PromptPreviewResponse(**preview_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Errore anteprima prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configurazione per sviluppo
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
