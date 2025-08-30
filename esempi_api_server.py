#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
BLUEPRINT API ESEMPI MULTI-TENANT
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-25
Descrizione: Blueprint Flask per gestione esempi multi-tenant con supporto
             per placeholder {{examples_text}} nei prompt
=====================================================================
"""

from flask import Blueprint, request, jsonify
import yaml
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Aggiungi il percorso del progetto al PYTHONPATH (percorso relativo)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from Utils.prompt_manager import PromptManager
from Utils.tenant import Tenant

# Crea blueprint invece di app Flask
esempi_bp = Blueprint('esempi', __name__)

# Configurazione logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_dir, 'esempi_api.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Inizializzazione PromptManager globale
prompt_manager = None

def init_prompt_manager():
    """Inizializza il PromptManager"""
    global prompt_manager
    try:
        prompt_manager = PromptManager()
        logger.info("✅ PromptManager inizializzato per API esempi")
        return True
    except Exception as e:
        logger.error(f"❌ Errore inizializzazione PromptManager: {e}")
        return False

def validate_tenant_id(tenant_id: str) -> bool:
    """
    Valida che il tenant_id sia fornito
    
    Args:
        tenant_id: ID del tenant da validare
        
    Returns:
        True se valido, False altrimenti
    """
    return tenant_id and len(tenant_id.strip()) > 0

def format_api_response(success: bool, data: Any = None, message: str = "", error: str = ""):
    """
    Formatta la risposta API standardizzata
    
    Args:
        success: Indica se operazione è riuscita
        data: Dati da restituire
        message: Messaggio di successo
        error: Messaggio di errore
        
    Returns:
        Dict con formato standardizzato
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
    }
    
    if success:
        response["message"] = message
        if data is not None:
            response["data"] = data
    else:
        response["error"] = error
        
    return response

# =====================================================================
# ENDPOINT API ESEMPI
# =====================================================================

@esempi_bp.route('/api/examples', methods=['GET'])
def get_examples():
    """
    Recupera lista esempi per un tenant
    
    Query Parameters:
        - tenant_id (required): ID del tenant
        - engine (optional): Tipo di engine (default: LLM)
        - esempio_type (optional): Tipo esempio per filtrare
    """
    try:
        tenant_id = request.args.get('tenant_id')
        engine = request.args.get('engine', 'LLM')
        esempio_type = request.args.get('esempio_type')
        
        if not validate_tenant_id(tenant_id):
            return jsonify(format_api_response(
                success=False,
                error="tenant_id è obbligatorio"
            )), 400
        
        logger.info(f"🔍 GET /api/examples - tenant: {tenant_id}, engine: {engine}")
        
        # Recupera esempi
        examples_list = prompt_manager.get_examples_list(
            tenant_or_id=tenant_id,
            engine=engine,
            esempio_type=esempio_type
        )
        
        return jsonify(format_api_response(
            success=True,
            data=examples_list,
            message=f"Recuperati {len(examples_list)} esempi"
        ))
        
    except Exception as e:
        logger.error(f"❌ Errore GET /api/examples: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

@esempi_bp.route('/api/examples', methods=['POST'])
def create_example():
    """
    Crea nuovo esempio
    
    Body JSON:
        - tenant_id (required): ID del tenant
        - esempio_name (required): Nome esempio
        - esempio_content (required): Contenuto formattato UTENTE:/ASSISTENTE:
        - engine (optional): Tipo di engine (default: LLM)
        - esempio_type (optional): Tipo esempio (default: CONVERSATION)
        - description (optional): Descrizione
        - categoria (optional): Categoria
        - livello_difficolta (optional): Difficoltà (default: MEDIO)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify(format_api_response(
                success=False,
                error="Body JSON richiesto"
            )), 400
        
        # Campi obbligatori
        tenant_id = data.get('tenant_id')
        esempio_name = data.get('esempio_name')
        esempio_content = data.get('esempio_content')
        
        if not validate_tenant_id(tenant_id):
            return jsonify(format_api_response(
                success=False,
                error="tenant_id è obbligatorio"
            )), 400
            
        if not esempio_name:
            return jsonify(format_api_response(
                success=False,
                error="esempio_name è obbligatorio"
            )), 400
            
        if not esempio_content:
            return jsonify(format_api_response(
                success=False,
                error="esempio_content è obbligatorio"
            )), 400
        
        # Campi opzionali con default
        engine = data.get('engine', 'LLM')
        esempio_type = data.get('esempio_type', 'CONVERSATION')
        description = data.get('description')
        categoria = data.get('categoria')
        livello_difficolta = data.get('livello_difficolta', 'MEDIO')
        
        logger.info(f"📝 POST /api/examples - tenant: {tenant_id}, nome: {esempio_name}")
        
        # Crea esempio
        esempio_id = prompt_manager.create_example(
            tenant_or_id=tenant_id,  # ✅ CORRETTO: parametro corretto
            esempio_name=esempio_name,
            esempio_content=esempio_content,
            engine=engine,
            esempio_type=esempio_type,
            description=description,
            categoria=categoria,
            livello_difficolta=livello_difficolta
        )
        
        if esempio_id:
            return jsonify(format_api_response(
                success=True,
                data={"esempio_id": esempio_id},
                message=f"Esempio '{esempio_name}' creato con successo"
            ))
        else:
            return jsonify(format_api_response(
                success=False,
                error="Errore nella creazione dell'esempio"
            )), 500
            
    except Exception as e:
        logger.error(f"❌ Errore POST /api/examples: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

@esempi_bp.route('/api/examples/<int:esempio_id>', methods=['PUT'])
def update_example(esempio_id: int):
    """
    Aggiorna esempio esistente
    
    Path Parameter:
        - esempio_id: ID dell'esempio da aggiornare
        
    Body JSON:
        - tenant_id (required): ID del tenant (per sicurezza)
        - Campi da aggiornare (esempio_name, esempio_content, etc.)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify(format_api_response(
                success=False,
                error="Body JSON richiesto"
            )), 400
        
        tenant_id = data.get('tenant_id')
        
        if not validate_tenant_id(tenant_id):
            return jsonify(format_api_response(
                success=False,
                error="tenant_id è obbligatorio"
            )), 400
        
        # Rimuovi tenant_id dai campi da aggiornare
        updates = {k: v for k, v in data.items() if k != 'tenant_id'}
        
        logger.info(f"🔄 PUT /api/examples/{esempio_id} - tenant: {tenant_id}")
        logger.info(f"🔍 DATI COMPLETI RICEVUTI: {data}")
        logger.info(f"🔍 UPDATES DA APPLICARE: {updates}")
        
        # Risolvi oggetto Tenant dal tenant_id
        try:
            tenant = Tenant.from_uuid(tenant_id)
        except Exception as e:
            logger.error(f"❌ Errore risoluzione tenant {tenant_id}: {e}")
            return jsonify(format_api_response(
                success=False,
                error=f"Tenant non valido: {tenant_id}"
            )), 400
        
        # Aggiorna esempio passando oggetto tenant completo
        success = prompt_manager.update_example(
            esempio_id,       # primo parametro posizionale
            tenant,           # secondo parametro posizionale (oggetto tenant)
            **updates
        )
        
        if success:
            return jsonify(format_api_response(
                success=True,
                message=f"Esempio ID {esempio_id} aggiornato con successo"
            ))
        else:
            return jsonify(format_api_response(
                success=False,
                error="Esempio non trovato o non autorizzato"
            )), 404
            
    except Exception as e:
        logger.error(f"❌ Errore PUT /api/examples/{esempio_id}: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

@esempi_bp.route('/api/examples/<int:esempio_id>', methods=['DELETE'])
def delete_example(esempio_id: int):
    """
    Elimina esempio (soft delete)
    
    Path Parameter:
        - esempio_id: ID dell'esempio da eliminare
        
    Query Parameter:
        - tenant_id (required): ID del tenant (per sicurezza)
    """
    try:
        tenant_id = request.args.get('tenant_id')
        
        if not validate_tenant_id(tenant_id):
            return jsonify(format_api_response(
                success=False,
                error="tenant_id è obbligatorio"
            )), 400
        
        logger.info(f"🗑️ DELETE /api/examples/{esempio_id} - tenant: {tenant_id}")
        
        # Elimina esempio (soft delete)
        success = prompt_manager.delete_example(
            esempio_id=esempio_id,
            tenant_or_id=tenant_id  # ✅ CORRETTO: parametro corretto
        )
        
        if success:
            return jsonify(format_api_response(
                success=True,
                message=f"Esempio ID {esempio_id} eliminato con successo"
            ))
        else:
            return jsonify(format_api_response(
                success=False,
                error="Esempio non trovato o non autorizzato"
            )), 404
            
    except Exception as e:
        logger.error(f"❌ Errore DELETE /api/examples/{esempio_id}: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

@esempi_bp.route('/api/examples/placeholder', methods=['GET'])
def get_examples_placeholder():
    """
    Recupera esempi formattati per placeholder {{examples_text}}
    
    Query Parameters:
        - tenant_id (required): ID del tenant
        - engine (optional): Tipo di engine (default: LLM) 
        - esempio_type (optional): Tipo esempio (default: CONVERSATION)
        - limit (optional): Numero massimo di esempi
    """
    try:
        tenant_id = request.args.get('tenant_id')
        engine = request.args.get('engine', 'LLM')
        esempio_type = request.args.get('esempio_type', 'CONVERSATION')
        limit = request.args.get('limit', type=int)
        
        if not validate_tenant_id(tenant_id):
            return jsonify(format_api_response(
                success=False,
                error="tenant_id è obbligatorio"
            )), 400
        
        logger.info(f"🔄 GET /api/examples/placeholder - tenant: {tenant_id}")
        
        # Recupera esempi per placeholder
        examples_text = prompt_manager.get_examples_for_placeholder(
            tenant_id=tenant_id,
            engine=engine,
            esempio_type=esempio_type,
            limit=limit
        )
        
        # Statistiche
        num_conversations = examples_text.count('UTENTE:') if examples_text else 0
        
        return jsonify(format_api_response(
            success=True,
            data={
                "examples_text": examples_text,
                "num_conversations": num_conversations,
                "length": len(examples_text) if examples_text else 0
            },
            message=f"Recuperati esempi per placeholder: {num_conversations} conversazioni"
        ))
        
    except Exception as e:
        logger.error(f"❌ Errore GET /api/examples/placeholder: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

@esempi_bp.route('/api/prompts/with-examples', methods=['GET'])
def get_prompt_with_examples():
    """
    Recupera prompt con esempi sostituiti nel placeholder {{examples_text}}
    
    Query Parameters:
        - tenant_id (required): ID del tenant
        - engine (required): Tipo di engine
        - prompt_type (required): Tipo di prompt
        - prompt_name (required): Nome del prompt
        - examples_limit (optional): Limite numero esempi
    """
    try:
        tenant_id = request.args.get('tenant_id')
        engine = request.args.get('engine')
        prompt_type = request.args.get('prompt_type')
        prompt_name = request.args.get('prompt_name')
        examples_limit = request.args.get('examples_limit', type=int)
        
        # Validazione parametri obbligatori
        missing_params = []
        if not validate_tenant_id(tenant_id):
            missing_params.append('tenant_id')
        if not engine:
            missing_params.append('engine')
        if not prompt_type:
            missing_params.append('prompt_type')
        if not prompt_name:
            missing_params.append('prompt_name')
            
        if missing_params:
            return jsonify(format_api_response(
                success=False,
                error=f"Parametri obbligatori mancanti: {', '.join(missing_params)}"
            )), 400
        
        logger.info(f"🔄 GET /api/prompts/with-examples - tenant: {tenant_id}, prompt: {prompt_name}")
        
        # Recupera prompt con esempi
        final_prompt = prompt_manager.get_prompt_with_examples(
            tenant_id=tenant_id,
            engine=engine,
            prompt_type=prompt_type,
            prompt_name=prompt_name,
            examples_limit=examples_limit
        )
        
        if final_prompt:
            # Statistiche
            has_examples = 'UTENTE:' in final_prompt
            num_conversations = final_prompt.count('UTENTE:') if has_examples else 0
            
            return jsonify(format_api_response(
                success=True,
                data={
                    "prompt": final_prompt,
                    "has_examples": has_examples,
                    "num_conversations": num_conversations,
                    "length": len(final_prompt)
                },
                message="Prompt con esempi recuperato con successo"
            ))
        else:
            return jsonify(format_api_response(
                success=False,
                error="Prompt non trovato o errore nella sostituzione esempi"
            )), 404
            
    except Exception as e:
        logger.error(f"❌ Errore GET /api/prompts/with-examples: {e}")
        return jsonify(format_api_response(
            success=False,
            error=f"Errore interno: {str(e)}"
        )), 500

# =====================================================================
# ENDPOINT SALUTE E INFORMAZIONI
# =====================================================================

@esempi_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint per verificare lo stato del server"""
    try:
        # Test connessione PromptManager
        test_result = prompt_manager is not None
        
        return jsonify(format_api_response(
            success=True,
            data={
                "status": "healthy",
                "prompt_manager": "connected" if test_result else "disconnected",
                "timestamp": datetime.now().isoformat()
            },
            message="Server esempi API operativo"
        ))
        
    except Exception as e:
        return jsonify(format_api_response(
            success=False,
            error=f"Problema di salute del server: {str(e)}"
        )), 500

@esempi_bp.route('/api/info', methods=['GET'])
def api_info():
    """Informazioni sulle API disponibili"""
    
    endpoints = {
        "GET /api/examples": "Recupera lista esempi per tenant",
        "POST /api/examples": "Crea nuovo esempio",
        "PUT /api/examples/<id>": "Aggiorna esempio esistente",
        "DELETE /api/examples/<id>": "Elimina esempio (soft delete)",
        "GET /api/examples/placeholder": "Recupera esempi formattati per {{examples_text}}",
        "GET /api/prompts/with-examples": "Recupera prompt con esempi sostituiti",
        "GET /health": "Verifica stato server",
        "GET /api/info": "Informazioni API (questo endpoint)"
    }
    
    return jsonify(format_api_response(
        success=True,
        data={
            "service": "API Server Esempi Multi-tenant",
            "version": "1.0.0",
            "endpoints": endpoints
        },
        message="Informazioni API esempi"
    ))

# =====================================================================
# INIZIALIZZAZIONE E AVVIO SERVER
# =====================================================================

# =====================================================================
# INIZIALIZZAZIONE BLUEPRINT
# =====================================================================

# Inizializzazione automatica quando il blueprint viene importato
if not init_prompt_manager():
    print("❌ Impossibile inizializzare PromptManager nel blueprint esempi")
else:
    print("✅ Blueprint esempi inizializzato correttamente")
