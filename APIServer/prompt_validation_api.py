#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
PROMPT VALIDATION API - ENDPOINT PER VALIDAZIONE E CONFIGURAZIONE
=====================================================================
Autore: Sistema di Classificazione AI
Data: 2025-08-24
Descrizione: API endpoint per validazione strict dei prompt e 
             configurazione obbligatoria per tenant
=====================================================================
"""

from flask import Blueprint, request, jsonify
from typing import Dict, List, Any
import logging
import os
from Utils.prompt_manager import PromptManager
from datetime import datetime

# Blueprint per le API di validazione prompt
prompt_validation_bp = Blueprint('prompt_validation', __name__)

# Logger specifico per la validazione
logger = logging.getLogger('PromptValidationAPI')
logger.setLevel(logging.INFO)

class PromptValidationAPI:
    """
    Classe per gestire validazione e configurazione prompt obbligatoria
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza l'API di validazione prompt
        
        Args:
            config_path: Percorso file configurazione
        """
        # Usa path relativo corretto se non specificato
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        self.prompt_manager = PromptManager(config_path)
        self.logger = logger
    
    def validate_tenant_prompts(self, tenant_id: str) -> Dict[str, Any]:
        """
        Valida tutti i prompt obbligatori per un tenant
        
        Args:
            tenant_id: ID del tenant da validare
            
        Returns:
            Dict con risultato validazione
        """
        try:
            # Definisce prompt obbligatori per il sistema di classificazione
            required_prompts = [
                {
                    'engine': 'LLM',
                    'prompt_type': 'SYSTEM',
                    'prompt_name': 'intelligent_classifier_system',
                    'description': 'Prompt di sistema per il classificatore intelligente'
                },
                {
                    'engine': 'LLM',
                    'prompt_type': 'TEMPLATE',
                    'prompt_name': 'intelligent_classifier_user',
                    'description': 'Template per messaggi utente nel classificatore'
                }
            ]
            
            # Esegue validazione STRICT
            validation_result = self.prompt_manager.validate_tenant_prompts_strict(
                tenant_id=tenant_id,
                required_prompts=required_prompts
            )
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'validation_result': validation_result,
                'required_prompts': required_prompts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'tenant_id': tenant_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_missing_prompts_report(self, tenant_id: str) -> Dict[str, Any]:
        """
        Genera report dettagliato dei prompt mancanti per un tenant
        
        Args:
            tenant_id: ID del tenant
            
        Returns:
            Dict con report dettagliato
        """
        try:
            required_prompts = [
                {
                    'engine': 'LLM',
                    'prompt_type': 'SYSTEM',
                    'prompt_name': 'intelligent_classifier_system',
                    'description': 'Prompt di sistema per il classificatore intelligente',
                    'example_content': 'Sei un classificatore esperto per l\'ospedale...'
                },
                {
                    'engine': 'LLM',
                    'prompt_type': 'TEMPLATE',
                    'prompt_name': 'intelligent_classifier_user',
                    'description': 'Template per messaggi utente nel classificatore',
                    'example_content': 'Analizza questo testo seguendo l\'approccio degli esempi...'
                }
            ]
            
            missing_prompts = []
            available_prompts = []
            
            for prompt_req in required_prompts:
                try:
                    # Controlla esistenza prompt
                    prompt_content = self.prompt_manager.get_prompt(
                        tenant_id=tenant_id,
                        engine=prompt_req['engine'],
                        prompt_type=prompt_req['prompt_type'],
                        prompt_name=prompt_req['prompt_name']
                    )
                    
                    if prompt_content:
                        available_prompts.append({
                            **prompt_req,
                            'status': 'available',
                            'content_length': len(prompt_content)
                        })
                    else:
                        missing_prompts.append({
                            **prompt_req,
                            'status': 'missing'
                        })
                        
                except Exception as e:
                    missing_prompts.append({
                        **prompt_req,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'tenant_id': tenant_id,
                'summary': {
                    'total_required': len(required_prompts),
                    'available': len(available_prompts),
                    'missing': len(missing_prompts),
                    'is_valid': len(missing_prompts) == 0
                },
                'available_prompts': available_prompts,
                'missing_prompts': missing_prompts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'tenant_id': tenant_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Istanza globale dell'API
prompt_validation_api = PromptValidationAPI()

@prompt_validation_bp.route('/validate_tenant_prompts/<tenant_id>', methods=['GET'])
def validate_tenant_prompts_endpoint(tenant_id: str):
    """
    Endpoint per validare prompt obbligatori di un tenant
    
    Args:
        tenant_id: ID del tenant da validare
        
    Returns:
        JSON con risultato validazione
    """
    try:
        result = prompt_validation_api.validate_tenant_prompts(tenant_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"❌ Errore validazione tenant {tenant_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant_id': tenant_id,
            'timestamp': datetime.now().isoformat()
        }), 500

@prompt_validation_bp.route('/missing_prompts_report/<tenant_id>', methods=['GET'])
def missing_prompts_report_endpoint(tenant_id: str):
    """
    Endpoint per report dettagliato prompt mancanti
    
    Args:
        tenant_id: ID del tenant
        
    Returns:
        JSON con report dettagliato
    """
    try:
        result = prompt_validation_api.get_missing_prompts_report(tenant_id)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Errore report missing prompts tenant {tenant_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'tenant_id': tenant_id,
            'timestamp': datetime.now().isoformat()
        }), 500

@prompt_validation_bp.route('/system_status', methods=['GET'])
def system_status_endpoint():
    """
    Endpoint per verificare stato del sistema di gestione prompt
    
    Returns:
        JSON con stato sistema
    """
    try:
        # Verifica connessione database
        if prompt_validation_api.prompt_manager.connect():
            prompt_validation_api.prompt_manager.disconnect()
            db_status = 'connected'
        else:
            db_status = 'disconnected'
        
        return jsonify({
            'success': True,
            'system_status': {
                'prompt_manager_available': True,
                'database_status': db_status,
                'validation_mode': 'strict',
                'fallback_enabled': False
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Errore verifica stato sistema: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'system_status': {
                'prompt_manager_available': False,
                'database_status': 'error',
                'validation_mode': 'strict',
                'fallback_enabled': False
            },
            'timestamp': datetime.now().isoformat()
        }), 500
