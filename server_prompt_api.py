#!/usr/bin/env python3
"""
Server Flask semplificato per API di gestione prompt
Autore: Sistema AI
Data: 16 Gennaio 2025

Questo server fornisce gli endpoint CRUD per la gestione dei prompt
senza importare dipendenze pesanti come BERTopic.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
from datetime import datetime

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))

from prompt_manager import PromptManager

app = Flask(__name__)
CORS(app)  # Abilita CORS per permettere richieste dal frontend React

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint di base per verificare che il server sia attivo
    """
    return jsonify({
        'message': 'Prompt Management API Server',
        'version': '1.0',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'prompt-management-api'
    })

# =============================================================================
# TENANT MANAGEMENT ENDPOINTS
# =============================================================================

@app.route('/api/tenants', methods=['GET'])
def get_tenants():
    """
    Recupera la lista dei tenant disponibili
    """
    try:
        # Lista hardcoded dei tenant - in futuro può essere recuperata da DB
        tenants = [
            {
                'id': 1,
                'name': 'humanitas',
                'display_name': 'Humanitas',
                'is_active': True
            },
            {
                'id': 2,
                'name': 'miulli',
                'display_name': 'Miulli',
                'is_active': True
            }
        ]
        
        return jsonify({
            'tenants': tenants,
            'count': len(tenants),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'tenants': []
        }), 500

# =============================================================================
# PROMPT MANAGEMENT API ENDPOINTS
# =============================================================================

@app.route('/api/prompts/tenant/<tenant_id>', methods=['GET'])
def get_prompts_for_tenant(tenant_id: str):
    """
    Recupera tutti i prompt per un tenant specifico
    
    GET /api/prompts/1
    
    Returns:
        [
            {
                "id": 1,
                "tenant_id": 1,
                "tenant_name": "humanitas", 
                "prompt_type": "classification_prompt",
                "content": "...",
                "variables": {...},
                "is_active": true,
                "created_at": "2025-01-16T10:00:00",
                "updated_at": "2025-01-16T10:00:00"
            }
        ]
    """
    try:
        print(f"🔍 API: Recupero prompt per tenant_id: {tenant_id}")
        
        prompt_manager = PromptManager()
        prompts = prompt_manager.get_all_prompts_for_tenant(int(tenant_id))
        
        print(f"✅ Recuperati {len(prompts)} prompt per tenant {tenant_id}")
        
        return jsonify(prompts)
        
    except Exception as e:
        print(f"❌ Errore recupero prompt per tenant {tenant_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'tenant_id': tenant_id
        }), 500


# COMMENTATO: Endpoint ridondante - L'UI usa direttamente /api/prompts/{tenant_id}/status
# @app.route('/api/prompts/tenant/<tenant_id>/status', methods=['GET'])
# def get_prompts_status_for_tenant(tenant_id: str):
#     """
#     Recupera lo status dei prompt per un tenant specifico (accetta tenant_slug)
#     
#     GET /api/prompts/tenant/alleanza/status
#     
#     Returns:
#         {
#             "tenant_id": "alleanza",
#             "tenant_name": "Alleanza",
#             "total_prompts": 5,
#             "active_prompts": 3,
#             "inactive_prompts": 2,
#             "last_updated": "2025-01-16T10:00:00",
#             "status": "ready"
#         }
#     """
#     try:
#         print(f"🔍 API: Recupero status prompt per tenant_slug: {tenant_id}")
#         
#         prompt_manager = PromptManager()
#         prompts = prompt_manager.get_all_prompts_for_tenant(tenant_id)
#         
#         # Calcola statistiche
#         total_prompts = len(prompts)
#         active_prompts = len([p for p in prompts if p.get('is_active', False)])
#         inactive_prompts = total_prompts - active_prompts
#         
#         # Trova ultimo aggiornamento
#         last_updated = None
#         if prompts:
#             last_updated = max(p.get('updated_at', '') for p in prompts if p.get('updated_at'))
#         
#         # Determina tenant name
#         tenant_name = prompts[0].get('tenant_name', 'unknown') if prompts else 'unknown'
#         
#         status = {
#             "tenant_id": tenant_id,
#             "tenant_name": tenant_name,
#             "total_prompts": total_prompts,
#             "active_prompts": active_prompts,
#             "inactive_prompts": inactive_prompts,
#             "last_updated": last_updated,
#             "status": "ready" if active_prompts > 0 else "no_active_prompts"
#         }
#         
#         print(f"✅ Status prompt per tenant {tenant_id}: {active_prompts}/{total_prompts} attivi")
#         
#         return jsonify(status)
#         
#     except Exception as e:
#         print(f"❌ Errore status prompt per tenant {tenant_id}: {e}")
#         return jsonify({
#             'error': str(e),
#             'tenant_id': tenant_id,
#             'status': 'error'
#         }), 500


@app.route('/api/prompts/<tenant_id>/status', methods=['GET'])
def get_prompts_status_by_tenant_id(tenant_id: str):
    """
    Recupera lo status dei prompt per un tenant usando tenant_id completo
    
    GET /api/prompts/a0fd7600-f4f7-11ef-9315-96000228e7fe/status
    
    Returns: Stesso formato dell'endpoint sopra
    """
    try:
        print(f"🔍 API: Recupero status prompt per tenant_id completo: {tenant_id}")
        
        prompt_manager = PromptManager()
        
        # Il PromptManager ora gestisce automaticamente la conversione
        prompts = prompt_manager.get_all_prompts_for_tenant(tenant_id)
        
        # Calcola statistiche
        total_prompts = len(prompts)
        active_prompts = len([p for p in prompts if p.get('is_active', False)])
        inactive_prompts = total_prompts - active_prompts
        
        # Trova ultimo aggiornamento
        last_updated = None
        if prompts:
            last_updated = max(p.get('updated_at', '') for p in prompts if p.get('updated_at'))
        
        # Determina tenant name
        tenant_name = prompts[0].get('tenant_name', 'unknown') if prompts else 'unknown'
        
        status = {
            "success": True,  # AGGIUNTO: Campo success richiesto dall'ApiService UI
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "total_prompts": total_prompts,
            "active_prompts": active_prompts,
            "inactive_prompts": inactive_prompts,
            "last_updated": last_updated,
            "status": "ready" if active_prompts > 0 else "no_active_prompts"
        }
        
        print(f"✅ Status prompt per tenant_id {tenant_id}: {active_prompts}/{total_prompts} attivi")
        
        return jsonify(status)
        
    except Exception as e:
        print(f"❌ Errore status prompt per tenant_id {tenant_id}: {e}")
        return jsonify({
            'success': False,  # AGGIUNTO: Campo success per coerenza
            'error': str(e),
            'tenant_id': tenant_id,
            'status': 'error'
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['GET'])
def get_prompt_by_id(prompt_id: int):
    """
    Recupera un prompt specifico tramite ID
    
    GET /api/prompts/5
    
    Returns:
        {
            "id": 5,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "classification_prompt", 
            "content": "...",
            "variables": {...},
            "is_active": true,
            "created_at": "2025-01-16T10:00:00",
            "updated_at": "2025-01-16T10:00:00"
        }
    """
    try:
        print(f"🔍 API: Recupero prompt con ID: {prompt_id}")
        
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        if prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        print(f"✅ Recuperato prompt ID {prompt_id}")
        
        return jsonify(prompt)
        
    except Exception as e:
        print(f"❌ Errore recupero prompt ID {prompt_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts', methods=['POST'])
def create_prompt():
    """
    Crea un nuovo prompt
    
    POST /api/prompts
    Content-Type: application/json
    
    {
        "tenant_id": 1,
        "tenant_name": "humanitas",
        "prompt_type": "new_classification_prompt",
        "content": "Classifica il testo seguente...",
        "variables": {"param1": "value1"},
        "is_active": true
    }
    
    Returns:
        {
            "id": 6,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "new_classification_prompt",
            "content": "Classifica il testo seguente...",
            "variables": {"param1": "value1"},
            "is_active": true,
            "created_at": "2025-01-16T10:30:00",
            "updated_at": "2025-01-16T10:30:00"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        # Validazione campi obbligatori
        required_fields = ['tenant_id', 'tenant_name', 'prompt_type', 'content']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo {field} mancante'}), 400
        
        print(f"📝 API: Creazione prompt per tenant {data['tenant_name']}")
        print(f"  🏷️ Tipo: {data['prompt_type']}")
        
        prompt_manager = PromptManager()
        prompt_id = prompt_manager.create_prompt(
            tenant_id=data['tenant_id'],
            tenant_name=data['tenant_name'],
            prompt_type=data['prompt_type'],
            content=data['content'],
            variables=data.get('variables', {}),
            is_active=data.get('is_active', True)
        )
        
        # Recupera il prompt creato
        created_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        print(f"✅ Creato prompt ID {prompt_id}")
        
        return jsonify(created_prompt), 201
        
    except Exception as e:
        print(f"❌ Errore creazione prompt: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['PUT'])
def update_prompt(prompt_id: int):
    """
    Aggiorna un prompt esistente
    
    PUT /api/prompts/5
    Content-Type: application/json
    
    {
        "content": "Nuovo contenuto del prompt...",
        "variables": {"new_param": "new_value"},
        "is_active": false
    }
    
    Returns:
        {
            "id": 5,
            "tenant_id": 1,
            "tenant_name": "humanitas",
            "prompt_type": "classification_prompt",
            "content": "Nuovo contenuto del prompt...",
            "variables": {"new_param": "new_value"},
            "is_active": false,
            "created_at": "2025-01-16T10:00:00",
            "updated_at": "2025-01-16T10:35:00"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        print(f"✏️ API: Aggiornamento prompt ID {prompt_id}")
        
        prompt_manager = PromptManager()
        
        # Verifica che il prompt esista
        existing_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        success = prompt_manager.update_prompt(
            prompt_id=prompt_id,
            content=data.get('content'),
            variables=data.get('variables'),
            is_active=data.get('is_active')
        )
        
        if not success:
            return jsonify({
                'error': f'Errore aggiornamento prompt ID {prompt_id}'
            }), 500
        
        # Recupera il prompt aggiornato
        updated_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        
        print(f"✅ Aggiornato prompt ID {prompt_id}")
        
        return jsonify(updated_prompt)
        
    except Exception as e:
        print(f"❌ Errore aggiornamento prompt ID {prompt_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts/<int:prompt_id>', methods=['DELETE'])
def delete_prompt(prompt_id: int):
    """
    Elimina un prompt
    
    DELETE /api/prompts/5
    
    Returns:
        {
            "success": true,
            "message": "Prompt 5 eliminato con successo",
            "prompt_id": 5
        }
    """
    try:
        print(f"🗑️ API: Eliminazione prompt ID {prompt_id}")
        
        prompt_manager = PromptManager()
        
        # Verifica che il prompt esista
        existing_prompt = prompt_manager.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        success = prompt_manager.delete_prompt(prompt_id)
        
        if not success:
            return jsonify({
                'error': f'Errore eliminazione prompt ID {prompt_id}'
            }), 500
        
        print(f"✅ Eliminato prompt ID {prompt_id}")
        
        return jsonify({
            'success': True,
            'message': f'Prompt {prompt_id} eliminato con successo',
            'prompt_id': prompt_id
        })
        
    except Exception as e:
        print(f"❌ Errore eliminazione prompt ID {prompt_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


@app.route('/api/prompts/<int:prompt_id>/preview', methods=['POST'])
def preview_prompt_with_variables(prompt_id: int):
    """
    Anteprima di un prompt con variabili sostituite
    
    POST /api/prompts/5/preview
    Content-Type: application/json
    
    {
        "variables": {
            "conversation_text": "Esempio di conversazione...",
            "available_tags": "tag1, tag2, tag3"
        }
    }
    
    Returns:
        {
            "prompt_id": 5,
            "prompt_type": "classification_prompt",
            "original_content": "Classifica il testo: {{conversation_text}}...",
            "rendered_content": "Classifica il testo: Esempio di conversazione...",
            "variables_used": ["conversation_text", "available_tags"]
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        variables = data.get('variables', {})
        
        print(f"👁️ API: Anteprima prompt ID {prompt_id}")
        print(f"  📝 Variabili: {list(variables.keys())}")
        
        prompt_manager = PromptManager()
        preview = prompt_manager.preview_prompt_with_variables(prompt_id, variables)
        
        if preview is None:
            return jsonify({
                'error': f'Prompt con ID {prompt_id} non trovato'
            }), 404
        
        print(f"✅ Generata anteprima prompt ID {prompt_id}")
        
        return jsonify(preview)
        
    except Exception as e:
        print(f"❌ Errore anteprima prompt ID {prompt_id}: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'prompt_id': prompt_id
        }), 500


# =============================================================================
# ESEMPI LLM E RIADDESTRAMENTO ENDPOINTS
# =============================================================================

@app.route('/api/examples/add-review-case', methods=['POST'])
def add_review_case_as_llm_example():
    """
    Aggiunge un caso di review umana come esempio LLM nel database.
    
    Scopo: Endpoint per salvare conversazioni corrette dall'umano come esempi
    Input: JSON con session_id, conversation_text, etichetta_corretta, categoria, tenant_id
    Output: Risultato dell'operazione con dettagli
    Data ultima modifica: 2025-09-07
    
    Body JSON:
    {
        "session_id": "session_001",
        "conversation_text": "Testo della conversazione...",
        "etichetta_corretta": "PRENOTAZIONE_VISITA", 
        "categoria": "CARDIOLOGIA",
        "tenant_id": 1,
        "note_utente": "L'utente aveva fretta e voleva una visita urgente, per questo è prenotazione"
    }
    
    Returns:
    {
        "success": true,
        "esempio_id": 123,
        "message": "Esempio LLM creato con successo",
        "details": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        # Validazione campi obbligatori
        required_fields = ['session_id', 'conversation_text', 'etichetta_corretta', 'tenant_id']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'error': f'Campi obbligatori mancanti: {", ".join(missing_fields)}'
            }), 400
        
        session_id = data['session_id']
        conversation_text = data['conversation_text']
        etichetta_corretta = data['etichetta_corretta']
        categoria = data.get('categoria')
        note_utente = data.get('note_utente')  # Note dall'interfaccia React
        tenant_id = data['tenant_id']
        
        print(f"📚 [API ESEMPI] Aggiunta caso come esempio LLM")
        print(f"   🆔 Session: {session_id}")
        print(f"   🏷️ Etichetta: {etichetta_corretta}")
        print(f"   🏢 Tenant ID: {tenant_id}")
        if note_utente:
            print(f"   📋 Note: {note_utente[:100]}{'...' if len(note_utente) > 100 else ''}")
        
        # Carica tenant
        try:
            from Utils.tenant_manager import TenantManager
        except ImportError:
            from Utils.mock_tenant_manager import TenantManager
        tenant_manager = TenantManager()
        tenant = tenant_manager.get_tenant_by_id(tenant_id)
        
        if not tenant:
            return jsonify({
                'error': f'Tenant con ID {tenant_id} non trovato'
            }), 404
        
        # Inizializza pipeline per il tenant specifico
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        pipeline = EndToEndPipeline(
            config_path="config.yaml",
            tenant=tenant
        )
        
        # Chiama il nuovo metodo
        result = pipeline.aggiungi_caso_come_esempio_llm(
            session_id=session_id,
            conversation_text=conversation_text,
            etichetta_corretta=etichetta_corretta,
            categoria=categoria,
            note_utente=note_utente
        )
        
        if result['success']:
            print(f"   ✅ Esempio creato: ID {result['esempio_id']}")
            return jsonify(result), 200
        else:
            print(f"   ❌ Errore: {result['message']}")
            return jsonify(result), 400
        
    except Exception as e:
        error_msg = f"Errore nell'aggiunta esempio LLM: {str(e)}"
        print(f"❌ [API ESEMPI ERROR] {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'details': {'exception': str(e)}
        }), 500


@app.route('/api/training/manual-retrain', methods=['POST'])
def manual_retrain_model():
    """
    Riaddestra manualmente il modello ML usando i dati corretti dalla review umana.
    
    Scopo: Endpoint per riaddestramento on-demand del modello
    Input: JSON con tenant_id e opzioni
    Output: Risultato del riaddestramento con metriche
    Data ultima modifica: 2025-09-07
    
    Body JSON:
    {
        "tenant_id": 1,
        "force": false
    }
    
    Returns:
    {
        "success": true,
        "accuracy": 0.85,
        "message": "Riaddestramento completato",
        "training_stats": {...}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dati JSON richiesti'}), 400
        
        tenant_id = data.get('tenant_id')
        force = data.get('force', False)
        
        if not tenant_id:
            return jsonify({'error': 'tenant_id obbligatorio'}), 400
        
        print(f"🔄 [API RIADDESTRAMENTO] Riaddestramento manuale modello")
        print(f"   🏢 Tenant ID: {tenant_id}")
        print(f"   ⚡ Force mode: {force}")
        
        # Carica tenant
        try:
            from Utils.tenant_manager import TenantManager
        except ImportError:
            from Utils.mock_tenant_manager import TenantManager
        tenant_manager = TenantManager()
        tenant = tenant_manager.get_tenant_by_id(tenant_id)
        
        if not tenant:
            return jsonify({
                'error': f'Tenant con ID {tenant_id} non trovato'
            }), 404
        
        # Inizializza pipeline per il tenant specifico
        from Pipeline.end_to_end_pipeline import EndToEndPipeline
        pipeline = EndToEndPipeline(
            config_path="config.yaml",
            tenant=tenant
        )
        
        # Esegui riaddestramento
        result = pipeline.manual_retrain_model(force=force)
        
        if result['success']:
            accuracy = result['accuracy']
            print(f"   ✅ Riaddestramento completato - Accuracy: {accuracy:.3f}")
            return jsonify(result), 200
        else:
            print(f"   ❌ Riaddestramento fallito: {result['message']}")
            return jsonify(result), 400
        
    except Exception as e:
        error_msg = f"Errore nel riaddestramento: {str(e)}"
        print(f"❌ [API RIADDESTRAMENTO ERROR] {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'details': {'exception': str(e)}
        }), 500


@app.route('/api/training/status/<int:tenant_id>', methods=['GET'])
def get_training_status(tenant_id):
    """
    Ottiene lo status del training per un tenant specifico.
    
    Scopo: Verifica stato e statistiche del modello attuale
    Input: tenant_id nell'URL
    Output: Informazioni sullo stato del training
    Data ultima modifica: 2025-09-07
    
    Returns:
    {
        "tenant_id": 1,
        "model_loaded": true,
        "last_training": "2025-01-07T10:30:00",
        "training_samples": 150,
        "accuracy": 0.85,
        "pending_reviews": 5
    }
    """
    try:
        print(f"📊 [API STATUS] Stato training per tenant {tenant_id}")
        
        # Carica tenant
        try:
            from Utils.tenant_manager import TenantManager
        except ImportError:
            from Utils.mock_tenant_manager import TenantManager
        tenant_manager = TenantManager()
        tenant = tenant_manager.get_tenant_by_id(tenant_id)
        
        if not tenant:
            return jsonify({
                'error': f'Tenant con ID {tenant_id} non trovato'
            }), 404
        
        # Carica informazioni MongoDB per dati review
        from mongo_classification_reader import MongoClassificationReader
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        # Conta review pending e completate
        training_data = mongo_reader.get_reviewed_sessions_for_training(
            include_representatives=True,
            include_outliers=True,
            only_human_reviewed=True
        )
        
        # TODO: Implementare conteggio review pending
        # pending_reviews = mongo_reader.count_pending_reviews()
        pending_reviews = 0  # Placeholder
        
        training_samples = len(training_data) if training_data else 0
        
        # Verifica se il modello è caricato (in memoria)
        # TODO: Implementare verifica modello in memoria
        model_loaded = True  # Placeholder
        
        status = {
            "tenant_id": tenant_id,
            "tenant_name": tenant.tenant_name,
            "model_loaded": model_loaded,
            "last_training": datetime.now().isoformat(),  # Placeholder
            "training_samples": training_samples,
            "accuracy": 0.85,  # Placeholder - dovrebbe venire da metriche salvate
            "pending_reviews": pending_reviews
        }
        
        print(f"   ✅ Status recuperato: {training_samples} samples, {pending_reviews} pending")
        return jsonify(status), 200
        
    except Exception as e:
        error_msg = f"Errore nel recupero status: {str(e)}"
        print(f"❌ [API STATUS ERROR] {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'error': error_msg,
            'details': {'exception': str(e)}
        }), 500


if __name__ == "__main__":
    print("🚀 Avvio del Prompt Management API Server")
    print("📡 Server disponibile su http://localhost:5001")
    print("📋 Endpoints disponibili:")
    print("  GET  /api/prompts/tenant/<tenant_id>   - Lista prompt per tenant")
    print("  GET  /api/prompts/<prompt_id>          - Dettagli prompt")
    print("  POST /api/prompts                      - Crea nuovo prompt")
    print("  PUT  /api/prompts/<prompt_id>          - Aggiorna prompt")
    print("  DELETE /api/prompts/<prompt_id>        - Elimina prompt")
    print("  POST /api/prompts/<prompt_id>/preview  - Anteprima prompt")
    print("  GET  /api/tenants                      - Lista tenant")
    print("  POST /api/examples/add-review-case     - 📚 Aggiungi caso come esempio LLM")
    print("  POST /api/training/manual-retrain      - 🔄 Riaddestramento manuale")
    print("  GET  /api/training/status/<tenant_id>  - 📊 Status training")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5001, debug=True)
