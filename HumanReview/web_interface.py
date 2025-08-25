"""
Human Review Web Interface - Interfaccia web per la revisione umana delle classificazioni

Questo modulo fornisce un'interfaccia web intuitiva per permettere agli operatori umani di:
1. Visualizzare i casi che richiedono revisione (ensemble disagreement, low confidence, edge cases)
2. Prendere decisioni informate sulla classificazione corretta
3. Fornire feedback che verrà utilizzato per il retraining dei modelli

L'interfaccia è progettata per essere user-friendly e fornire tutto il contesto necessario
per prendere decisioni di qualità.
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Aggiungi path per i moduli
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'QualityGate'))
from quality_gate_engine import QualityGateEngine, ReviewCase

# Blueprint per le route della review interface
review_bp = Blueprint('review', __name__, url_prefix='/review')

class HumanReviewWebInterface:
    """
    Interfaccia web per la revisione umana delle classificazioni.
    
    Fornisce un'interfaccia user-friendly per visualizzare e decidere sui casi
    che richiedono supervisione umana.
    """
    
    def __init__(self):
        self.quality_gates: Dict[str, QualityGateEngine] = {}
    
    def get_quality_gate(self, tenant_name: str) -> QualityGateEngine:
        """
        Ottieni o crea il QualityGateEngine per un tenant.
        
        Args:
            tenant_name: Nome del tenant
            
        Returns:
            QualityGateEngine per il tenant
        """
        if tenant_name not in self.quality_gates:
            self.quality_gates[tenant_name] = QualityGateEngine(
                tenant_name=tenant_name,
                training_log_path=f"training_decisions_{tenant_name}.jsonl"
            )
        return self.quality_gates[tenant_name]
    
    def get_pending_cases(self, tenant_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recupera i casi pending per un tenant.
        
        Args:
            tenant_name: Nome del tenant
            limit: Numero massimo di casi da recuperare
            
        Returns:
            Lista di casi in attesa di revisione
        """
        quality_gate = self.get_quality_gate(tenant_name)
        pending_cases = quality_gate.get_pending_reviews(tenant=tenant_name, limit=limit)
        
        # Converti ReviewCase in dict per il template
        cases_data = []
        for case in pending_cases:
            case_dict = {
                'case_id': case.case_id,
                'session_id': case.session_id,
                'conversation_text': case.conversation_text[:500] + "..." if len(case.conversation_text) > 500 else case.conversation_text,
                'full_conversation': case.conversation_text,
                'ml_prediction': case.ml_prediction,
                'ml_confidence': round(case.ml_confidence, 3),
                'llm_prediction': case.llm_prediction,
                'llm_confidence': round(case.llm_confidence, 3),
                'uncertainty_score': round(case.uncertainty_score, 3),
                'novelty_score': round(case.novelty_score, 3),
                'reason': case.reason,
                'created_at': case.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'tenant': case.tenant,
                'cluster_id': case.cluster_id
            }
            cases_data.append(case_dict)
        
        return cases_data
    
    def resolve_case(self, tenant_name: str, case_id: str, human_decision: str, 
                     confidence: float, notes: str = "") -> Dict[str, Any]:
        """
        Risolvi un caso di revisione con la decisione umana.
        
        Args:
            tenant_name: Nome del tenant
            case_id: ID del caso
            human_decision: Decisione dell'operatore umano
            confidence: Confidenza dell'operatore nella decisione
            notes: Note aggiuntive (opzionale)
            
        Returns:
            Risultato dell'operazione
        """
        quality_gate = self.get_quality_gate(tenant_name)
        
        try:
            quality_gate.resolve_review_case(
                case_id=case_id,
                human_decision=human_decision,
                confidence=confidence,
                reviewer="web_interface",  # TODO: implementare autenticazione
                notes=notes
            )
            
            return {
                'success': True,
                'message': f'Caso {case_id} risolto con decisione: {human_decision}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Istanza globale dell'interfaccia
review_interface = HumanReviewWebInterface()

@review_bp.route('/<tenant_name>')
def review_dashboard(tenant_name: str):
    """
    Dashboard principale per la revisione di un tenant.
    
    Mostra una panoramica dei casi pending e statistiche.
    """
    try:
        # Recupera casi pending
        pending_cases = review_interface.get_pending_cases(tenant_name, limit=20)
        
        # Recupera statistiche
        quality_gate = review_interface.get_quality_gate(tenant_name)
        stats = quality_gate.get_statistics(tenant=tenant_name)
        
        return render_template('review_dashboard.html', 
                             tenant=tenant_name,
                             pending_cases=pending_cases,
                             stats=stats)
    
    except Exception as e:
        return render_template('error.html', 
                             error=f"Errore nel caricamento dashboard: {str(e)}",
                             tenant=tenant_name)

@review_bp.route('/<tenant_name>/case/<case_id>')
def review_case_detail(tenant_name: str, case_id: str):
    """
    Pagina di dettaglio per la revisione di un caso specifico.
    
    Mostra tutti i dettagli del caso e l'interfaccia per la decisione.
    """
    try:
        quality_gate = review_interface.get_quality_gate(tenant_name)
        pending_cases = quality_gate.get_pending_reviews(tenant=tenant_name, limit=100)
        
        # Trova il caso specifico
        target_case = None
        for case in pending_cases:
            if case.case_id == case_id:
                target_case = case
                break
        
        if not target_case:
            return render_template('error.html', 
                                 error=f"Caso {case_id} non trovato",
                                 tenant=tenant_name)
        
        # Converti in dict per il template
        case_data = {
            'case_id': target_case.case_id,
            'session_id': target_case.session_id,
            'conversation_text': target_case.conversation_text,
            'ml_prediction': target_case.ml_prediction,
            'ml_confidence': round(target_case.ml_confidence, 3),
            'llm_prediction': target_case.llm_prediction,
            'llm_confidence': round(target_case.llm_confidence, 3),
            'uncertainty_score': round(target_case.uncertainty_score, 3),
            'novelty_score': round(target_case.novelty_score, 3),
            'reason': target_case.reason,
            'created_at': target_case.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'tenant': target_case.tenant,
            'cluster_id': target_case.cluster_id
        }
        
        return render_template('case_detail.html', 
                             tenant=tenant_name,
                             case=case_data)
    
    except Exception as e:
        return render_template('error.html', 
                             error=f"Errore nel caricamento caso: {str(e)}",
                             tenant=tenant_name)

@review_bp.route('/<tenant_name>/resolve', methods=['POST'])
def resolve_case_endpoint(tenant_name: str):
    """
    Endpoint per risolvere un caso con la decisione umana.
    
    Accetta dati del form e aggiorna il caso.
    """
    try:
        # Recupera dati dal form
        case_id = request.form.get('case_id')
        human_decision = request.form.get('human_decision')
        confidence = float(request.form.get('confidence', 0.8))
        notes = request.form.get('notes', '')
        
        if not case_id or not human_decision:
            return jsonify({
                'success': False,
                'error': 'Case ID e decisione umana sono richiesti'
            }), 400
        
        # Risolvi il caso
        result = review_interface.resolve_case(
            tenant_name=tenant_name,
            case_id=case_id,
            human_decision=human_decision,
            confidence=confidence,
            notes=notes
        )
        
        if result['success']:
            # Redirect alla dashboard dopo successo
            return redirect(url_for('review.review_dashboard', tenant_name=tenant_name))
        else:
            return jsonify(result), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Errore nella risoluzione del caso: {str(e)}'
        }), 500

@review_bp.route('/<tenant_name>/api/cases')
def api_get_cases(tenant_name: str):
    """
    API endpoint per recuperare i casi pending (per AJAX).
    
    Restituisce JSON con i casi in attesa di revisione.
    """
    try:
        limit = int(request.args.get('limit', 10))
        pending_cases = review_interface.get_pending_cases(tenant_name, limit=limit)
        
        return jsonify({
            'success': True,
            'cases': pending_cases,
            'total': len(pending_cases)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@review_bp.route('/<tenant_name>/api/stats')
def api_get_stats(tenant_name: str):
    """
    API endpoint per recuperare le statistiche (per AJAX).
    
    Restituisce JSON con statistiche aggiornate.
    """
    try:
        quality_gate = review_interface.get_quality_gate(tenant_name)
        stats = quality_gate.get_statistics(tenant=tenant_name)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
