#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del cluster_id nelle API di review per verificare la disponibilità dei dati

Autore: AI Assistant
Data: 2025-01-27
Scopo: Verificare che le API restituiscano correttamente il cluster_id per la visualizzazione UI
"""

import requests
import json
import sys
from typing import Dict, List, Any

def test_review_cases_cluster_id(base_url: str = "http://localhost:5000", client: str = "humanitas") -> Dict[str, Any]:
    """
    Testa l'endpoint /api/review/{client}/cases per verificare la presenza di cluster_id
    
    Args:
        base_url: URL base del server
        client: Nome del client/tenant
        
    Returns:
        Dict con risultati del test
    """
    
    endpoint = f"{base_url}/api/review/{client}/cases"
    
    try:
        print(f"🔍 Testando endpoint: {endpoint}")
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "endpoint": endpoint
            }
        
        data = response.json()
        cases = data.get('cases', [])
        
        print(f"📊 Trovati {len(cases)} casi nel review queue")
        
        # Analizza i casi per cluster_id
        cluster_info = {
            "total_cases": len(cases),
            "cases_with_cluster": 0,
            "cases_without_cluster": 0,
            "cluster_ids_found": set(),
            "sample_cases": []
        }
        
        for case in cases[:5]:  # Analizza primi 5 casi come campione
            session_id = case.get('session_id', 'N/A')
            cluster_id = case.get('cluster_id')
            
            if cluster_id is not None:
                cluster_info["cases_with_cluster"] += 1
                cluster_info["cluster_ids_found"].add(str(cluster_id))
            else:
                cluster_info["cases_without_cluster"] += 1
            
            cluster_info["sample_cases"].append({
                "session_id": session_id[:12] + "..." if len(session_id) > 12 else session_id,
                "cluster_id": cluster_id,
                "has_cluster": cluster_id is not None
            })
        
        return {
            "success": True,
            "endpoint": endpoint,
            "cluster_info": cluster_info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "endpoint": endpoint
        }

def test_all_sessions_cluster_id(base_url: str = "http://localhost:5000", client: str = "humanitas") -> Dict[str, Any]:
    """
    Testa l'endpoint /api/review/{client}/all-sessions per verificare la presenza di cluster_id
    
    Args:
        base_url: URL base del server
        client: Nome del client/tenant
        
    Returns:
        Dict con risultati del test
    """
    
    endpoint = f"{base_url}/api/review/{client}/all-sessions"
    
    try:
        print(f"🔍 Testando endpoint: {endpoint}")
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "endpoint": endpoint
            }
        
        data = response.json()
        sessions = data.get('sessions', [])
        
        print(f"📊 Trovate {len(sessions)} sessioni totali")
        
        # Analizza le sessioni per cluster_id nelle classificazioni
        cluster_info = {
            "total_sessions": len(sessions),
            "sessions_with_cluster": 0,
            "sessions_without_cluster": 0,
            "cluster_ids_found": set(),
            "sample_sessions": []
        }
        
        for session in sessions[:5]:  # Analizza prime 5 sessioni come campione
            session_id = session.get('session_id', 'N/A')
            classifications = session.get('classifications', [])
            
            cluster_id = None
            if classifications and len(classifications) > 0:
                cluster_id = classifications[0].get('cluster_id')
            
            if cluster_id is not None:
                cluster_info["sessions_with_cluster"] += 1
                cluster_info["cluster_ids_found"].add(str(cluster_id))
            else:
                cluster_info["sessions_without_cluster"] += 1
            
            cluster_info["sample_sessions"].append({
                "session_id": session_id[:12] + "..." if len(session_id) > 12 else session_id,
                "cluster_id": cluster_id,
                "has_cluster": cluster_id is not None,
                "classifications_count": len(classifications)
            })
        
        return {
            "success": True,
            "endpoint": endpoint,
            "cluster_info": cluster_info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "endpoint": endpoint
        }

def print_test_results(results: Dict[str, Any]) -> None:
    """
    Stampa i risultati del test in formato leggibile
    
    Args:
        results: Risultati del test
    """
    
    print("\n" + "="*60)
    print(f"📊 RISULTATI TEST: {results['endpoint']}")
    print("="*60)
    
    if not results['success']:
        print(f"❌ ERRORE: {results['error']}")
        return
    
    cluster_info = results['cluster_info']
    
    print(f"✅ Test completato con successo")
    print(f"📈 Statistiche cluster:")
    print(f"   - Totale elementi: {cluster_info['total_cases'] if 'total_cases' in cluster_info else cluster_info['total_sessions']}")
    print(f"   - Con cluster_id: {cluster_info['sessions_with_cluster'] if 'sessions_with_cluster' in cluster_info else cluster_info['cases_with_cluster']}")
    print(f"   - Senza cluster_id: {cluster_info['sessions_without_cluster'] if 'sessions_without_cluster' in cluster_info else cluster_info['cases_without_cluster']}")
    print(f"   - Cluster IDs trovati: {sorted(list(cluster_info['cluster_ids_found']))}")
    
    print(f"\n🔍 Campione di elementi:")
    samples = cluster_info['sample_sessions'] if 'sample_sessions' in cluster_info else cluster_info['sample_cases']
    for i, sample in enumerate(samples, 1):
        status = "✅" if sample['has_cluster'] else "❌"
        cluster_display = f"CLUSTER: {sample['cluster_id']}" if sample['has_cluster'] else "Nessun cluster"
        print(f"   {i}. {status} {sample['session_id']} - {cluster_display}")

def main():
    """
    Funzione principale che esegue tutti i test
    """
    
    print("🚀 AVVIO TEST CLUSTER_ID nelle API di Review")
    print("="*60)
    
    # Configurazione
    base_url = "http://localhost:5000"
    client = "humanitas"
    
    # Test 1: Review Cases
    print("\n1️⃣ TEST REVIEW CASES...")
    review_results = test_review_cases_cluster_id(base_url, client)
    print_test_results(review_results)
    
    # Test 2: All Sessions
    print("\n2️⃣ TEST ALL SESSIONS...")
    sessions_results = test_all_sessions_cluster_id(base_url, client)
    print_test_results(sessions_results)
    
    # Riepilogo finale
    print("\n" + "="*60)
    print("📋 RIEPILOGO FINALE")
    print("="*60)
    
    if review_results['success'] and sessions_results['success']:
        print("✅ Tutti i test completati con successo!")
        print("📊 Il cluster_id è disponibile nelle API per la visualizzazione UI")
        
        # Statistiche combinate
        review_clusters = len(review_results['cluster_info']['cluster_ids_found'])
        sessions_clusters = len(sessions_results['cluster_info']['cluster_ids_found'])
        
        print(f"\n📈 Cluster diversi trovati:")
        print(f"   - Review Queue: {review_clusters} cluster")
        print(f"   - All Sessions: {sessions_clusters} cluster")
        
    else:
        print("❌ Alcuni test sono falliti - verificare la connessione al server")
        if not review_results['success']:
            print(f"   - Review Cases: {review_results['error']}")
        if not sessions_results['success']:
            print(f"   - All Sessions: {sessions_results['error']}")

if __name__ == "__main__":
    main()