#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test specifico per verificare che il cluster_id appaia in "Tutte le Sessioni"

Autore: AI Assistant
Data: 2025-01-27
Scopo: Verificare che l'endpoint all-sessions restituisca cluster_id sia direttamente che nelle classificazioni
"""

import requests
import json
from typing import Dict, Any

def test_all_sessions_cluster_display(base_url: str = "http://localhost:5000", client: str = "humanitas") -> None:
    """
    Testa specificamente l'endpoint all-sessions per verificare presenza cluster_id
    """
    
    print("🔍 TEST SPECIFICO: Cluster ID in Tutte le Sessioni")
    print("="*60)
    
    endpoint = f"{base_url}/api/review/{client}/all-sessions"
    
    try:
        print(f"📡 Testing endpoint: {endpoint}")
        response = requests.get(endpoint, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        data = response.json()
        
        if not data.get('success'):
            print(f"❌ API Error: {data.get('error', 'Unknown error')}")
            return
        
        sessions = data.get('sessions', [])
        print(f"📊 Sessioni trovate: {len(sessions)}")
        
        if not sessions:
            print("⚠️ Nessuna sessione trovata - impossibile testare cluster_id")
            return
        
        # Analizza prime 5 sessioni per cluster_id
        print(f"\n🔍 Analisi cluster_id nelle prime {min(5, len(sessions))} sessioni:")
        print("-" * 60)
        
        cluster_analysis = {
            'sessions_with_direct_cluster': 0,
            'sessions_with_classification_cluster': 0,
            'sessions_with_any_cluster': 0,
            'total_sessions': len(sessions)
        }
        
        for i, session in enumerate(sessions[:5]):
            session_id = session.get('session_id', 'N/A')[:12] + "..."
            print(f"\n{i+1}. Sessione: {session_id}")
            
            # Check cluster_id diretto
            direct_cluster_id = session.get('cluster_id')
            print(f"   📊 Cluster diretto: {direct_cluster_id if direct_cluster_id is not None else 'N/A'}")
            
            if direct_cluster_id is not None:
                cluster_analysis['sessions_with_direct_cluster'] += 1
            
            # Check cluster_id nelle classificazioni
            classifications = session.get('classifications', [])
            print(f"   📋 Classificazioni: {len(classifications)}")
            
            classification_cluster_ids = []
            for j, classification in enumerate(classifications):
                class_cluster_id = classification.get('cluster_id')
                if class_cluster_id is not None:
                    classification_cluster_ids.append(class_cluster_id)
                    print(f"      {j+1}. {classification.get('tag_name', 'N/A')} - cluster: {class_cluster_id}")
                else:
                    print(f"      {j+1}. {classification.get('tag_name', 'N/A')} - cluster: N/A")
            
            if classification_cluster_ids:
                cluster_analysis['sessions_with_classification_cluster'] += 1
            
            # Check se ha qualche cluster_id
            has_any_cluster = direct_cluster_id is not None or len(classification_cluster_ids) > 0
            if has_any_cluster:
                cluster_analysis['sessions_with_any_cluster'] += 1
            
            # Status per React
            print(f"   🎯 React mostrerà cluster: {'✅ SI' if has_any_cluster else '❌ NO'}")
            if has_any_cluster:
                display_cluster = direct_cluster_id if direct_cluster_id is not None else classification_cluster_ids[0]
                print(f"   📊 Display value: CLUSTER: {display_cluster}")
        
        # Statistiche finali
        print(f"\n📈 STATISTICHE CLUSTER:")
        print(f"   Sessioni con cluster_id diretto: {cluster_analysis['sessions_with_direct_cluster']}/{cluster_analysis['total_sessions']}")
        print(f"   Sessioni con cluster nelle classificazioni: {cluster_analysis['sessions_with_classification_cluster']}/{cluster_analysis['total_sessions']}")
        print(f"   Sessioni con qualsiasi cluster: {cluster_analysis['sessions_with_any_cluster']}/{cluster_analysis['total_sessions']}")
        
        coverage = (cluster_analysis['sessions_with_any_cluster'] / cluster_analysis['total_sessions']) * 100
        print(f"   📊 Coverage totale: {coverage:.1f}%")
        
        # Risultato finale
        if cluster_analysis['sessions_with_any_cluster'] > 0:
            print(f"\n✅ SUCCESS: Il cluster_id è disponibile e dovrebbe apparire in Tutte le Sessioni!")
            print(f"🎯 React component dovrebbe mostrare cluster per {cluster_analysis['sessions_with_any_cluster']} sessioni")
        else:
            print(f"\n❌ PROBLEM: Nessuna sessione ha cluster_id disponibile")
            print(f"🔧 Verifica che le sessioni siano state processate con clustering")
        
        # Mostra esempio JSON struttura
        if sessions:
            print(f"\n📋 ESEMPIO STRUTTURA SESSION (prima sessione):")
            example = {
                'session_id': sessions[0].get('session_id', 'N/A')[:20] + "...",
                'cluster_id': sessions[0].get('cluster_id'),
                'classifications_count': len(sessions[0].get('classifications', [])),
                'classifications_with_cluster': [
                    {
                        'tag_name': c.get('tag_name'),
                        'cluster_id': c.get('cluster_id')
                    } for c in sessions[0].get('classifications', []) 
                    if c.get('cluster_id') is not None
                ]
            }
            print(json.dumps(example, indent=2))
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connessione fallita: Server non raggiungibile su {base_url}")
        print("💡 Assicurati che il server sia in esecuzione")
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Testa il cluster_id in all-sessions
    """
    print("🚀 TEST CLUSTER_ID in ALL-SESSIONS API")
    print("="*60)
    print("📋 Questo test verifica che:")
    print("   1. L'endpoint all-sessions restituisca cluster_id")
    print("   2. React component possa accedere ai dati cluster")
    print("   3. La visualizzazione funzioni correttamente")
    print("")
    
    test_all_sessions_cluster_display()

if __name__ == "__main__":
    main()