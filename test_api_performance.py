#!/usr/bin/env python3
"""
Script di test per verificare l'integrazione dell'API ottimizzata nel server
Autore: Valerio Bignardi  
Data: 2025-09-06
"""

import requests
import time
import json
from datetime import datetime

def test_api_performance():
    """
    Testa le performance dell'API /api/review/<tenant_id>/cases con la funzione ottimizzata
    """
    try:
        print("🚀 INIZIO TEST API PERFORMANCE")
        print("=" * 60)
        
        # Configurazione test
        base_url = "http://localhost:5000"
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"  # Humanitas UUID
        
        # Test cases con diversi filtri
        test_cases = [
            {
                'name': 'TUTTI_FILTRI_ATTIVI',
                'params': {
                    'limit': 50,
                    'include_representatives': 'true',
                    'include_propagated': 'true', 
                    'include_outliers': 'true'
                }
            },
            {
                'name': 'SOLO_OUTLIERS',
                'params': {
                    'limit': 50,
                    'include_representatives': 'false',
                    'include_propagated': 'false',
                    'include_outliers': 'true'
                }
            },
            {
                'name': 'TUTTI_FILTRI_DISATTIVATI',
                'params': {
                    'limit': 50,
                    'include_representatives': 'false',
                    'include_propagated': 'false',
                    'include_outliers': 'false'
                }
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🧪 TEST API: {test_case['name']}")
            print("-" * 40)
            
            # Costruisci URL con parametri
            url = f"{base_url}/api/review/{tenant_id}/cases"
            
            start_time = time.time()
            
            try:
                response = requests.get(url, params=test_case['params'], timeout=10)
                end_time = time.time()
                execution_time = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    result = {
                        'test_name': test_case['name'],
                        'execution_time': execution_time,
                        'status_code': response.status_code,
                        'cases_count': len(data.get('cases', [])),
                        'success': data.get('success', False),
                        'params': test_case['params']
                    }
                    
                    print(f"✅ Successo:")
                    print(f"   📊 Casi recuperati: {len(data.get('cases', []))}")
                    print(f"   ⏱️ Tempo risposta API: {execution_time:.3f}s")
                    print(f"   🎯 Status: {response.status_code}")
                    
                    # Verifica early exit per filtri disattivati
                    if test_case['name'] == 'TUTTI_FILTRI_DISATTIVATI':
                        if len(data.get('cases', [])) == 0 and execution_time < 1.0:
                            print(f"   🚀 EARLY EXIT OK: Risposta veloce per filtri disattivati")
                        else:
                            print(f"   ⚠️ EARLY EXIT: Potrebbe essere più veloce")
                    
                else:
                    print(f"❌ Errore HTTP: {response.status_code}")
                    print(f"   Risposta: {response.text[:200]}...")
                    result = {
                        'test_name': test_case['name'],
                        'execution_time': execution_time,
                        'status_code': response.status_code,
                        'cases_count': 0,
                        'success': False,
                        'error': f"HTTP {response.status_code}",
                        'params': test_case['params']
                    }
                
                results.append(result)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Errore connessione: {e}")
                results.append({
                    'test_name': test_case['name'],
                    'execution_time': None,
                    'status_code': None,
                    'cases_count': 0,
                    'success': False,
                    'error': str(e),
                    'params': test_case['params']
                })
        
        # Summary risultati
        print("\n" + "=" * 60)
        print("📊 SUMMARY RISULTATI API PERFORMANCE TEST")
        print("=" * 60)
        
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        print(f"✅ Test riusciti: {len(successful_tests)}/{len(results)}")
        print(f"❌ Test falliti: {len(failed_tests)}")
        
        if successful_tests:
            avg_time = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
            total_cases = sum(r['cases_count'] for r in successful_tests)
            
            print(f"⏱️ Tempo medio risposta API: {avg_time:.3f}s")
            print(f"📊 Casi totali recuperati: {total_cases}")
            
            # Performance benchmark
            if avg_time < 1.0:
                print("🚀 PERFORMANCE API: Ottima (< 1s)")
            elif avg_time < 3.0:
                print("⚡ PERFORMANCE API: Buona (< 3s)")
            else:
                print("🐌 PERFORMANCE API: Da migliorare (> 3s)")
        
        # Dettagli per ogni test
        print(f"\n🔍 DETTAGLI PER TEST API:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            time_str = f"{result.get('execution_time', 0):.3f}s" if result.get('execution_time') else "N/A"
            print(f"  {status} {result['test_name']}: {time_str} → {result.get('cases_count', 0)} casi (HTTP {result.get('status_code', 'N/A')})")
        
        return len(failed_tests) == 0
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE NEL TEST API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"🧪 AVVIO TEST API PERFORMANCE - {datetime.now().isoformat()}")
    print("🏃‍♂️ Assicurati che il server sia in esecuzione su localhost:5000")
    
    # Verifica se server è raggiungibile
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server raggiungibile")
        else:
            print(f"⚠️ Server risponde ma status: {response.status_code}")
    except:
        print("❌ Server non raggiungibile su localhost:5000")
        print("💡 Avvia il server con: python3 server.py")
        exit(1)
    
    success = test_api_performance()
    
    if success:
        print(f"\n🎉 TUTTI I TEST API COMPLETATI CON SUCCESSO!")
        exit(0)
    else:
        print(f"\n💥 ALCUNI TEST API SONO FALLITI!")
        exit(1)
