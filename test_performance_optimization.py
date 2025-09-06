#!/usr/bin/env python3
"""
Script di test per verificare le performance della funzione get_review_queue_sessions ottimizzata
Autore: Valerio Bignardi
Data: 2025-09-06
"""

import sys
import os
import time
from datetime import datetime

# Aggiungi path per moduli
sys.path.append(os.path.dirname(__file__))

# Import dal progetto
from mongo_classification_reader import MongoClassificationReader

# Aggiungi path per i moduli Utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from tenant import Tenant

def test_performance_optimization():
    """
    Testa le performance della funzione get_review_queue_sessions ottimizzata
    
    Test eseguiti:
    1. Chiamata con tutti i filtri attivati
    2. Chiamata con singoli filtri
    3. Chiamata con tutti i filtri disattivati (early exit test)
    4. Misurazione tempi di esecuzione
    """
    try:
        print("🚀 INIZIO TEST PERFORMANCE OPTIMIZATION")
        print("=" * 60)
        
        # 1. SETUP - Risolvi tenant di test
        print("🔧 Setup test environment...")
        try:
            # Usa tenant Humanitas per il test
            tenant = Tenant.from_slug('humanitas')
            client_name = tenant.tenant_slug
            print(f"✅ Tenant di test: {tenant.tenant_name} ({client_name})")
        except Exception as e:
            print(f"❌ Errore setup tenant: {e}")
            return False
        
        # 2. INIZIALIZZA MONGO READER
        print("🔧 Inizializzazione MongoClassificationReader...")
        mongo_reader = MongoClassificationReader(tenant=tenant)
        
        if not mongo_reader.connect():
            print("❌ Impossibile connettersi a MongoDB")
            return False
        
        print("✅ Connessione MongoDB stabilita")
        
        # 3. TEST CASES - Diversi scenari di filtri
        test_cases = [
            {
                'name': 'TUTTI_FILTRI_ATTIVI',
                'params': {
                    'show_representatives': True,
                    'show_propagated': True,
                    'show_outliers': True,
                    'limit': 50
                }
            },
            {
                'name': 'SOLO_REPRESENTATIVES',
                'params': {
                    'show_representatives': True,
                    'show_propagated': False,
                    'show_outliers': False,
                    'limit': 50
                }
            },
            {
                'name': 'SOLO_OUTLIERS',
                'params': {
                    'show_representatives': False,
                    'show_propagated': False,
                    'show_outliers': True,
                    'limit': 50
                }
            },
            {
                'name': 'TUTTI_FILTRI_DISATTIVATI',
                'params': {
                    'show_representatives': False,
                    'show_propagated': False,
                    'show_outliers': False,
                    'limit': 50
                }
            }
        ]
        
        # 4. ESECUZIONE TEST
        results = []
        
        for test_case in test_cases:
            print(f"\n🧪 TEST: {test_case['name']}")
            print("-" * 40)
            
            # Misura tempo di esecuzione
            start_time = time.time()
            
            try:
                sessions = mongo_reader.get_review_queue_sessions(
                    client_name=client_name,
                    **test_case['params']
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Risultati
                result = {
                    'test_name': test_case['name'],
                    'execution_time': execution_time,
                    'sessions_count': len(sessions),
                    'success': True,
                    'params': test_case['params']
                }
                
                print(f"✅ Successo:")
                print(f"   📊 Sessioni recuperate: {len(sessions)}")
                print(f"   ⏱️ Tempo esecuzione: {execution_time:.3f}s")
                
                # Test early exit per caso tutti filtri disattivati
                if test_case['name'] == 'TUTTI_FILTRI_DISATTIVATI':
                    if len(sessions) == 0 and execution_time < 0.1:
                        print(f"   🚀 EARLY EXIT OK: Tempo < 100ms per filtri disattivati")
                    else:
                        print(f"   ⚠️ EARLY EXIT FAIL: Dovrebbe essere più veloce")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Errore nel test: {e}")
                results.append({
                    'test_name': test_case['name'],
                    'execution_time': None,
                    'sessions_count': 0,
                    'success': False,
                    'error': str(e),
                    'params': test_case['params']
                })
        
        # 5. SUMMARY RISULTATI
        print("\n" + "=" * 60)
        print("📊 SUMMARY RISULTATI PERFORMANCE TEST")
        print("=" * 60)
        
        successful_tests = [r for r in results if r['success']]
        failed_tests = [r for r in results if not r['success']]
        
        print(f"✅ Test riusciti: {len(successful_tests)}/{len(results)}")
        print(f"❌ Test falliti: {len(failed_tests)}")
        
        if successful_tests:
            avg_time = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
            total_sessions = sum(r['sessions_count'] for r in successful_tests)
            
            print(f"⏱️ Tempo medio esecuzione: {avg_time:.3f}s")
            print(f"📊 Sessioni totali recuperate: {total_sessions}")
            
            # Performance benchmark
            if avg_time < 1.0:
                print("🚀 PERFORMANCE: Ottima (< 1s)")
            elif avg_time < 3.0:
                print("⚡ PERFORMANCE: Buona (< 3s)")
            else:
                print("🐌 PERFORMANCE: Da migliorare (> 3s)")
        
        # Dettagli per ogni test
        print(f"\n🔍 DETTAGLI PER TEST:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            time_str = f"{result.get('execution_time', 0):.3f}s" if result.get('execution_time') else "N/A"
            print(f"  {status} {result['test_name']}: {time_str} → {result.get('sessions_count', 0)} sessioni")
        
        return len(failed_tests) == 0
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE NEL TEST: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            mongo_reader.disconnect()
            print("🔌 Disconnessione MongoDB completata")
        except:
            pass

if __name__ == "__main__":
    print(f"🧪 AVVIO TEST PERFORMANCE - {datetime.now().isoformat()}")
    success = test_performance_optimization()
    
    if success:
        print(f"\n🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        exit(0)
    else:
        print(f"\n💥 ALCUNI TEST SONO FALLITI!")
        exit(1)
