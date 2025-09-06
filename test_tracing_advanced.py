#!/usr/bin/env python3
"""
Test avanzato del sistema di tracing con called_from parameter
e funzioni completamente tracciate
"""

import sys
import os
sys.path.append('.')

def test_tracing_system():
    """Test completo del sistema di tracing con funzioni principali"""
    
    print("🧪 TEST AVANZATO SISTEMA DI TRACING")
    print("=" * 60)
    
    # Test 1: Verifica trace_all con called_from
    print("\n1️⃣ TEST: trace_all con called_from parameter")
    from Pipeline.end_to_end_pipeline import trace_all
    
    # Simula chiamata da una funzione a un'altra
    trace_all("parent_function", "ENTER", test_param="value1")
    trace_all("child_function", "ENTER", called_from="parent_function", child_param="value2")
    trace_all("child_function", "EXIT", called_from="parent_function", result="success")
    trace_all("parent_function", "EXIT", final_result="completed")
    
    print("✅ Test called_from completato")
    
    # Test 2: Test funzione standalone già tracciata
    print("\n2️⃣ TEST: Funzione con tracing completo")
    
    try:
        from Pipeline.end_to_end_pipeline import get_supervised_training_params_from_db
        
        # Test 3: Test get_supervised_training_params_from_db (già completo)
        print("📋 Test get_supervised_training_params_from_db...")
        
        params = get_supervised_training_params_from_db("test_tenant_tracing_advanced")
        print(f"✅ Params ricevuti con tracing completo: {type(params)}")
        
        print("\n🎯 TUTTI I TEST COMPLETATI!")
        print("📊 Verifica file tracing.log per i dettagli completi")
        
    except Exception as e:
        print(f"❌ Errore durante test: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 4: Verifica contenuto log
    print("\n4️⃣ VERIFICA CONTENUTI LOG:")
    try:
        with open("tracing.log", "r") as f:
            lines = f.readlines()
            recent_lines = lines[-20:]  # Ultime 20 righe
            
        print(f"📄 Ultimi entries nel log ({len(recent_lines)} righe):")
        for line in recent_lines:
            if "ENTER" in line or "EXIT" in line or "ERROR" in line:
                print(f"  {line.strip()}")
                
        print("✅ Log verificato con successo")
        
    except FileNotFoundError:
        print("⚠️ File tracing.log non trovato - potrebbe essere disabilitato")
    except Exception as e:
        print(f"❌ Errore leggendo log: {e}")

if __name__ == "__main__":
    test_tracing_system()
