#!/usr/bin/env python3
"""
Test rapido del sistema di tracing per verificare che funzioni correttamente

Autore: Valerio Bignardi
Data: 2025-09-06
"""

import sys
import os

# Aggiungi il path della pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pipeline'))

def test_tracing():
    """
    Test del sistema di tracing
    """
    print("🧪 Test del sistema di tracing...")
    
    try:
        # Importa la funzione trace_all
        from end_to_end_pipeline import trace_all, get_supervised_training_params_from_db
        
        print("✅ Import della funzione trace_all riuscito")
        
        # Test del tracing semplice
        trace_all("test_function", "ENTER", param1="valore1", param2=123)
        print("✅ Test trace_all ENTER riuscito")
        
        # Test del tracing con return value
        trace_all("test_function", "EXIT", return_value={"result": "success", "data": [1,2,3]})
        print("✅ Test trace_all EXIT riuscito")
        
        # Test del tracing con eccezione
        try:
            raise ValueError("Test exception")
        except Exception as e:
            trace_all("test_function", "ERROR", exception=e)
            print("✅ Test trace_all ERROR riuscito")
        
        # Test di una funzione reale con tracing
        print("\n🔧 Test funzione reale con tracing...")
        result = get_supervised_training_params_from_db("test-tenant-id")
        print(f"✅ Funzione completata, risultato tipo: {type(result)}")
        
        # Verifica che il file tracing.log sia stato creato
        tracing_file = "tracing.log"
        if os.path.exists(tracing_file):
            print(f"✅ File {tracing_file} creato correttamente")
            
            # Mostra le ultime righe del file
            with open(tracing_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"\n📄 Ultime {min(10, len(lines))} righe del tracing.log:")
                for line in lines[-10:]:
                    print(f"  {line.strip()}")
        else:
            print(f"⚠️ File {tracing_file} non trovato")
        
        print("\n🎉 Test del sistema di tracing completato con successo!")
        
    except Exception as e:
        print(f"❌ Errore nel test del tracing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tracing()
