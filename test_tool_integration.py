#!/usr/bin/env python3
"""
Test Script: Verifica integrazione tool database nel IntelligentClassifier

Autore: Valerio Bignardi
Data: 2025-08-30
Scopo: Testare la nuova implementazione che recupera i tool dal database
       invece di usare definizioni hardcoded
"""

import sys
import json
from Utils.prompt_manager import PromptManager
from Utils.tool_manager import ToolManager

def test_tool_integration():
    """
    Testa l'integrazione completa prompt -> tool IDs -> tools dal database
    """
    print("🧪 TEST: Integrazione Tool Database")
    print("=" * 60)
    
    try:
        # 1. Inizializza i manager
        prompt_manager = PromptManager()
        tool_manager = ToolManager()
        
        # Il PromptManager ha il metodo connect, il ToolManager gestisce le connessioni internamente
        if not prompt_manager.connect():
            print("❌ Errore connessione PromptManager")
            return False
        
        # 2. Simula il recupero come nel classifier
        print("\n🔍 STEP 1: Recupero tool IDs dal prompt")
        
        # Usa il tenant_id corretto e ora il prompt dovrebbe avere ID numerici
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        prompt_name = "intelligent_classifier_system"
        engine = "LLM"
        
        tool_ids = prompt_manager.get_prompt_tools(
            tenant_id=tenant_id,
            prompt_name=prompt_name, 
            engine=engine
        )
        
        print(f"📋 Prompt: {engine}/{prompt_name}")
        print(f"👤 Tenant ID: {tenant_id}")
        print(f"🔧 Tool IDs recuperati: {tool_ids}")
        
        if not tool_ids:
            print("⚠️ Nessun tool ID trovato per questo prompt")
            
            # Proviamo a vedere quali prompt esistono
            print("\n🔍 DEBUG: Controllo prompt disponibili...")
            cursor = prompt_manager.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, prompt_name, prompt_type, engine, tenant_id, tools
                FROM prompts 
                WHERE is_active = 1 
                AND tools IS NOT NULL 
                AND tools != 'null'
                LIMIT 10
            """)
            
            prompts = cursor.fetchall()
            print(f"📊 Trovati {len(prompts)} prompt con tools:")
            
            for p in prompts:
                tools_preview = p['tools'][:100] + "..." if p['tools'] and len(p['tools']) > 100 else p['tools']
                print(f"  ID:{p['id']} | {p['engine']}/{p['prompt_type']}/{p['prompt_name']} | Tenant:{p['tenant_id']}")
                print(f"    Tools: {tools_preview}")
            
            cursor.close()
            return False
        
        # 3. Recupera i tool dal database
        print(f"\n🔍 STEP 2: Recupero tool dal database per {len(tool_ids)} IDs")
        
        classification_tools = []
        
        for tool_id in tool_ids:
            print(f"\n🛠️ Recupero tool ID: {tool_id}")
            
            db_tool = tool_manager.get_tool_by_id(tool_id)
            
            if db_tool:
                print(f"✅ Tool trovato: {db_tool['tool_name']}")
                print(f"   Descrizione: {db_tool['description']}")
                print(f"   Schema: {str(db_tool['function_schema'])[:100]}...")
                
                # Costruisce il tool per Ollama
                try:
                    classification_tool = {
                        "type": "function",
                        "function": {
                            "name": db_tool['tool_name'],
                            "description": db_tool['description'], 
                            "parameters": db_tool['function_schema']
                        }
                    }
                    
                    classification_tools.append(classification_tool)
                    print(f"✅ Tool convertito per Ollama")
                    
                except Exception as e:
                    print(f"❌ Errore conversione tool: {e}")
                    
            else:
                print(f"❌ Tool ID {tool_id} non trovato nel database")
        
        # 4. Verifica finale
        print(f"\n🎯 RISULTATO FINALE:")
        print(f"✅ Tool recuperati con successo: {len(classification_tools)}")
        
        if classification_tools:
            print(f"🔧 Tools disponibili per Ollama:")
            for i, tool in enumerate(classification_tools, 1):
                print(f"   {i}. {tool['function']['name']}")
            
            print("\n🧪 TEST COMPLETATO CON SUCCESSO! ✅")
            return True
        else:
            print("❌ Nessun tool valido recuperato")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup connections
        try:
            if 'prompt_manager' in locals():
                prompt_manager.connection.close()
            # ToolManager gestisce le connessioni internamente, non serve cleanup esplicito
        except:
            pass

if __name__ == "__main__":
    success = test_tool_integration()
    sys.exit(0 if success else 1)
