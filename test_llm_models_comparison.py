#!/usr/bin/env python3
"""
Test comparativo per analizzare come diversi modelli LLM gestiscono i function tools
Testa sia mistral:7b che mistral-nemo:latest in modalità standard e RAW
Autore: Valerio Bignardi
Data: 2025-09-01
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
sys.path.append('/home/ubuntu/classificatore')

class LLMToolTester:
    """Tester per comparare diversi modelli e modalità di function calling"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.session = requests.Session()
        self.timeout = 30
        
        # Tool di test standard per classificazione
        self.test_tool = {
            "type": "function",
            "function": {
                "name": "classify_conversation",
                "description": "Classifica una conversazione medica",
                "parameters": {
                    "type": "object",
                    "required": ["predicted_label", "confidence", "motivation"],
                    "properties": {
                        "predicted_label": {
                            "type": "string",
                            "enum": ["prenotazione", "informazioni", "problema_tecnico", "altro"],
                            "description": "Categoria della conversazione"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Livello di confidenza (0.0-1.0)"
                        },
                        "motivation": {
                            "type": "string",
                            "description": "Breve spiegazione del ragionamento"
                        }
                    }
                }
            }
        }
        
        # Conversazione di test
        self.test_conversation = """
        [UTENTE] Ciao, vorrei prenotare una visita cardiologica
        [ASSISTENTE] Certo, la aiuto con la prenotazione. Per quando preferirebbe?
        [UTENTE] La prossima settimana se possibile
        """
    
    def test_standard_function_calling(self, model_name: str) -> dict:
        """Testa function calling standard via /api/chat"""
        print(f"\n🧪 TEST STANDARD FUNCTION CALLING - {model_name}")
        print("="*60)
        
        system_message = """Sei un classificatore di conversazioni mediche.
        Usa SEMPRE la function tool 'classify_conversation' per rispondere.
        Non scrivere testo libero - usa SOLO la function call."""
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Classifica questa conversazione: {self.test_conversation}"}
            ],
            "tools": [self.test_tool],
            "stream": False,
            "options": {
                "temperature": 0.01,
                "num_predict": 200,
                "top_p": 0.8,
                "top_k": 20
            }
        }
        
        try:
            print(f"📤 Invio richiesta a {self.ollama_url}/api/chat")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()
            
            print(f"⏱️  Tempo risposta: {response_time:.2f}s")
            print(f"📥 Risposta completa RAW:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Analizza la risposta
            analysis = self._analyze_response(result, "standard", model_name)
            analysis["response_time"] = response_time
            analysis["success"] = True
            
            return analysis
            
        except Exception as e:
            print(f"❌ ERRORE: {e}")
            return {
                "model": model_name,
                "mode": "standard",
                "success": False,
                "error": str(e),
                "response_time": None,
                "has_tool_calls": False,
                "has_content": False,
                "parsed_result": None
            }
    
    def test_raw_function_calling(self, model_name: str) -> dict:
        """Testa function calling RAW mode via /api/generate"""
        print(f"\n🧪 TEST RAW FUNCTION CALLING - {model_name}")
        print("="*60)
        
        # Costruisce prompt RAW per Mistral function calling
        tools_json = json.dumps([self.test_tool], ensure_ascii=False)
        
        system_message = """Sei un classificatore di conversazioni mediche.
        Usa SEMPRE gli available tools per rispondere."""
        
        raw_prompt = f"""[AVAILABLE_TOOLS] {tools_json}[/AVAILABLE_TOOLS][INST] {system_message}

Classifica questa conversazione medica:
{self.test_conversation} [/INST]"""
        
        payload = {
            "model": model_name,
            "prompt": raw_prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": 0.01,
                "num_predict": 300,
                "top_p": 0.8,
                "top_k": 20
            }
        }
        
        try:
            print(f"📤 Invio richiesta RAW a {self.ollama_url}/api/generate")
            print(f"📝 Prompt RAW (primi 200 char): {raw_prompt[:200]}...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()
            
            print(f"⏱️  Tempo risposta: {response_time:.2f}s")
            print(f"📥 Risposta completa RAW:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Analizza la risposta RAW
            analysis = self._analyze_raw_response(result, model_name)
            analysis["response_time"] = response_time
            analysis["success"] = True
            
            return analysis
            
        except Exception as e:
            print(f"❌ ERRORE: {e}")
            return {
                "model": model_name,
                "mode": "raw",
                "success": False,
                "error": str(e),
                "response_time": None,
                "has_tool_calls": False,
                "has_content": False,
                "parsed_result": None
            }
    
    def _analyze_response(self, result: dict, mode: str, model: str) -> dict:
        """Analizza una risposta standard /api/chat"""
        analysis = {
            "model": model,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "has_message": "message" in result,
            "has_tool_calls": False,
            "has_content": False,
            "tool_calls_count": 0,
            "content_length": 0,
            "parsed_result": None,
            "tool_call_details": None,
            "content_preview": None
        }
        
        if "message" in result:
            message = result["message"]
            
            # Controlla tool_calls
            if "tool_calls" in message and message["tool_calls"]:
                analysis["has_tool_calls"] = True
                analysis["tool_calls_count"] = len(message["tool_calls"])
                
                # Estrae dettagli della prima tool call
                tool_call = message["tool_calls"][0]
                analysis["tool_call_details"] = {
                    "function_name": tool_call.get("function", {}).get("name"),
                    "has_arguments": "arguments" in tool_call.get("function", {}),
                    "arguments_raw": tool_call.get("function", {}).get("arguments")
                }
                
                # Prova a parsare gli argomenti
                if analysis["tool_call_details"]["has_arguments"]:
                    try:
                        arguments = tool_call["function"]["arguments"]
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        analysis["parsed_result"] = arguments
                        print(f"✅ PARSED TOOL CALL: {arguments}")
                    except Exception as e:
                        print(f"⚠️  Errore parsing arguments: {e}")
            
            # Controlla content
            if "content" in message and message["content"]:
                analysis["has_content"] = True
                analysis["content_length"] = len(message["content"])
                analysis["content_preview"] = message["content"][:200]
                
                # Prova parsing fallback se non ci sono tool_calls
                if not analysis["has_tool_calls"]:
                    fallback_result = self._try_fallback_parsing(message["content"])
                    if fallback_result:
                        analysis["parsed_result"] = fallback_result
                        print(f"✅ PARSED FALLBACK: {fallback_result}")
        
        return analysis
    
    def _analyze_raw_response(self, result: dict, model: str) -> dict:
        """Analizza una risposta RAW /api/generate"""
        analysis = {
            "model": model,
            "mode": "raw",
            "timestamp": datetime.now().isoformat(),
            "has_response": "response" in result,
            "response_length": 0,
            "response_preview": None,
            "parsed_result": None,
            "has_tool_calls_format": False
        }
        
        if "response" in result:
            raw_content = result["response"].strip()
            analysis["response_length"] = len(raw_content)
            analysis["response_preview"] = raw_content[:200]
            
            # Controlla formato [TOOL_CALLS]
            if "[TOOL_CALLS]" in raw_content:
                analysis["has_tool_calls_format"] = True
                print(f"✅ TROVATO [TOOL_CALLS] FORMAT")
                
                # Prova a estrarre tool calls
                try:
                    start_idx = raw_content.find("[TOOL_CALLS]") + len("[TOOL_CALLS]")
                    tool_calls_json = raw_content[start_idx:].strip()
                    
                    if tool_calls_json.startswith('[') and tool_calls_json.endswith(']'):
                        tool_calls = json.loads(tool_calls_json)
                        if tool_calls and "arguments" in tool_calls[0]:
                            analysis["parsed_result"] = tool_calls[0]["arguments"]
                            print(f"✅ PARSED RAW TOOL CALL: {analysis['parsed_result']}")
                except Exception as e:
                    print(f"⚠️  Errore parsing RAW tool calls: {e}")
            
            # Fallback parsing se non c'è formato tool calls
            if not analysis["parsed_result"]:
                fallback_result = self._try_fallback_parsing(raw_content)
                if fallback_result:
                    analysis["parsed_result"] = fallback_result
                    print(f"✅ PARSED RAW FALLBACK: {fallback_result}")
        
        return analysis
    
    def _try_fallback_parsing(self, content: str) -> dict:
        """Prova vari metodi di parsing fallback"""
        # Metodo 1: JSON diretto
        try:
            return json.loads(content)
        except:
            pass
        
        # Metodo 2: YAML-like parsing
        try:
            lines = content.strip().split('\n')
            result = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    
                    if key == 'predicted_label':
                        result['predicted_label'] = value
                    elif key == 'confidence':
                        result['confidence'] = float(value)
                    elif key == 'motivation':
                        result['motivation'] = value
            
            if len(result) == 3:  # Tutti i campi trovati
                return result
        except:
            pass
        
        # Metodo 3: Mistral pseudo function call
        try:
            if '**{"name":' in content and '"arguments":' in content:
                start_idx = content.find('"arguments":') + len('"arguments":')
                json_part = content[start_idx:].strip()
                if json_part.endswith('}}'):
                    json_part = json_part[:-1]
                return json.loads(json_part)
        except:
            pass
        
        return None
    
    def run_comprehensive_test(self):
        """Esegue test completo su tutti i modelli e modalità"""
        print("🚀 AVVIO TEST COMPARATIVO COMPLETO")
        print("="*80)
        
        models_to_test = ["mistral:7b", "mistral-nemo:latest"]
        results = []
        
        for model in models_to_test:
            print(f"\n🤖 TESTANDO MODELLO: {model}")
            print("="*60)
            
            # Test standard function calling
            std_result = self.test_standard_function_calling(model)
            results.append(std_result)
            
            # Test RAW function calling
            raw_result = self.test_raw_function_calling(model)
            results.append(raw_result)
        
        # Riassunto finale
        print("\n🎯 RIASSUNTO COMPARATIVO FINALE")
        print("="*80)
        
        for result in results:
            model = result["model"]
            mode = result["mode"]
            success = "✅" if result["success"] else "❌"
            has_tools = "🔧" if result.get("has_tool_calls", False) else "❌"
            has_content = "📝" if result.get("has_content", False) else "❌"
            parsed = "✅" if result.get("parsed_result") else "❌"
            
            print(f"\n{model} ({mode}):")
            print(f"  Status: {success}")
            print(f"  Tool Calls: {has_tools}")
            print(f"  Content: {has_content}")
            print(f"  Parsed Result: {parsed}")
            
            if result.get("response_time"):
                print(f"  Response Time: {result['response_time']:.2f}s")
            
            if result.get("parsed_result"):
                pr = result["parsed_result"]
                # Gestisce sia dict che list (nel caso di mistral:7b che restituisce array)
                if isinstance(pr, list) and len(pr) > 0:
                    pr = pr[0].get("arguments", {}) if "arguments" in pr[0] else pr[0]
                
                print(f"  → Label: {pr.get('predicted_label', 'N/A')}")
                print(f"  → Confidence: {pr.get('confidence', 'N/A')}")
                print(f"  → Motivation: {str(pr.get('motivation', 'N/A'))[:50]}...")
        
        return results

def main():
    tester = LLMToolTester()
    results = tester.run_comprehensive_test()
    
    # Salva risultati in JSON per analisi future
    with open('llm_tool_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Risultati salvati in: llm_tool_test_results.json")

if __name__ == "__main__":
    main()
