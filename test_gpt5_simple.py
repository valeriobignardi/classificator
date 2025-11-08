#!/usr/bin/env python3
"""
Test GPT-5 con chat/completions (non responses API)
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'Services'))
from openai_service import OpenAIService


async def test_gpt5_chat_completions():
    """
    Test GPT-5 usando chat/completions endpoint
    """
    print("\n" + "="*80)
    print("🚀 TEST GPT-5 con chat/completions")
    print("="*80)
    
    service = OpenAIService()
    
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente conciso e utile."
        },
        {
            "role": "user",
            "content": "Dimmi in 2 frasi cosa ti rende migliore di GPT-4o."
        }
    ]
    
    try:
        print("\n🚀 Chiamata a GPT-5 (chat/completions)...")
        
        response = await service.chat_completion(
            model='gpt-5',
            messages=messages,
            temperature=0.7,
            max_tokens=150  # Verrà convertito in max_completion_tokens
        )
        
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            usage = response.get('usage', {})
            
            print(f"\n✅ RISPOSTA GPT-5:")
            print(f"   {content}")
            print(f"\n📊 Token utilizzati:")
            print(f"   Prompt: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Completion: {usage.get('completion_tokens', 'N/A')}")
            print(f"   Totale: {usage.get('total_tokens', 'N/A')}")
            
            return True
        else:
            print(f"\n❌ ERRORE: Risposta non valida")
            print(f"   Response: {response}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERRORE:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gpt5_chat_completions())
    sys.exit(0 if success else 1)
