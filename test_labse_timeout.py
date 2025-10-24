#!/usr/bin/env python3
"""
Test LaBSE con file lungo per verificare timeout e performance
"""
import requests
import time
import json

def test_labse_with_long_file(file_path: str, max_lines: int = 5000):
    """
    Testa LaBSE con un file lungo
    
    Args:
        file_path: Path al file di log
        max_lines: Numero massimo di righe da processare
    """
    print(f"🔍 Lettura file: {file_path}")
    
    # Leggi il file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"📄 Totale righe nel file: {total_lines}")
    
    # Limita il numero di righe
    if total_lines > max_lines:
        lines = lines[:max_lines]
        print(f"⚠️  Limitato a {max_lines} righe per il test")
    
    # Crea testi da embeddings (raggruppa ogni 10 righe)
    texts = []
    chunk_size = 10
    for i in range(0, len(lines), chunk_size):
        chunk = ''.join(lines[i:i+chunk_size])
        if chunk.strip():  # Solo se non vuoto
            texts.append(chunk)
    
    print(f"📦 Creati {len(texts)} testi (chunk size: {chunk_size} righe)")
    
    # Testa LaBSE
    url = "http://localhost:8081/embed"
    
    print(f"\n🚀 Invio richiesta a LaBSE...")
    print(f"   URL: {url}")
    print(f"   Testi: {len(texts)}")
    print(f"   Timeout: 14400 secondi (4 ore)")
    
    payload = {
        "texts": texts,
        "normalize": True
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=14400  # 4 ore come nel backend
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Risposta ricevuta in {elapsed:.2f} secondi")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            embeddings = result.get('embeddings', [])
            print(f"   Embeddings generati: {len(embeddings)}")
            if embeddings:
                print(f"   Dimensione embedding: {len(embeddings[0])}")
                print(f"\n🎉 Test completato con successo!")
            return True
        else:
            print(f"❌ Errore: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n❌ TIMEOUT dopo {elapsed:.2f} secondi")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERRORE dopo {elapsed:.2f} secondi: {str(e)}")
        return False

def test_health_check():
    """Test health check di LaBSE"""
    print("🏥 Test health check...")
    try:
        response = requests.get("http://localhost:8081/health", timeout=10)
        if response.status_code == 200:
            print("✅ LaBSE è online e funzionante")
            return True
        else:
            print(f"❌ LaBSE health check fallito: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LaBSE non raggiungibile: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("TEST LABSE - TIMEOUT E PERFORMANCE")
    print("="*80)
    print()
    
    # Test health check
    if not test_health_check():
        print("\n⚠️  LaBSE non disponibile, esco dal test")
        exit(1)
    
    print("\n" + "="*80)
    print()
    
    # Test con file lungo
    test_labse_with_long_file("/home/ubuntu/classificatore/tracing.log", max_lines=5000)
    
    print("\n" + "="*80)
    print("TEST COMPLETATO")
    print("="*80)
