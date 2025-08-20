#!/usr/bin/env python3
"""
Script per controllare lo stato della pipeline in esecuzione
"""

import sys
import requests
import time
from datetime import datetime

def check_server_status():
    """Controlla se il server è raggiungibile"""
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        return True, response.status_code
    except Exception as e:
        return False, str(e)

def check_pipeline_progress():
    """Tenta di fare una richiesta di test per vedere se la pipeline risponde"""
    try:
        # Prova una richiesta semplice
        response = requests.get("http://localhost:5000/", timeout=5)
        return True, "Server responding"
    except Exception as e:
        return False, str(e)

def main():
    print(f"🔍 CONTROLLO STATO PIPELINE - {datetime.now()}")
    print("-" * 50)
    
    # 1. Controllo server
    server_ok, server_msg = check_server_status()
    print(f"🌐 Server: {'✅ Online' if server_ok else '❌ Offline'} ({server_msg})")
    
    if server_ok:
        print("📡 Server Flask attivo e raggiungibile")
        print("🔄 La pipeline potrebbe essere in esecuzione...")
        print("👤 Se la supervisione umana è abilitata, potrebbe essere in attesa di input")
        print("\n💡 Suggerimenti:")
        print("   - Controlla il terminale dove è in esecuzione server.py per vedere l'output")
        print("   - La pipeline potrebbe essere ferma in attesa di input umano")
        print("   - Cerca messaggi come 'Cluster X: <etichetta>' o prompt di input")
    
    print(f"\n🕐 Controllo completato alle {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
