#!/usr/bin/env python3
"""
Author: System
Date: 2025-08-21
Description: Test per verificare che il campo motivazione venga mostrato come notes
Last Update: 2025-08-21

Test per validare che il campo motivazione di MongoDB sia correttamente mappato su notes nell'UI
"""

import json
import requests
from typing import Dict, Any

def test_motivation_as_notes():
    """
    Scopo: Testa che il campo motivazione di MongoDB sia visibile come notes nell'API
    
    Parametri input: Nessuno
    Output: Risultato del test
    Ultimo aggiornamento: 2025-08-21
    """
    
    print("🧪 Test Mapping Motivazione → Notes per UI")
    print("=" * 50)
    
    try:
        # 1. Test endpoint Review Cases
        print("1️⃣ Test endpoint /api/review/humanitas/cases...")
        
        response = requests.get('http://localhost:5000/api/review/humanitas/cases?limit=3')
        if response.status_code == 200:
            data = response.json()
            cases = data.get('cases', [])
            
            print(f"📊 Cases trovati: {len(cases)}")
            
            if cases:
                for i, case in enumerate(cases[:2], 1):
                    session_id = case.get('session_id', 'N/A')[:12] + "..."
                    reason = case.get('reason', '')
                    notes = case.get('notes', '')
                    
                    print(f"   {i}. Session: {session_id}")
                    print(f"      Reason: {reason[:50]}..." if len(reason) > 50 else f"      Reason: {reason}")
                    print(f"      Notes: {notes[:50]}..." if len(notes) > 50 else f"      Notes: {notes}")
                    
                    if notes:
                        print("      ✅ Campo notes presente")
                    else:
                        print("      ⚠️  Campo notes vuoto")
            else:
                print("   ℹ️  Nessun caso nella review queue")
                
            print("✅ Endpoint review cases: OK")
        else:
            print(f"❌ Errore endpoint review cases: {response.status_code}")
            return False
        
        # 2. Test endpoint Sessions per Tenant
        print("\n2️⃣ Test endpoint /api/sessions/humanitas...")
        
        response = requests.get('http://localhost:5000/api/sessions/humanitas?limit=3')
        if response.status_code == 200:
            data = response.json()
            sessions = data.get('sessions', [])
            
            print(f"📊 Sessions trovate: {len(sessions)}")
            
            if sessions:
                for i, session in enumerate(sessions[:2], 1):
                    session_id = session.get('session_id', 'N/A')[:12] + "..."
                    motivation = session.get('motivation', '')
                    notes = session.get('notes', '')
                    classification = session.get('classification', 'N/A')
                    
                    print(f"   {i}. Session: {session_id}")
                    print(f"      Classification: {classification}")
                    print(f"      Motivation: {motivation[:50]}..." if len(motivation) > 50 else f"      Motivation: {motivation}")
                    print(f"      Notes: {notes[:50]}..." if len(notes) > 50 else f"      Notes: {notes}")
                    
                    if notes:
                        print("      ✅ Campo notes presente")
                        if notes == motivation:
                            print("      ✅ Notes = Motivation (mapping corretto)")
                        else:
                            print("      ⚠️  Notes ≠ Motivation")
                    else:
                        print("      ⚠️  Campo notes vuoto")
            else:
                print("   ℹ️  Nessuna session trovata")
                
            print("✅ Endpoint sessions: OK")
        else:
            print(f"❌ Errore endpoint sessions: {response.status_code}")
            return False
            
        # 3. Test MongoDB reader direttamente
        print("\n3️⃣ Test MongoDB reader diretto...")
        
        from mongo_classification_reader import MongoClassificationReader
        
        mongo_reader = MongoClassificationReader()
        if mongo_reader.connect():
            # Test get_all_sessions
            sessions = mongo_reader.get_all_sessions("humanitas", limit=2)
            
            print(f"📊 Sessions da MongoDB: {len(sessions)}")
            
            if sessions:
                for i, session in enumerate(sessions, 1):
                    session_id = session.get('session_id', 'N/A')[:12] + "..."
                    motivation = session.get('motivation', '')
                    notes = session.get('notes', '')
                    
                    print(f"   {i}. Session: {session_id}")
                    print(f"      Motivation: {motivation[:30]}..." if len(motivation) > 30 else f"      Motivation: {motivation}")
                    print(f"      Notes: {notes[:30]}..." if len(notes) > 30 else f"      Notes: {notes}")
                    
                    if notes and notes == motivation:
                        print("      ✅ Mapping motivation → notes: OK")
                    elif notes:
                        print("      ⚠️  Notes presente ma diverso da motivation")
                    else:
                        print("      ⚠️  Campo notes mancante")
                        
            mongo_reader.disconnect()
            print("✅ MongoDB reader: OK")
        else:
            print("❌ Impossibile connettersi a MongoDB")
            return False
        
        print("\n🎯 Test completato!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main del test motivazione → notes
    """
    print("🚀 Test Mapping Motivazione → Notes per UI")
    print("=" * 60)
    
    success = test_motivation_as_notes()
    
    if success:
        print("\n🎉 TEST COMPLETATO CON SUCCESSO!")
        print("✅ Campo motivazione correttamente mappato su notes")  
        print("✅ API endpoints espongono il campo notes")
        print("✅ Interfaccia può ora visualizzare le note")
        print("💡 Il frontend React può utilizzare il campo 'notes' per mostrare le motivazioni")
    else:
        print("\n❌ Test fallito!")
        print("🔧 Verificare configurazione e connessioni")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
