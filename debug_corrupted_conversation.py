#!/usr/bin/env python3
"""
Script di debug per analizzare conversazioni corrotte nel database
Author: Assistant
Created: 2025-08-25
"""

import sys
import os
import re
import base64

sys.path.append(os.path.join(os.path.dirname(__file__), 'LettoreConversazioni'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'MySql'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Preprocessing'))

from lettore import LettoreConversazioni
from connettore import MySqlConnettore
from session_aggregator import SessionAggregator


def detect_binary_corruption(text):
    """
    Rileva se un testo contiene dati binari corrotti
    
    Args:
        text (str): Testo da analizzare
        
    Returns:
        dict: Informazioni sulla corruzione rilevata
    """
    corruption_info = {
        'is_corrupted': False,
        'type': 'normal',
        'base64_like': False,
        'binary_chars': 0,
        'total_chars': len(text),
        'corruption_percentage': 0.0,
        'sample_chars': text[:100] if text else ''
    }
    
    if not text:
        return corruption_info
        
    # Conta caratteri sospetti (non stampabili, sequenze binarie)
    binary_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]')
    binary_matches = binary_pattern.findall(text)
    corruption_info['binary_chars'] = len(binary_matches)
    
    # Rileva pattern tipici di base64 corrotto
    base64_pattern = re.compile(r'^[A-Za-z0-9+/=]{20,}')
    if base64_pattern.match(text.replace('/', '').replace('=', '').replace('+', '')):
        corruption_info['base64_like'] = True
        corruption_info['type'] = 'base64_corruption'
    
    # Rileva sequenze di caratteri speciali ripetuti
    special_pattern = re.compile(r'[/+=-]{5,}')
    if special_pattern.search(text):
        corruption_info['type'] = 'special_chars_corruption'
        
    # Rileva sequenze di A ripetute (tipico di corruzione dati)
    a_pattern = re.compile(r'A{10,}')
    if a_pattern.search(text):
        corruption_info['type'] = 'repeated_a_corruption'
    
    # Calcola percentuale di corruzione
    if corruption_info['total_chars'] > 0:
        corruption_info['corruption_percentage'] = (corruption_info['binary_chars'] / corruption_info['total_chars']) * 100
    
    # Determina se è corrotto
    corruption_info['is_corrupted'] = (
        corruption_info['binary_chars'] > 10 or 
        corruption_info['corruption_percentage'] > 5 or
        corruption_info['base64_like'] or
        corruption_info['type'] != 'normal'
    )
    
    return corruption_info


def analyze_wopta_conversations():
    """
    Analizza le conversazioni di Wopta per trovare dati corrotti
    """
    print("🔍 ANALISI CONVERSAZIONI CORROTTE - WOPTA")
    print("="*80)
    
    # Inizializza aggregator per Wopta
    aggregator = SessionAggregator()
    
    try:
        print("📊 Estrazione sessioni aggregate da Wopta...")
        sessioni = aggregator.estrai_sessioni_aggregate('wopta', giorni_indietro=30, limit=None)
        
        if not sessioni:
            print("❌ Nessuna sessione trovata")
            return
            
        print(f"✅ Trovate {len(sessioni)} sessioni da analizzare")
        
        corrupted_sessions = []
        total_corrupted_chars = 0
        
        for i, sessione in enumerate(sessioni):
            session_id = sessione.get('session_id')
            testo_completo = sessione.get('testo_completo', '')
            
            # Analizza corruzione
            corruption_info = detect_binary_corruption(testo_completo)
            
            if corruption_info['is_corrupted']:
                corrupted_sessions.append({
                    'session_id': session_id,
                    'corruption_info': corruption_info,
                    'text_length': len(testo_completo),
                    'text_sample': testo_completo[:200]
                })
                total_corrupted_chars += len(testo_completo)
                
                print(f"\n🚨 SESSIONE CORROTTA TROVATA #{len(corrupted_sessions)}")
                print(f"   Session ID: {session_id}")
                print(f"   Lunghezza: {len(testo_completo):,} caratteri")
                print(f"   Tipo corruzione: {corruption_info['type']}")
                print(f"   Caratteri binari: {corruption_info['binary_chars']}")
                print(f"   Corruzione %: {corruption_info['corruption_percentage']:.2f}%")
                print(f"   Sample: {testo_completo[:100]}...")
                print(f"   Sample finale: ...{testo_completo[-100:]}")
        
        # Stampa sommario
        print(f"\n📊 SOMMARIO ANALISI CORRUZIONE")
        print("="*50)
        print(f"   Sessioni totali analizzate: {len(sessioni)}")
        print(f"   Sessioni corrotte: {len(corrupted_sessions)}")
        print(f"   Caratteri corrotti totali: {total_corrupted_chars:,}")
        print(f"   Percentuale sessioni corrotte: {(len(corrupted_sessions)/len(sessioni)*100):.2f}%")
        
        if corrupted_sessions:
            print(f"\n🔧 DETTAGLI SESSIONI PIÙ CORROTTE:")
            sorted_corrupted = sorted(corrupted_sessions, key=lambda x: x['text_length'], reverse=True)
            
            for i, session in enumerate(sorted_corrupted[:5]):  # Top 5 più lunghe
                print(f"\n   [{i+1}] Session: {session['session_id']}")
                print(f"       Lunghezza: {session['text_length']:,} caratteri")
                print(f"       Tipo: {session['corruption_info']['type']}")
                print(f"       Sample: {session['text_sample'][:100]}...")
        
        return corrupted_sessions
        
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()
        return []


def deep_database_analysis():
    """
    Analisi profonda del database per identificare la fonte della corruzione
    """
    print("\n🔬 ANALISI PROFONDA DATABASE")
    print("="*80)
    
    connettore = MySqlConnettore()
    
    try:
        # Query per cercare messaggi sospetti nel database
        query_sospetti = """
        SELECT cs.session_id,
               csm.conversation_status_message_id,
               csm.conversation_message,
               csm.said_by,
               csm.created_at,
               LENGTH(csm.conversation_message) as message_length
        FROM wopta.conversation_status cs
        INNER JOIN wopta.conversation_status_messages csm 
            ON cs.conversation_status_id = csm.conversation_status_id
        WHERE LENGTH(csm.conversation_message) > 50000
           OR csm.conversation_message LIKE '%AAAAAAEA%'
           OR csm.conversation_message LIKE '%/%/%/%'
           OR csm.conversation_message REGEXP '[/+=]{10,}'
        ORDER BY LENGTH(csm.conversation_message) DESC
        LIMIT 10
        """
        
        print("🔍 Cercando messaggi sospetti nel database...")
        risultati = connettore.esegui_query(query_sospetti)
        
        if risultati:
            print(f"✅ Trovati {len(risultati)} messaggi sospetti")
            
            for i, riga in enumerate(risultati):
                session_id, message_id, message, said_by, created_at, message_length = riga
                
                print(f"\n🚨 MESSAGGIO SOSPETTO #{i+1}")
                print(f"   Session ID: {session_id}")
                print(f"   Message ID: {message_id}")
                print(f"   Said by: {said_by}")
                print(f"   Created: {created_at}")
                print(f"   Length: {message_length:,} caratteri")
                
                # Analizza il tipo di corruzione
                corruption_info = detect_binary_corruption(message)
                print(f"   Corruption type: {corruption_info['type']}")
                print(f"   Corruption %: {corruption_info['corruption_percentage']:.2f}%")
                
                # Mostra sample iniziale e finale
                print(f"   Inizio: {message[:100]}")
                print(f"   Fine: {message[-100:]}")
                
        else:
            print("✅ Nessun messaggio sospetto trovato nel database")
            
    except Exception as e:
        print(f"❌ Errore durante l'analisi database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connettore.disconnetti()


if __name__ == "__main__":
    print("🚀 AVVIO DEBUG CONVERSAZIONI CORROTTE")
    print("="*80)
    
    # Analisi sessioni aggregate
    corrupted_sessions = analyze_wopta_conversations()
    
    # Analisi profonda database
    deep_database_analysis()
    
    print("\n✅ ANALISI COMPLETATA")
