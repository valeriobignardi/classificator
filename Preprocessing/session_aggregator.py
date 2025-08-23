"""
Aggregatore delle sessioni - raggruppa i messaggi per sessione
"""

import sys
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import re

# Aggiunge il percorso per importare il lettore conversazioni
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'LettoreConversazioni'))
from lettore import LettoreConversazioni

class SessionAggregator:
    """
    Classe per aggregare i messaggi delle conversazioni per sessione
    """
    
    def __init__(self, schema: str = 'humanitas'):
        """
        Inizializza l'aggregatore
        
        Args:
            schema: Schema del database da utilizzare
        """
        self.schema = schema
        self.lettore = LettoreConversazioni(schema=schema)
        
    def estrai_sessioni_aggregate(self, limit: Optional[int] = None) -> Dict[str, Dict]:
        """
        Estrae e aggrega tutte le sessioni dal database
        
        Args:
            limit: Limite numero di righe da processare (None per tutte)
            
        Returns:
            Dizionario con session_id come chiave e dati aggregati come valore
        """
        print(f"📊 Estrazione sessioni aggregate dal schema '{self.schema}'...")
        
        # Legge tutti i messaggi dal database
        if limit:
            # SOLUZIONE MYSQL COMPATIBILE: Query in due passaggi per evitare subquery con LIMIT
            print(f"📋 Estrazione limitata: prime {limit} sessioni")
            
            # Passaggio 1: Ottieni i primi N session_id
            query_session_ids = f"""
            SELECT DISTINCT session_id 
            FROM {self.schema}.conversation_status 
            ORDER BY session_id 
            LIMIT {limit}
            """
            session_ids_result = self.lettore.connettore.esegui_query(query_session_ids)
            
            if not session_ids_result:
                print("❌ Nessuna sessione trovata")
                return {}
            
            # Estrai solo gli ID delle sessioni
            session_ids = [row[0] for row in session_ids_result]
            print(f"✅ Trovate {len(session_ids)} sessioni da analizzare")
            
            # Passaggio 2: Ottieni tutti i messaggi per quelle sessioni
            # Crea la lista di ID per la query IN
            session_ids_str = ','.join([f"'{sid}'" for sid in session_ids])
            
            query = f"""
            SELECT cs.session_id
                 , (SELECT a.agent_name
                      FROM {self.schema}.agents a
                     WHERE a.agent_id = cs.conversation_agent_id) AS conversation_agent_name
                 , csm.conversation_status_message_id
                 , csm.conversation_message
                 , csm.said_by
                 , csm.created_at AS message_created_at
              FROM {self.schema}.conversation_status cs
             INNER JOIN {self.schema}.conversation_status_messages csm
                ON cs.conversation_status_id = csm.conversation_status_id
             WHERE csm.said_by IN ('USER', 'AGENT')
               AND cs.session_id IN ({session_ids_str})
             ORDER BY cs.session_id, csm.created_at ASC
            """
            risultati = self.lettore.connettore.esegui_query(query)
        else:
            risultati = self.lettore.leggi_conversazioni()
        
        if not risultati:
            print("❌ Nessuna conversazione trovata")
            return {}
        
        print(f"✅ Trovate {len(risultati)} righe da aggregare")
        
        # Aggrega per sessione
        sessioni_aggregate = {}
        
        for riga in risultati:
            session_id, agent_name, message_id, message, said_by, created_at = riga
            
            # Inizializza la sessione se non esiste
            if session_id not in sessioni_aggregate:
                sessioni_aggregate[session_id] = {
                    'session_id': session_id,
                    'agent_name': agent_name,
                    'messaggi': [],
                    'testo_completo': '',
                    'primo_messaggio': created_at,
                    'ultimo_messaggio': created_at,
                    'num_messaggi_user': 0,
                    'num_messaggi_agent': 0,
                    'num_messaggi_totali': 0
                }
            
            # Aggiungi il messaggio
            sessioni_aggregate[session_id]['messaggi'].append({
                'message_id': message_id,
                'message': message,
                'said_by': said_by,
                'created_at': created_at
            })
            
            # Aggiorna statistiche
            if said_by == 'USER':
                sessioni_aggregate[session_id]['num_messaggi_user'] += 1
            elif said_by == 'AGENT':
                sessioni_aggregate[session_id]['num_messaggi_agent'] += 1
                
            sessioni_aggregate[session_id]['num_messaggi_totali'] += 1
            sessioni_aggregate[session_id]['ultimo_messaggio'] = created_at
        
        # Genera il testo completo per ogni sessione
        messaggi_saltati = 0
        sessioni_vuote = 0
        
        for session_id, dati in sessioni_aggregate.items():
            testo_parti = []
            
            # Ordina i messaggi per timestamp
            messaggi_ordinati = sorted(dati['messaggi'], key=lambda x: x['created_at'])
            
            # Conta i messaggi che salterei (per statistiche)
            messaggi_che_salto = min(2, len(messaggi_ordinati))
            messaggi_saltati += messaggi_che_salto
            
            if len(messaggi_ordinati) <= 2:
                # Se ci sono solo i messaggi di benvenuto, segna come sessione vuota
                dati['testo_completo'] = ""
                sessioni_vuote += 1
                continue
            
            # MODIFICA: Salta i primi 2 messaggi (benvenuto utente + assistente)
            messaggi_da_processare = messaggi_ordinati[2:]
            
            for msg in messaggi_da_processare:
                prefisso = "[UTENTE]" if msg['said_by'] == 'USER' else "[ASSISTENTE]"
                testo_parti.append(f"{prefisso} {msg['message']}")
            
            dati['testo_completo'] = " ".join(testo_parti)
        
        print(f"✅ Aggregate {len(sessioni_aggregate)} sessioni uniche")
        print(f"🔄 Saltati {messaggi_saltati} messaggi di benvenuto (primi 2 per sessione)")
        print(f"🔄 Sessioni vuote (solo benvenuto): {sessioni_vuote}")
        return sessioni_aggregate
    
    def filtra_sessioni_vuote(self, sessioni: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filtra le sessioni che contengono solo saluti base
        
        Args:
            sessioni: Dizionario delle sessioni aggregate
            
        Returns:
            Dizionario delle sessioni filtrate
        """
        print("🔍 Filtraggio sessioni vuote/irrilevanti...")
        
        sessioni_filtrate = {}
        sessioni_scartate = 0
        
        # Pattern per riconoscere conversazioni vuote/irrilevanti
        pattern_saluti = re.compile(
            r'^\s*(ciao|salve|buongiorno|buonasera|hello|hi)\s*$',
            re.IGNORECASE
        )
        
        pattern_welcome = re.compile(
            r'(ciao.*sono.*giulia|assistente.*digitale|come.*posso.*aiutarti)',
            re.IGNORECASE
        )
        
        for session_id, dati in sessioni.items():
            # Criteri per scartare la sessione
            scarta = False
            
            # Sessioni con troppo pochi messaggi
            if dati['num_messaggi_totali'] < 3:
                # Verifica se è solo saluto + welcome
                messaggi_user = [msg['message'] for msg in dati['messaggi'] if msg['said_by'] == 'USER']
                messaggi_agent = [msg['message'] for msg in dati['messaggi'] if msg['said_by'] == 'AGENT']
                
                if (len(messaggi_user) == 1 and 
                    pattern_saluti.match(messaggi_user[0]) and
                    len(messaggi_agent) <= 1 and
                    any(pattern_welcome.search(msg) for msg in messaggi_agent)):
                    scarta = True
            
            # Sessioni con solo messaggi molto corti
            lunghezza_media = sum(len(msg['message']) for msg in dati['messaggi']) / len(dati['messaggi'])
            if lunghezza_media < 10 and dati['num_messaggi_totali'] < 5:
                scarta = True
            
            if scarta:
                sessioni_scartate += 1
            else:
                sessioni_filtrate[session_id] = dati
        
        print(f"✅ Filtro completato:")
        print(f"  📊 Sessioni originali: {len(sessioni)}")
        print(f"  ❌ Sessioni scartate: {sessioni_scartate}")
        print(f"  ✅ Sessioni valide: {len(sessioni_filtrate)}")
        
        return sessioni_filtrate
    
    def get_statistiche_sessioni(self, sessioni: Dict[str, Dict]) -> Dict:
        """
        Calcola statistiche sulle sessioni aggregate
        
        Args:
            sessioni: Dizionario delle sessioni
            
        Returns:
            Dizionario con statistiche
        """
        if not sessioni:
            return {}
        
        num_messaggi = [s['num_messaggi_totali'] for s in sessioni.values()]
        lunghezze_testo = [len(s['testo_completo']) for s in sessioni.values()]
        num_user_msgs = [s['num_messaggi_user'] for s in sessioni.values()]
        num_agent_msgs = [s['num_messaggi_agent'] for s in sessioni.values()]
        
        stats = {
            'num_sessioni_totali': len(sessioni),
            'messaggi_per_sessione': {
                'media': sum(num_messaggi) / len(num_messaggi),
                'min': min(num_messaggi),
                'max': max(num_messaggi),
                'mediana': sorted(num_messaggi)[len(num_messaggi)//2]
            },
            'lunghezza_testo': {
                'media': sum(lunghezze_testo) / len(lunghezze_testo),
                'min': min(lunghezze_testo),
                'max': max(lunghezze_testo),
                'mediana': sorted(lunghezze_testo)[len(lunghezze_testo)//2]
            },
            'messaggi_utente': {
                'media': sum(num_user_msgs) / len(num_user_msgs),
                'totale': sum(num_user_msgs)
            },
            'messaggi_agent': {
                'media': sum(num_agent_msgs) / len(num_agent_msgs),
                'totale': sum(num_agent_msgs)
            }
        }
        
        return stats
    
    def stampa_statistiche(self, stats: Dict):
        """Stampa le statistiche in formato leggibile"""
        if not stats:
            print("📊 Nessuna statistica disponibile")
            return
        
        print(f"\n📊 STATISTICHE SESSIONI:")
        print(f"📈 Numero totale sessioni: {stats['num_sessioni_totali']}")
        print(f"\n💬 Messaggi per sessione:")
        print(f"  Media: {stats['messaggi_per_sessione']['media']:.1f}")
        print(f"  Range: {stats['messaggi_per_sessione']['min']}-{stats['messaggi_per_sessione']['max']}")
        print(f"  Mediana: {stats['messaggi_per_sessione']['mediana']}")
        print(f"\n📝 Lunghezza testo:")
        print(f"  Media: {stats['lunghezza_testo']['media']:.0f} caratteri")
        print(f"  Range: {stats['lunghezza_testo']['min']}-{stats['lunghezza_testo']['max']} caratteri")
        print(f"\n👤 Messaggi utente totali: {stats['messaggi_utente']['totale']}")
        print(f"🤖 Messaggi agent totali: {stats['messaggi_agent']['totale']}")
    
    def chiudi_connessione(self):
        """Chiude la connessione al database"""
        self.lettore.chiudi_connessione()

# Test della classe
if __name__ == "__main__":
    aggregator = SessionAggregator(schema='humanitas')
    
    try:
        # Test con limite per velocità
        print("=== TEST SESSION AGGREGATOR ===\n")
        
        # Estrai e aggrega (limitato per test)
        sessioni = aggregator.estrai_sessioni_aggregate(limit=100)
        
        if sessioni:
            # Filtra sessioni vuote
            sessioni_filtrate = aggregator.filtra_sessioni_vuote(sessioni)
            
            # Statistiche
            stats = aggregator.get_statistiche_sessioni(sessioni_filtrate)
            aggregator.stampa_statistiche(stats)
            
            # Mostra alcune sessioni di esempio
            print(f"\n🔍 ESEMPI DI SESSIONI AGGREGATE:")
            print("=" * 80)
            
            for i, (session_id, dati) in enumerate(list(sessioni_filtrate.items())[:3]):
                print(f"\n📱 Sessione {i+1}: {session_id}")
                print(f"🤖 Agent: {dati['agent_name']}")
                print(f"💬 Messaggi: {dati['num_messaggi_totali']} ({dati['num_messaggi_user']} user, {dati['num_messaggi_agent']} agent)")
                print(f"📝 Testo ({len(dati['testo_completo'])} caratteri):")
                print(f"   {dati['testo_completo'][:200]}{'...' if len(dati['testo_completo']) > 200 else ''}")
                print("-" * 60)
        
    finally:
        aggregator.chiudi_connessione()
