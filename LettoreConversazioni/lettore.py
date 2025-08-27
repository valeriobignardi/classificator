import sys
import os
import yaml

# Aggiunge il percorso della directory MySql al path per importare il connettore
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MySql'))

from connettore import MySqlConnettore

# Aggiunge il percorso per importare l'helper configurazioni tenant
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Utils'))
from tenant_config_helper import get_only_user_for_tenant

class LettoreConversazioni:
    """
    Classe per leggere le conversazioni dal database MySQL
    Supporta configurazioni per tenant, incluso il parametro only_user
    """
    
    def __init__(self, schema='common', config_path=None, tenant_id=None):
        """
        Inizializza il lettore delle conversazioni
        
        Args:
            schema (str): Nome dello schema del database (default: 'common')
            config_path (str): Percorso file configurazione (default: '../config.yaml')
            tenant_id (str): ID del tenant per parametri personalizzati (opzionale)
            
        Ultima modifica: 2025-08-26
        """
        self.connettore = MySqlConnettore()
        self.schema = schema
        self.tenant_id = tenant_id
        
        # Carica configurazione
        if not config_path:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ Errore caricamento config.yaml: {e}")
            self.config = {}
        
        # 🆕 NUOVA LOGICA: Usa helper tenant se tenant_id è fornito
        if tenant_id:
            try:
                self.only_user = get_only_user_for_tenant(tenant_id)
                print(f"🎯 [LETTORE] Tenant {tenant_id}: only_user = {self.only_user} (da config tenant)")
            except Exception as e:
                print(f"⚠️ [LETTORE] Errore config tenant {tenant_id}: {e}")
                self.only_user = False
                print(f"� [LETTORE] Fallback: only_user = False (default)")
        else:
            # 🔄 LOGICA LEGACY: Per retrocompatibilità, default a False
            self.only_user = False
            print(f"📖 [LETTORE] Schema {schema}: only_user = {self.only_user} (legacy - no tenant_id)")
    
    def leggi_conversazioni(self):
        """
        Legge i dati delle conversazioni dal database usando la query specificata
        
        Returns:
            list: Lista di tuple contenenti i dati delle conversazioni
                  Format: (session_id, conversation_agent_name, conversation_status_message_id, 
                          conversation_message, said_by, message_created_at)
        """
        # Determina il filtro said_by in base alla configurazione
        if self.only_user:
            said_by_filter = "WHERE csm.said_by = 'USER'"
            print("🎯 [DEBUG ONLY_USER] Filtraggio attivo: solo messaggi USER")
        else:
            said_by_filter = "WHERE csm.said_by IN ('USER', 'AGENT')"
            print("📄 [DEBUG ONLY_USER] Filtraggio standard: messaggi USER e AGENT")
            
        query = f"""
        SELECT cs.session_id
             , (SELECT a.agent_name
                  FROM `{self.schema}`.agents a
                 WHERE a.agent_id = cs.conversation_agent_id) AS conversation_agent_name
             , csm.conversation_status_message_id
             , csm.conversation_message
             , csm.said_by
             , csm.created_at AS message_created_at
          FROM `{self.schema}`.conversation_status cs
         INNER JOIN `{self.schema}`.conversation_status_messages csm
            ON cs.conversation_status_id = csm.conversation_status_id
         {said_by_filter}
         ORDER BY cs.session_id
             , cs.created_at ASC
             , csm.created_at ASC
             , csm.said_by ASC
        """
        
        try:
            print(f"🔍 [DEBUG ONLY_USER] Esecuzione query per leggere conversazioni dal schema '{self.schema}'...")
            print(f"🎯 [DEBUG ONLY_USER] Parametro only_user = {self.only_user}")
            print(f"📋 [DEBUG ONLY_USER] Filtro applicato: {said_by_filter}")
            
            risultati = self.connettore.esegui_query(query)
            
            if risultati is not None:
                # DEBUG: Analizza i risultati per verificare il filtro
                total_rows = len(risultati)
                user_messages = sum(1 for row in risultati if row[4] == 'USER')
                agent_messages = sum(1 for row in risultati if row[4] == 'AGENT')
                
                print(f"✅ [DEBUG ONLY_USER] Query eseguita con successo!")
                print(f"📊 [DEBUG ONLY_USER] Totale righe: {total_rows}")
                print(f"👤 [DEBUG ONLY_USER] Messaggi USER: {user_messages}")
                print(f"🤖 [DEBUG ONLY_USER] Messaggi AGENT: {agent_messages}")
                
                if self.only_user and agent_messages > 0:
                    print(f"⚠️ [DEBUG ONLY_USER] ERRORE: only_user=True ma trovati {agent_messages} messaggi AGENT!")
                elif not self.only_user and agent_messages == 0:
                    print(f"ℹ️ [DEBUG ONLY_USER] INFO: only_user=False ma non ci sono messaggi AGENT nel dataset")
                else:
                    print(f"✅ [DEBUG ONLY_USER] Filtro funziona correttamente!")
                
                return risultati
            else:
                print("❌ [DEBUG ONLY_USER] Errore durante l'esecuzione della query")
                return []
                
        except Exception as e:
            print(f"❌ Errore durante la lettura delle conversazioni: {e}")
            return []
    
    def leggi_conversazioni_per_sessione(self, session_id):
        """
        Legge i dati delle conversazioni per una specifica sessione
        
        Args:
            session_id (str): ID della sessione da cercare
            
        Returns:
            list: Lista di tuple contenenti i dati delle conversazioni per la sessione specificata
        """
        query = f"""
        SELECT cs.session_id
             , (SELECT a.agent_name
                  FROM `{self.schema}`.agents a
                 WHERE a.agent_id = cs.conversation_agent_id) AS conversation_agent_name
             , csm.conversation_status_message_id
             , csm.conversation_message
             , csm.said_by
             , csm.created_at AS message_created_at
          FROM `{self.schema}`.conversation_status cs
         INNER JOIN `{self.schema}`.conversation_status_messages csm
            ON cs.conversation_status_id = csm.conversation_status_id
         WHERE csm.said_by IN ('USER', 'AGENT')
           AND cs.session_id = %s
         ORDER BY cs.created_at ASC
             , csm.created_at ASC
             , csm.said_by ASC
        """
        
        try:
            print(f"🔍 Cercando conversazioni per sessione: {session_id}")
            risultati = self.connettore.esegui_query(query, (session_id,))
            
            if risultati is not None:
                print(f"✅ Trovate {len(risultati)} righe per la sessione {session_id}")
                return risultati
            else:
                print("❌ Errore durante l'esecuzione della query")
                return []
                
        except Exception as e:
            print(f"❌ Errore durante la lettura della sessione: {e}")
            return []
    
    def stampa_conversazioni(self, risultati):
        """
        Stampa le conversazioni in formato leggibile
        
        Args:
            risultati (list): Lista di tuple contenenti i dati delle conversazioni
        """
        if not risultati:
            print("📭 Nessuna conversazione trovata")
            return
        
        print(f"\n📋 CONVERSAZIONI TROVATE ({len(risultati)} messaggi):")
        print("=" * 80)
        
        current_session = None
        for riga in risultati:
            session_id, agent_name, message_id, message, said_by, created_at = riga
            
            # Stampa header della sessione se è cambiata
            if current_session != session_id:
                current_session = session_id
                print(f"\n🗣️  SESSIONE: {session_id}")
                print(f"🤖 AGENT: {agent_name or 'N/A'}")
                print("-" * 60)
            
            # Determina l'icona per il tipo di messaggio
            icon = "👤" if said_by == "USER" else "🤖"
            
            # Stampa il messaggio
            print(f"{icon} [{said_by}] ({created_at}):")
            print(f"   {message}")
            print()
    
    def chiudi_connessione(self):
        """
        Chiude la connessione al database
        """
        self.connettore.disconnetti()

    def estrai_conversazioni_periodo(self, tenant_slug: str, data_inizio, data_fine=None):
        """
        Estrae conversazioni per un tenant specifico in un periodo temporale.
        
        Args:
            tenant_slug: Il slug del tenant (es. 'humanitas')
            data_inizio: Data di inizio del periodo 
            data_fine: Data di fine del periodo (opzionale, default: ora)
            
        Returns:
            pandas.DataFrame: DataFrame con le conversazioni filtrate
        """
        import pandas as pd
        from datetime import datetime
        
        if data_fine is None:
            data_fine = datetime.now()
        
        # Converti il tenant_slug al database corrispondente
        tenant_db_map = {
            'humanitas': 'humanitasdb',
            'demo': 'ai',
            'boots': 'boots',
            'wopta': 'wopta',
            # Aggiungi altri mapping se necessario
        }
        
        db_name = tenant_db_map.get(tenant_slug, f'{tenant_slug}db')
        
        query = f"""
        SELECT cs.session_id
             , (SELECT a.agent_name
                  FROM {db_name}.agents a
                 WHERE a.agent_id = cs.conversation_agent_id) AS conversation_agent_name
             , csm.conversation_status_message_id
             , csm.conversation_message
             , csm.said_by
             , csm.created_at AS message_created_at
        FROM {db_name}.conversation_sessions cs
        INNER JOIN {db_name}.conversation_status_messages csm 
            ON cs.session_id = csm.conversation_session_id
        WHERE cs.created_at >= %s 
            AND cs.created_at <= %s
            AND csm.conversation_message IS NOT NULL
            AND csm.conversation_message != ''
        ORDER BY cs.session_id, csm.created_at
        """
        
        try:
            risultati = self.connettore.esegui_query(query, (data_inizio, data_fine))
            
            if risultati:
                df = pd.DataFrame(risultati, columns=[
                    'session_id', 'conversation_agent_name', 'conversation_status_message_id',
                    'conversation_message', 'said_by', 'message_created_at'
                ])
                print(f"Estratte {len(df)} conversazioni per tenant '{tenant_slug}' dal {data_inizio} al {data_fine}")
                return df
            else:
                print(f"Nessuna conversazione trovata per tenant '{tenant_slug}' nel periodo specificato")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Errore nell'estrazione delle conversazioni: {e}")
            return pd.DataFrame()

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea un'istanza del lettore per Humanitas
    lettore = LettoreConversazioni(schema='humanitas')
    
    try:
        # Legge tutte le conversazioni di Humanitas (limitiamo a 20 per il test)
        print("=== LETTURA CONVERSAZIONI HUMANITAS ===\n")
        
        # Prima modifichiamo temporaneamente la query per limitare i risultati
        original_query = lettore.leggi_conversazioni.__code__
        
        # Eseguiamo una query limitata per il test
        query_limitata = """
        SELECT cs.session_id
             , (SELECT a.agent_name
                  FROM humanitas.agents a
                 WHERE a.agent_id = cs.conversation_agent_id) AS conversation_agent_name
             , csm.conversation_status_message_id
             , csm.conversation_message
             , csm.said_by
             , csm.created_at AS message_created_at
          FROM humanitas.conversation_status cs
         INNER JOIN humanitas.conversation_status_messages csm
            ON cs.conversation_status_id = csm.conversation_status_id
         WHERE csm.said_by IN ('USER', 'AGENT')
         ORDER BY cs.session_id
             , cs.created_at ASC
             , csm.created_at ASC
             , csm.said_by ASC
         LIMIT 20
        """
        
        print("🔍 Esecuzione query limitata (prime 20 righe) per test...")
        risultati = lettore.connettore.esegui_query(query_limitata)
        
        if risultati:
            print(f"✅ Query eseguita! Trovate {len(risultati)} righe")
            # Stampa i risultati
            lettore.stampa_conversazioni(risultati)
        else:
            print("❌ Nessun risultato trovato")
        
    finally:
        # Chiude la connessione
        lettore.chiudi_connessione()