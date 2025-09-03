
import yaml
from pymongo import MongoClient

def analyze_mongo_data():
    """
    Si connette a MongoDB, legge i dati di classificazione e conta il numero di outliers,
    rappresentanti e propagati in base ai filtri del frontend.
    """
    try:
        # Carica la configurazione del database da config.yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mongo_url = config.get('mongodb', {}).get('url', 'mongodb://localhost:27017')
        db_name = config.get('mongodb', {}).get('database', 'classificazioni')
        
        print(f"Tentativo di connessione a MongoDB: {mongo_url}")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # Verifica la connessione
        client.server_info()
        print("Connessione a MongoDB riuscita.")
        
        db = client[db_name]
        
        # Il frontend sembra basarsi sulla collezione 'review_cases'
        # Se non esiste, proviamo con un nome comune come 'sessions'
        collection_name = 'review_cases'
        if collection_name not in db.list_collection_names():
            print(f"Collezione '{collection_name}' non trovata. Cerco alternative...")
            # Logica di fallback se necessario, per ora usiamo un nome fisso
            # Potrebbe essere 'sessions' o altro, a seconda della struttura del DB
            # Per ora, terminiamo se non troviamo la collezione attesa.
            # In un caso reale, potremmo ispezionare le collezioni disponibili.
            print("Collezioni disponibili:", db.list_collection_names())
            print(f"Impossibile trovare la collezione '{collection_name}'. Interruzione.")
            return

        collection = db[collection_name]
        print(f"Analisi della collezione: '{collection_name}' nel database '{db_name}'")

        # 1. Conteggio degli Outliers
        # Il frontend li filtra tramite `classification_type`
        outliers_count = collection.count_documents({'classification_type': 'OUTLIER'})
        
        # 2. Conteggio dei Rappresentanti
        # Il frontend li identifica con `classification_type` o `is_representative: true`
        # Usiamo il filtro più specifico per coerenza
        representatives_count = collection.count_documents({'classification_type': 'RAPPRESENTANTE'})
        
        # 3. Conteggio dei Propagati
        # Il frontend li identifica con `classification_type` o `is_representative: false`
        propagated_count = collection.count_documents({'classification_type': 'PROPAGATO'})

        print("\n--- Risultati dell'Analisi su MongoDB ---")
        print(f"📊 Totale documenti nella collezione '{collection_name}': {collection.estimated_document_count()}")
        print("-------------------------------------------")
        print(f"🟧 Outliers: {outliers_count}")
        print(f"👑 Rappresentanti: {representatives_count}")
        print(f"🔗 Propagati: {propagated_count}")
        print("-------------------------------------------\n")

        print("🔍 Dettaglio Query Eseguite:")
        print("   - Outliers: db.review_cases.count_documents({'classification_type': 'OUTLIER'})")
        print("   - Rappresentanti: db.review_cases.count_documents({'classification_type': 'RAPPRESENTANTE'})")
        print("   - Propagati: db.review_cases.count_documents({'classification_type': 'PROPAGATO'})\n")

    except yaml.YAMLError as e:
        print(f"Errore nella lettura del file config.yaml: {e}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    analyze_mongo_data()
