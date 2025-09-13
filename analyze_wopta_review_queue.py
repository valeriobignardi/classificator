#!/usr/bin/env python3
"""
Autore: Valerio Bignardi
Data: 2025-09-12
Descrizione: Analizza perché la review queue di wopta risulta vuota dopo training supervisionato

Scopo della funzione: Ispezionare MongoDB collection wopta per identificare problemi
Parametri di input: tenant_id wopta, criteri filtro review queue
Parametri di output: Report dettagliato con analisi discrepanze
Valori di ritorno: Statistiche complete e cause del problema
Tracciamento aggiornamenti: 2025-09-12 - Creazione iniziale
"""

import sys
import os
import yaml
from pymongo import MongoClient
from datetime import datetime
import json

# Aggiungi percorsi necessari
sys.path.append(os.path.join(os.path.dirname(__file__), 'Utils'))
from tenant import Tenant

def analyze_wopta_mongodb_collection():
    """
    Analizza la collection MongoDB wopta per identificare problemi review queue
    
    Scopo: Verificare perché la review queue risulta vuota
    Input: Collection wopta_16c222a9-f293-11ef-9315-96000228e7fe
    Output: Report analisi completa
    """
    print("=" * 80)
    print("🔍 ANALISI MONGODB COLLECTION WOPTA - REVIEW QUEUE DEBUGGING")
    print("=" * 80)
    
    try:
        # 1. CONNESSIONE MONGODB
        print("\n📊 STEP 1: Connessione MongoDB")
        print("-" * 40)
        
        mongodb_url = "mongodb://localhost:27017"
        database_name = "classificazioni"
        collection_name = "wopta_16c222a9-f293-11ef-9315-96000228e7fe"
        
        client = MongoClient(mongodb_url)
        db = client[database_name]
        collection = db[collection_name]
        
        print(f"✅ Connesso a MongoDB: {mongodb_url}")
        print(f"📂 Database: {database_name}")
        print(f"📊 Collection: {collection_name}")
        
        # 2. STATISTICHE GENERALI COLLECTION
        print("\n📊 STEP 2: Statistiche Generali Collection")
        print("-" * 40)
        
        total_docs = collection.count_documents({})
        print(f"📋 Totale documenti: {total_docs}")
        
        if total_docs == 0:
            print("❌ PROBLEMA: Collection completamente vuota!")
            return
        
        # 3. ANALISI REVIEW_STATUS
        print("\n📊 STEP 3: Analisi Review Status")
        print("-" * 40)
        
        # Pipeline per analizzare review_status
        review_status_pipeline = [
            {"$group": {
                "_id": "$review_status",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        review_status_results = list(collection.aggregate(review_status_pipeline))
        
        print("📊 Distribuzione Review Status:")
        for status in review_status_results:
            status_name = status['_id'] or 'NULL/UNDEFINED'
            count = status['count']
            percentage = (count / total_docs) * 100
            print(f"   - {status_name}: {count} ({percentage:.1f}%)")
        
        # Count specifici
        pending_count = collection.count_documents({"review_status": "pending"})
        auto_classified_count = collection.count_documents({"review_status": "auto_classified"})
        completed_count = collection.count_documents({"review_status": "completed"})
        null_review_status = collection.count_documents({"review_status": None})
        
        print(f"\n🔍 Dettaglio Review Status:")
        print(f"   📌 Pending (da rivedere): {pending_count}")
        print(f"   🤖 Auto classified: {auto_classified_count}")
        print(f"   ✅ Completed: {completed_count}")
        print(f"   ❓ NULL/Undefined: {null_review_status}")
        
        # 4. ANALISI METADATI CLUSTER
        print("\n📊 STEP 4: Analisi Metadati Cluster")
        print("-" * 40)
        
        # Verifica presenza metadata
        docs_with_metadata = collection.count_documents({"metadata": {"$exists": True, "$ne": None}})
        docs_without_metadata = total_docs - docs_with_metadata
        
        print(f"📊 Presenza Metadata:")
        print(f"   ✅ Con metadata: {docs_with_metadata}")
        print(f"   ❌ Senza metadata: {docs_without_metadata}")
        
        # Analisi rappresentanti
        representative_docs = collection.count_documents({"metadata.representative": True})
        non_representative_docs = collection.count_documents({"metadata.representative": False})
        is_representative_legacy = collection.count_documents({"metadata.is_representative": True})
        
        print(f"📊 Rappresentanti Cluster:")
        print(f"   👑 metadata.representative=True: {representative_docs}")
        print(f"   👤 metadata.representative=False: {non_representative_docs}")
        print(f"   📜 metadata.is_representative=True (legacy): {is_representative_legacy}")
        
        # Analisi outlier
        outlier_docs = collection.count_documents({"metadata.outlier": True})
        non_outlier_docs = collection.count_documents({"metadata.outlier": False})
        cluster_id_minus_one = collection.count_documents({"metadata.cluster_id": -1})
        
        print(f"📊 Outliers:")
        print(f"   🔍 metadata.outlier=True: {outlier_docs}")
        print(f"   🎯 metadata.outlier=False: {non_outlier_docs}")
        print(f"   📊 metadata.cluster_id=-1: {cluster_id_minus_one}")
        
        # Analisi propagated
        propagated_docs = collection.count_documents({"metadata.propagated": True})
        non_propagated_docs = collection.count_documents({"metadata.propagated": False})
        
        print(f"📊 Propagated:")
        print(f"   🔄 metadata.propagated=True: {propagated_docs}")
        print(f"   🎯 metadata.propagated=False: {non_propagated_docs}")
        
        # 5. SIMULAZIONE FILTRI REVIEW QUEUE
        print("\n📊 STEP 5: Simulazione Filtri Review Queue")
        print("-" * 40)
        
        # Simula il filtro della funzione get_review_queue_sessions
        review_queue_base_query = {"review_status": "pending"}
        
        # Test con tutti i filtri attivi (default UI)
        show_representatives = True
        show_propagated = True  
        show_outliers = True
        
        # Trova documenti che soddisfano criteri review queue
        matching_docs = []
        
        # Effettua la query base
        base_docs = list(collection.find(review_queue_base_query))
        print(f"🔍 Documenti con review_status='pending': {len(base_docs)}")
        
        if len(base_docs) == 0:
            print("❌ PROBLEMA CRITICO: Nessun documento con review_status='pending'!")
            print("💡 La review queue è vuota perché nessun documento è marcato per revisione!")
        
        # Analizza ogni documento pending per capire i tipi
        for doc in base_docs:
            metadata = doc.get('metadata', {})
            
            # Logica di categorizzazione (stessa del codice review queue)
            is_representative = metadata.get('representative', False)
            is_propagated = metadata.get('propagated', False)
            is_outlier = metadata.get('outlier', False)
            
            # Fallback per compatibilità con dati legacy
            if not (is_representative or is_propagated or is_outlier):
                if metadata.get('is_representative', False):
                    is_representative = True
                elif metadata.get('propagated_from'):
                    is_propagated = True
                elif metadata.get('cluster_id') in [-1, "-1"] or not metadata.get('cluster_id'):
                    is_outlier = True
            
            # Verifica se sarebbe incluso nei filtri
            include_document = (
                (show_representatives and is_representative) or
                (show_propagated and is_propagated) or 
                (show_outliers and is_outlier)
            )
            
            doc_info = {
                'session_id': doc.get('session_id', 'N/A'),
                'is_representative': is_representative,
                'is_propagated': is_propagated,
                'is_outlier': is_outlier,
                'include_document': include_document,
                'metadata': metadata
            }
            
            matching_docs.append(doc_info)
        
        # Report dei filtri
        included_count = sum(1 for doc in matching_docs if doc['include_document'])
        excluded_count = len(matching_docs) - included_count
        
        print(f"📊 Risultato Simulazione Filtri:")
        print(f"   ✅ Documenti inclusi: {included_count}")
        print(f"   ❌ Documenti esclusi: {excluded_count}")
        
        # Dettaglio dei documenti esclusi
        if excluded_count > 0:
            print(f"\n🔍 Analisi Documenti Esclusi:")
            for doc in matching_docs:
                if not doc['include_document']:
                    print(f"   📋 Session: {doc['session_id']}")
                    print(f"      - representative: {doc['is_representative']}")
                    print(f"      - propagated: {doc['is_propagated']}")
                    print(f"      - outlier: {doc['is_outlier']}")
                    print(f"      - metadata keys: {list(doc['metadata'].keys())}")
        
        # 6. ANALISI DETTAGLIATA CONFIDENCE VALUES
        print("\n📊 STEP 6: Analisi Dettagliata Confidence Values")
        print("-" * 40)
        
        # Analizza TUTTI i documenti per trovare confidence sottosoglia
        print("🔍 Analizzando TUTTI i documenti nella collection per confidence values...")
        
        confidence_values = []
        consensus_values = []
        disagreement_values = []
        low_confidence_docs = []
        doc_count = 0
        
        for doc in collection.find({}):
            doc_count += 1
            session_id = doc.get('session_id', 'N/A')
            classification = doc.get('classification', 'N/A')
            confidence = doc.get('confidence', None)
            review_status = doc.get('review_status', None)
            metadata = doc.get('metadata', {})
            
            # Analizza confidence se presente
            if confidence is not None:
                confidence_values.append(confidence)
                
                # Salva documenti con confidence sotto varie soglie
                if confidence < 0.95:
                    low_confidence_docs.append({
                        'session_id': session_id,
                        'confidence': confidence,
                        'classification': classification,
                        'review_status': review_status,
                        'is_representative': metadata.get('representative', metadata.get('is_representative', False)),
                        'is_outlier': metadata.get('outlier', False),
                        'is_propagated': metadata.get('propagated', False),
                        'cluster_id': metadata.get('cluster_id', 'N/A')
                    })
            
            # Analizza altri valori se presenti
            consensus = doc.get('consensus', None)
            if consensus is not None:
                consensus_values.append(consensus)
                
            disagreement = doc.get('ensemble_disagreement', None)
            if disagreement is not None:
                disagreement_values.append(disagreement)
            
            if doc_count % 500 == 0:
                print(f"  Processati {doc_count} documenti...")
        
        print(f"\n📈 STATISTICHE COMPLETE SU {doc_count} DOCUMENTI:")
        
        if confidence_values:
            print(f"\n📊 CONFIDENCE VALUES ({len(confidence_values)} documenti con confidence):")
            print(f"  Min: {min(confidence_values):.4f}")
            print(f"  Max: {max(confidence_values):.4f}")
            print(f"  Media: {sum(confidence_values)/len(confidence_values):.4f}")
            print(f"  Mediana: {sorted(confidence_values)[len(confidence_values)//2]:.4f}")
            
            # Conteggi per diverse soglie
            sotto_095 = len([c for c in confidence_values if c < 0.95])
            sotto_090 = len([c for c in confidence_values if c < 0.90])
            sotto_085 = len([c for c in confidence_values if c < 0.85])
            sotto_080 = len([c for c in confidence_values if c < 0.80])
            sotto_070 = len([c for c in confidence_values if c < 0.70])
            sotto_060 = len([c for c in confidence_values if c < 0.60])
            sotto_050 = len([c for c in confidence_values if c < 0.50])
            
            print(f"\n🎯 DISTRIBUZIONE CONFIDENCE (QUESTI SONO I CANDIDATI PER REVIEW!):")
            print(f"  Sotto 0.95: {sotto_095}/{len(confidence_values)} ({sotto_095/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.90: {sotto_090}/{len(confidence_values)} ({sotto_090/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.85: {sotto_085}/{len(confidence_values)} ({sotto_085/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.80: {sotto_080}/{len(confidence_values)} ({sotto_080/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.70: {sotto_070}/{len(confidence_values)} ({sotto_070/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.60: {sotto_060}/{len(confidence_values)} ({sotto_060/len(confidence_values)*100:.1f}%)")
            print(f"  Sotto 0.50: {sotto_050}/{len(confidence_values)} ({sotto_050/len(confidence_values)*100:.1f}%)")
            
            print(f"\n🔍 PRIMI 15 ESEMPI DI CONFIDENCE SOTTO 0.95:")
            for i, doc in enumerate(low_confidence_docs[:15]):
                status_icon = "⏳" if doc['review_status'] == 'pending' else "✅" if doc['review_status'] == 'completed' else "🤖"
                rep_icon = "👑" if doc['is_representative'] else "👤"
                print(f"  [{i+1:2d}] {status_icon} Session: {doc['session_id'][:8]}... | "
                      f"Conf: {doc['confidence']:.4f} | "
                      f"{rep_icon} Rep: {doc['is_representative']} | "
                      f"Status: {doc['review_status']} | "
                      f"Cluster: {doc['cluster_id']}")
            
            # Analisi rappresentanti con confidence bassa
            low_conf_representatives = [doc for doc in low_confidence_docs if doc['is_representative']]
            print(f"\n👑 RAPPRESENTANTI CON CONFIDENCE SOTTO 0.95: {len(low_conf_representatives)}")
            
            if low_conf_representatives:
                print(f"  📊 Status distribution dei rappresentanti con confidence bassa:")
                pending_reps = len([d for d in low_conf_representatives if d['review_status'] == 'pending'])
                completed_reps = len([d for d in low_conf_representatives if d['review_status'] == 'completed'])
                auto_reps = len([d for d in low_conf_representatives if d['review_status'] == 'auto_classified'])
                other_reps = len(low_conf_representatives) - pending_reps - completed_reps - auto_reps
                
                print(f"    ⏳ Pending: {pending_reps}")
                print(f"    ✅ Completed: {completed_reps}")
                print(f"    🤖 Auto-classified: {auto_reps}")
                print(f"    ❓ Altri: {other_reps}")
                
                print(f"\n  📋 Primi 10 rappresentanti con confidence bassa:")
                for i, doc in enumerate(low_conf_representatives[:10]):
                    status_icon = "⏳" if doc['review_status'] == 'pending' else "✅" if doc['review_status'] == 'completed' else "🤖"
                    print(f"    [{i+1:2d}] {status_icon} Session: {doc['session_id'][:8]}... | "
                          f"Conf: {doc['confidence']:.4f} | "
                          f"Status: {doc['review_status']} | "
                          f"Label: {doc['classification'][:30]}")
        
        if consensus_values:
            print(f"\n📊 CONSENSUS VALUES:")
            print(f"  Min: {min(consensus_values):.4f}")
            print(f"  Max: {max(consensus_values):.4f}")
            print(f"  Media: {sum(consensus_values)/len(consensus_values):.4f}")
        
        if disagreement_values:
            print(f"\n📊 ENSEMBLE DISAGREEMENT VALUES:")
            print(f"  Min: {min(disagreement_values):.4f}")
            print(f"  Max: {max(disagreement_values):.4f}")
            print(f"  Media: {sum(disagreement_values)/len(disagreement_values):.4f}")
        
        # 7. ESEMPI DI DOCUMENTI
        print("\n📊 STEP 7: Esempi di Documenti")
        print("-" * 40)
        
        # Mostra esempi di documenti per ogni categoria
        sample_limit = 2
        
        # Esempio documento pending
        pending_sample = collection.find_one({"review_status": "pending"})
        if pending_sample:
            print(f"📝 Esempio Documento PENDING:")
            print(f"   Session ID: {pending_sample.get('session_id', 'N/A')}")
            print(f"   Classification: {pending_sample.get('classification', 'N/A')}")
            print(f"   Confidence: {pending_sample.get('confidence', 'N/A')}")
            print(f"   Review Status: {pending_sample.get('review_status', 'N/A')}")
            print(f"   Metadata: {json.dumps(pending_sample.get('metadata', {}), indent=2, default=str)}")
        
        # 8. VERIFICA CONFIGURAZIONE TENANT
        print("\n📊 STEP 8: Verifica Configurazione Tenant")
        print("-" * 40)
        
        try:
            # Carica configurazione database TAG
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            import mysql.connector
            tag_db_config = config.get('tag_database', {})
            
            # Connessione MySQL TAG
            mysql_connection = mysql.connector.connect(
                host=tag_db_config['host'],
                port=tag_db_config['port'],
                user=tag_db_config['user'],
                password=tag_db_config['password'],
                database=tag_db_config['database']
            )
            
            cursor = mysql_connection.cursor()
            
            # Cerca tenant wopta
            cursor.execute("""
                SELECT tenant_id, tenant_name, tenant_slug, is_active
                FROM tenants 
                WHERE tenant_id = %s OR tenant_slug LIKE %s
            """, ('16c222a9-f293-11ef-9315-96000228e7fe', '%wopta%'))
            
            tenant_results = cursor.fetchall()
            
            print(f"🔍 Configurazione Tenant wopta:")
            if tenant_results:
                for tenant_id, tenant_name, tenant_slug, is_active in tenant_results:
                    print(f"   📋 Tenant ID: {tenant_id}")
                    print(f"   📋 Nome: {tenant_name}")
                    print(f"   📋 Slug: {tenant_slug}")
                    print(f"   📋 Attivo: {is_active}")
            else:
                print("   ❌ Nessun tenant wopta trovato nel database TAG!")
            
            cursor.close()
            mysql_connection.close()
            
        except Exception as e:
            print(f"   ❌ Errore verifica tenant: {e}")
        
        # 9. DIAGNOSI E RACCOMANDAZIONI
        print("\n📊 STEP 9: Diagnosi e Raccomandazioni")
        print("-" * 40)
        
        print(f"🎯 DIAGNOSI:")
        
        if pending_count == 0:
            print(f"❌ PROBLEMA PRINCIPALE: Nessun documento ha review_status='pending'")
            print(f"💡 CAUSA: Il training supervisionato non ha marcato documenti per revisione")
            print(f"💡 SOLUZIONE: Verificare configurazione soglie nel training supervisionato")
        elif included_count == 0 and pending_count > 0:
            print(f"❌ PROBLEMA PRINCIPALE: Documenti pending esistono ma non passano i filtri")
            print(f"💡 CAUSA: Metadati cluster non correttamente impostati")
            print(f"💡 SOLUZIONE: Verificare logica di assegnazione metadati nel clustering")
        elif included_count > 0:
            print(f"✅ Review queue dovrebbe funzionare correttamente")
            print(f"💡 Possibile problema nel frontend o nella chiamata API")
        
        print(f"\n🔧 RACCOMANDAZIONI:")
        print(f"1. Verificare parametri soglie training supervisionato")
        print(f"2. Controllare logica assegnazione review_status='pending'")
        print(f"3. Verificare metadati cluster (representative/propagated/outlier)")
        print(f"4. Testare API review queue con filtri diversi")
        print(f"5. Controllare log del training supervisionato per errori")
        
        print("\n" + "=" * 80)
        print("✅ ANALISI COMPLETATA")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    analyze_wopta_mongodb_collection()
