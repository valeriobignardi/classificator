#!/usr/bin/env python3
"""
============================================================================
Analisi MongoDB - Caso Specifico e Review Queue
============================================================================

Autore: Valerio Bignardi
Data creazione: 2025-01-31
Ultima modifica: 2025-01-31

Descrizione:
    Script per analizzare caso specifico 68bcec2b34ea2bca071d45be in MongoDB
    e investigare il problema delle classificazioni N/A dopo training supervisionato.

============================================================================
"""

import os
import yaml
import pymongo
from pymongo import MongoClient
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from collections import Counter

def load_config() -> Dict[str, Any]:
    """
    Carica configurazione da config.yaml
    
    Returns:
        Dizionario con configurazione
        
    Data ultima modifica: 2025-01-31
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    return load_config()

def connect_mongodb(config: Dict[str, Any]) -> MongoClient:
    """
    Connette a MongoDB usando configurazione
    
    Args:
        config: Configurazione completa
        
    Returns:
        Client MongoDB connesso
        
    Data ultima modifica: 2025-01-31
    """
    mongodb_config = config.get('mongodb', {})
    url = mongodb_config.get('url', 'mongodb://localhost:27017')
    database_name = mongodb_config.get('database', 'classificazioni')
    
    print(f"🔗 Connessione a MongoDB: {url}")
    print(f"📊 Database: {database_name}")
    
    client = MongoClient(url)
    
    # Test connessione
    try:
        client.admin.command('ismaster')
        print("✅ Connessione MongoDB riuscita")
        return client
    except Exception as e:
        print(f"❌ Errore connessione MongoDB: {e}")
        raise

def analyze_specific_case(db, case_id: str) -> Dict[str, Any]:
    """
    Analizza caso specifico in tutte le collezioni
    
    Args:
        db: Database MongoDB
        case_id: ID del caso da analizzare
        
    Returns:
        Informazioni complete sul caso
        
    Data ultima modifica: 2025-01-31
    """
    print(f"\n🔍 ANALISI CASO SPECIFICO: {case_id}")
    print("=" * 60)
    
    case_info = {
        'case_id': case_id,
        'found_in_collections': [],
        'details': {},
        'status': 'unknown'
    }
    
    # Prima cerca nella collezione specifica dell'utente
    main_collection_name = 'humanitas_015007d9-d413-11ef-86a5-96000228e7fe'
    
    print(f"🎯 Ricerca principale nella collezione: {main_collection_name}")
    try:
        main_collection = db[main_collection_name]
        
        # Cerca il caso usando _id direttamente
        from bson import ObjectId
from config_loader import load_config
        try:
            # Prova prima come ObjectId
            main_documents = list(main_collection.find({"_id": ObjectId(case_id)}))
        except:
            # Se non è un ObjectId valido, cerca come stringa in vari campi
            main_documents = list(main_collection.find({
                "$or": [
                    {"session_id": case_id},
                    {"_id": case_id},
                    {"id": case_id}
                ]
            }))
        
        if main_documents:
            case_info['found_in_collections'].append(main_collection_name)
            case_info['details'][main_collection_name] = main_documents
            
            print(f"✅ TROVATO nella collezione principale: {len(main_documents)} documento(i)")
            
            for doc in main_documents:
                print(f"   📄 Documento trovato:")
                
                # Campi importanti da mostrare
                important_fields = [
                    '_id', 'session_id', 'status', 'label', 'predicted_label', 
                    'confidence', 'is_outlier', 'is_representative', 
                    'cluster_id', 'timestamp', 'classification_method',
                    'ml_prediction', 'llm_prediction', 'text', 'embedding'
                ]
                
                for field in important_fields:
                    if field in doc:
                        value = doc[field]
                        # Tronca embedding per visualizzazione
                        if field == 'embedding' and isinstance(value, list):
                            print(f"      {field}: [array di {len(value)} elementi]")
                        elif field == 'text' and isinstance(value, str) and len(value) > 100:
                            print(f"      {field}: {value[:100]}...")
                        else:
                            print(f"      {field}: {value}")
        else:
            print(f"❌ Non trovato nella collezione principale")
    
    except Exception as e:
        print(f"⚠️ Errore controllo collezione principale: {e}")
    
    # Lista altre collezioni da controllare
    other_collections = [
        'rappresentanti',
        'outliers', 
        'propagati',
        'sessioni',
        'review_queue',
        'training_decisions',
        'clustering_results',
        'classification_results'
    ]
    
    print(f"\n🔍 Ricerca in altre collezioni...")
    for collection_name in other_collections:
        try:
            collection = db[collection_name]
            
            # Cerca il caso nella collezione
            documents = list(collection.find({"session_id": case_id}))
            
            if documents:
                case_info['found_in_collections'].append(collection_name)
                case_info['details'][collection_name] = documents
                
                print(f"✅ Trovato in collezione '{collection_name}': {len(documents)} documento(i)")
                
                # Stampa dettagli principali
                for doc in documents:
                    if '_id' in doc:
                        del doc['_id']  # Rimuovi ObjectId per visualizzazione
                    
                    print(f"   📄 Documento:")
                    
                    # Campi importanti da mostrare
                    important_fields = [
                        'session_id', 'status', 'label', 'predicted_label', 
                        'confidence', 'is_outlier', 'is_representative', 
                        'cluster_id', 'timestamp', 'classification_method',
                        'ml_prediction', 'llm_prediction'
                    ]
                    
                    for field in important_fields:
                        if field in doc:
                            print(f"      {field}: {doc[field]}")
                    
                    # Se ci sono altri campi interessanti
                    other_fields = {k: v for k, v in doc.items() if k not in important_fields}
                    if other_fields:
                        print(f"      Altri campi: {list(other_fields.keys())}")
            else:
                print(f"❌ Non trovato in collezione '{collection_name}'")
                
        except Exception as e:
            print(f"⚠️ Errore controllo collezione '{collection_name}': {e}")
    
    # Determina lo status del caso
    if 'rappresentanti' in case_info['found_in_collections']:
        case_info['status'] = 'rappresentante'
    elif 'outliers' in case_info['found_in_collections']:
        case_info['status'] = 'outlier'
    elif 'propagati' in case_info['found_in_collections']:
        case_info['status'] = 'propagato'
    
    print(f"\n📊 STATO DEL CASO: {case_info['status'].upper()}")
    
    return case_info

def count_representatives(db) -> Dict[str, Any]:
    """
    Conta tutti i rappresentanti salvati per tenant
    
    Args:
        db: Database MongoDB
        
    Returns:
        Statistiche sui rappresentanti
        
    Data ultima modifica: 2025-01-31
    """
    print(f"\n📊 CONTEGGIO RAPPRESENTANTI")
    print("=" * 40)
    
    try:
        rappresentanti_collection = db['rappresentanti']
        
        # Conta totale
        total_count = rappresentanti_collection.count_documents({})
        print(f"📈 Totale rappresentanti: {total_count}")
        
        # Raggruppa per tenant
        pipeline = [
            {
                "$group": {
                    "_id": "$tenant_id",
                    "count": {"$sum": 1},
                    "labels": {"$addToSet": "$label"}
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        tenant_stats = list(rappresentanti_collection.aggregate(pipeline))
        
        print(f"\n📊 Rappresentanti per tenant:")
        for stat in tenant_stats:
            tenant_id = stat['_id'] or 'NESSUN_TENANT'
            count = stat['count']
            labels = stat['labels']
            print(f"   🏢 {tenant_id}: {count} rappresentanti")
            print(f"      Etichette: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
        
        # Raggruppa per label
        pipeline_labels = [
            {
                "$group": {
                    "_id": "$label", 
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}}
        ]
        
        label_stats = list(rappresentanti_collection.aggregate(pipeline_labels))
        
        print(f"\n🏷️ Rappresentanti per etichetta:")
        for stat in label_stats:
            label = stat['_id'] or 'NESSUNA_ETICHETTA'
            count = stat['count']
            print(f"   📌 {label}: {count} rappresentanti")
        
        return {
            'total_count': total_count,
            'by_tenant': tenant_stats,
            'by_label': label_stats
        }
        
    except Exception as e:
        print(f"❌ Errore conteggio rappresentanti: {e}")
        return {}

def analyze_review_queue_na_problem(db) -> Dict[str, Any]:
    """
    Analizza il problema delle classificazioni N/A nella review queue
    
    Args:
        db: Database MongoDB
        
    Returns:
        Analisi del problema
        
    Data ultima modifica: 2025-01-31
    """
    print(f"\n🚨 ANALISI PROBLEMA CLASSIFICAZIONI N/A")
    print("=" * 50)
    
    analysis = {
        'review_queue_stats': {},
        'na_classifications': {},
        'missing_models': {},
        'recommendations': []
    }
    
    try:
        # Analizza review queue
        review_collection = db['review_queue']
        
        total_review = review_collection.count_documents({})
        na_ml_count = review_collection.count_documents({"ml_prediction": {"$in": ["N/A", None, ""]}})
        na_llm_count = review_collection.count_documents({"llm_prediction": {"$in": ["N/A", None, ""]}})
        
        print(f"📊 Review Queue Statistics:")
        print(f"   Totale elementi: {total_review}")
        print(f"   ML Prediction N/A: {na_ml_count} ({na_ml_count/total_review*100:.1f}%)")
        print(f"   LLM Prediction N/A: {na_llm_count} ({na_llm_count/total_review*100:.1f}%)")
        
        analysis['review_queue_stats'] = {
            'total': total_review,
            'na_ml': na_ml_count,
            'na_llm': na_llm_count,
            'na_ml_percentage': na_ml_count/total_review*100 if total_review > 0 else 0,
            'na_llm_percentage': na_llm_count/total_review*100 if total_review > 0 else 0
        }
        
        # Campiona documenti con N/A per capire la struttura
        na_samples = list(review_collection.find({
            "$or": [
                {"ml_prediction": {"$in": ["N/A", None, ""]}},
                {"llm_prediction": {"$in": ["N/A", None, ""]}}
            ]
        }).limit(3))
        
        print(f"\n📄 Campioni con N/A:")
        for i, sample in enumerate(na_samples, 1):
            sample.pop('_id', None)
            print(f"   Campione {i}:")
            print(f"      session_id: {sample.get('session_id', 'N/A')}")
            print(f"      ml_prediction: {sample.get('ml_prediction', 'N/A')}")
            print(f"      llm_prediction: {sample.get('llm_prediction', 'N/A')}")
            print(f"      confidence: {sample.get('confidence', 'N/A')}")
            print(f"      timestamp: {sample.get('timestamp', 'N/A')}")
            print(f"      tenant_id: {sample.get('tenant_id', 'N/A')}")
        
        # Controlla se esistono modelli ML salvati
        print(f"\n🤖 CONTROLLO MODELLI ML:")
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.joblib', '.h5'))]
            print(f"   Cartella models esiste: {len(model_files)} file trovati")
            for model_file in model_files:
                print(f"      📁 {model_file}")
            analysis['missing_models']['files_found'] = model_files
        else:
            print(f"   ❌ Cartella models non esiste: {models_dir}")
            analysis['missing_models']['models_dir_exists'] = False
        
        # Cerca informazioni sui modelli nel database
        models_collection = db.get_collection('ml_models', create=False)
        if models_collection:
            model_docs = list(models_collection.find({}).limit(5))
            print(f"   📊 Documenti nella collezione ml_models: {len(model_docs)}")
            analysis['missing_models']['db_models'] = len(model_docs)
        
        # Raccomandazioni
        recommendations = []
        
        if na_ml_count > total_review * 0.5:
            recommendations.append("🔴 CRITICO: Oltre il 50% delle predizioni ML sono N/A - possibile mancanza modello ML")
        
        if na_llm_count > total_review * 0.3:
            recommendations.append("🟡 ATTENZIONE: Oltre il 30% delle predizioni LLM sono N/A - verificare configurazione LLM")
        
        if not os.path.exists(models_dir) or not model_files:
            recommendations.append("🔴 CRITICO: Nessun modello ML trovato - eseguire training supervisionato")
        
        recommendations.append("✅ SUGGERIMENTO: Verificare pipeline.py per assicurarsi che i modelli siano caricati correttamente")
        recommendations.append("✅ SUGGERIMENTO: Controllare log del server per errori di caricamento modelli")
        
        analysis['recommendations'] = recommendations
        
        print(f"\n💡 RACCOMANDAZIONI:")
        for rec in recommendations:
            print(f"   {rec}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Errore analisi review queue: {e}")
        return analysis

def main():
    """
    Funzione principale per analisi completa
    
    Data ultima modifica: 2025-01-31
    """
    print("🔍 ANALISI MONGODB - CASO SPECIFICO E REVIEW QUEUE")
    print("=" * 70)
    
    try:
        # Carica configurazione
        config = load_config()
        
        # Connetti a MongoDB
        client = connect_mongodb(config)
        db_name = config.get('mongodb', {}).get('database', 'classificazioni')
        db = client[db_name]
        
        # Analizza caso specifico
        case_id = "68bcec2b34ea2bca071d45be"
        case_analysis = analyze_specific_case(db, case_id)
        
        # Conta rappresentanti
        representatives_stats = count_representatives(db)
        
        # Analizza problema N/A
        na_problem_analysis = analyze_review_queue_na_problem(db)
        
        # Summary finale
        print(f"\n🎯 SUMMARY COMPLETO")
        print("=" * 40)
        print(f"✅ Caso {case_id}: {case_analysis['status']}")
        print(f"✅ Totale rappresentanti: {representatives_stats.get('total_count', 0)}")
        print(f"⚠️ Predizioni ML N/A: {na_problem_analysis['review_queue_stats'].get('na_ml_percentage', 0):.1f}%")
        print(f"⚠️ Predizioni LLM N/A: {na_problem_analysis['review_queue_stats'].get('na_llm_percentage', 0):.1f}%")
        
        # Salva risultati completi
        results = {
            'timestamp': datetime.now().isoformat(),
            'case_analysis': case_analysis,
            'representatives_stats': representatives_stats,
            'na_problem_analysis': na_problem_analysis
        }
        
        output_file = 'mongodb_analysis_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Risultati salvati in: {output_file}")
        
    except Exception as e:
        print(f"❌ Errore durante analisi: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()
            print("🔌 Connessione MongoDB chiusa")

if __name__ == "__main__":
    main()
