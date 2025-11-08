#!/usr/bin/env python3
"""
============================================================================
Analisi MongoDB - Statistiche Rappresentanti, Outlier e Propagati
============================================================================

Autore: Valerio Bignardi  
Data creazione: 2025-09-07
Ultima modifica: 2025-09-07

Descrizione:
    Script per analizzare la collezione MongoDB del tenant Humanitas
    e calcolare statistiche dettagliate su:
    - Rappresentanti (is_representative=true)
    - Outlier (classification_type="OUTLIER")
    - Propagati (metadata.propagated=true)

Parametri:
    - Database: classificazioni
    - Collection: humanitas_015007d9-d413-11ef-86a5-96000228e7fe
    - Tenant: Humanitas (015007d9-d413-11ef-86a5-96000228e7fe)

============================================================================
"""

import sys
import os
import yaml
from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
import json
from config_loader import load_config

# Aggiungi path progetto
sys.path.append('/home/ubuntu/classificatore')

class HumanitasMongoAnalyzer:
    """
    Analizzatore MongoDB per statistiche tenant Humanitas
    
    Scopo:
        Analizza la collezione MongoDB per contare rappresentanti,
        outlier e propagati con statistiche dettagliate
        
    Metodi:
        analyze_collection: Analisi completa collezione
        count_representatives: Conta rappresentanti
        count_outliers: Conta outlier
        count_propagated: Conta propagati
        
    Data ultima modifica: 2025-09-07
    """
    
    def __init__(self):
        """
        Inizializza analyzer con configurazione MongoDB
        
        Data ultima modifica: 2025-09-07
        """
        self.config = self._load_config()
        self.mongo_client = None
        self.db = None
        self.collection = None
        
        # Tenant Humanitas
        self.tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        self.tenant_name = "Humanitas"
        self.collection_name = f"humanitas_{self.tenant_id}"
        
        print(f"🏥 Analizzatore MongoDB per {self.tenant_name}")
        print(f"📊 Collection: {self.collection_name}")
    
    def _load_config(self) -> dict:
        """
        Carica configurazione MongoDB da config.yaml
        
        Returns:
            dict: Configurazione MongoDB
            
        Data ultima modifica: 2025-09-07
        """
        try:
            config_path = '/home/ubuntu/classificatore/config.yaml'
            config = load_config()
                return config.get('mongodb', {})
        except Exception as e:
            print(f"❌ Errore caricamento config: {e}")
            return {}
    
    def connect_mongodb(self) -> bool:
        """
        Stabilisce connessione a MongoDB
        
        Returns:
            bool: True se connessione riuscita
            
        Data ultima modifica: 2025-09-07
        """
        try:
            mongo_url = self.config.get('url', 'mongodb://localhost:27017')
            self.mongo_client = MongoClient(mongo_url)
            
            # Database classificazioni
            self.db = self.mongo_client['classificazioni']
            self.collection = self.db[self.collection_name]
            
            # Test connessione
            self.mongo_client.admin.command('ping')
            print(f"✅ Connesso a MongoDB: {mongo_url}")
            print(f"🗄️  Database: classificazioni")
            print(f"📋 Collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore connessione MongoDB: {e}")
            return False
    
    def get_collection_stats(self) -> dict:
        """
        Ottiene statistiche generali della collezione
        
        Returns:
            dict: Statistiche collezione
            
        Data ultima modifica: 2025-09-07
        """
        try:
            # Conteggio documenti totali
            total_count = self.collection.count_documents({})
            
            # Conteggio per tenant_id
            tenant_count = self.collection.count_documents({
                'tenant_id': self.tenant_id
            })
            
            # Conteggio per classification_method
            pipeline = [
                {'$match': {'tenant_id': self.tenant_id}},
                {'$group': {
                    '_id': '$classification_method',
                    'count': {'$sum': 1}
                }}
            ]
            
            method_stats = list(self.collection.aggregate(pipeline))
            
            return {
                'total_documents': total_count,
                'tenant_documents': tenant_count,
                'classification_methods': {item['_id']: item['count'] for item in method_stats}
            }
            
        except Exception as e:
            print(f"❌ Errore statistiche collezione: {e}")
            return {}
    
    def count_representatives(self) -> dict:
        """
        Conta documenti rappresentanti
        
        Returns:
            dict: Statistiche rappresentanti
            
        Data ultima modifica: 2025-09-07
        """
        try:
            print("\n📊 ANALISI RAPPRESENTANTI...")
            
            # Conta rappresentanti tramite metadata.is_representative
            representatives_metadata = self.collection.count_documents({
                'tenant_id': self.tenant_id,
                'metadata.is_representative': True
            })
            
            # Conta rappresentanti tramite campo representative
            representatives_field = self.collection.count_documents({
                'tenant_id': self.tenant_id,
                'metadata.representative': True
            })
            
            # Esempi di rappresentanti
            sample_representatives = list(self.collection.find({
                'tenant_id': self.tenant_id,
                'metadata.is_representative': True
            }).limit(3))
            
            # Raggruppa per classificazione
            pipeline = [
                {'$match': {
                    'tenant_id': self.tenant_id,
                    'metadata.is_representative': True
                }},
                {'$group': {
                    '_id': '$classification',
                    'count': {'$sum': 1},
                    'examples': {'$push': '$session_id'}
                }}
            ]
            
            by_classification = list(self.collection.aggregate(pipeline))
            
            return {
                'total_by_metadata': representatives_metadata,
                'total_by_field': representatives_field,
                'by_classification': {item['_id']: {
                    'count': item['count'],
                    'example_sessions': item['examples'][:3]
                } for item in by_classification},
                'sample_documents': sample_representatives
            }
            
        except Exception as e:
            print(f"❌ Errore conteggio rappresentanti: {e}")
            return {}
    
    def count_outliers(self) -> dict:
        """
        Conta documenti outlier
        
        Returns:
            dict: Statistiche outlier
            
        Data ultima modifica: 2025-09-07
        """
        try:
            print("\n🔍 ANALISI OUTLIER...")
            
            # Conta outlier tramite classification_type
            outliers_type = self.collection.count_documents({
                'tenant_id': self.tenant_id,
                'classification_type': 'OUTLIER'
            })
            
            # Conta outlier tramite metadata.outlier
            outliers_metadata = self.collection.count_documents({
                'tenant_id': self.tenant_id,
                'metadata.outlier': True
            })
            
            # Esempi di outlier
            sample_outliers = list(self.collection.find({
                'tenant_id': self.tenant_id,
                'classification_type': 'OUTLIER'
            }).limit(3))
            
            # Raggruppa per classificazione
            pipeline = [
                {'$match': {
                    'tenant_id': self.tenant_id,
                    'classification_type': 'OUTLIER'
                }},
                {'$group': {
                    '_id': '$classification',
                    'count': {'$sum': 1},
                    'examples': {'$push': '$session_id'}
                }}
            ]
            
            by_classification = list(self.collection.aggregate(pipeline))
            
            return {
                'total_by_type': outliers_type,
                'total_by_metadata': outliers_metadata,
                'by_classification': {item['_id']: {
                    'count': item['count'],
                    'example_sessions': item['examples'][:3]
                } for item in by_classification},
                'sample_documents': sample_outliers
            }
            
        except Exception as e:
            print(f"❌ Errore conteggio outlier: {e}")
            return {}
    
    def count_propagated(self) -> dict:
        """
        Conta documenti propagati
        
        Returns:
            dict: Statistiche propagati
            
        Data ultima modifica: 2025-09-07
        """
        try:
            print("\n🔄 ANALISI PROPAGATI...")
            
            # Conta propagati tramite metadata.propagated
            propagated_count = self.collection.count_documents({
                'tenant_id': self.tenant_id,
                'metadata.propagated': True
            })
            
            # Esempi di propagati
            sample_propagated = list(self.collection.find({
                'tenant_id': self.tenant_id,
                'metadata.propagated': True
            }).limit(3))
            
            # Raggruppa per classificazione
            pipeline = [
                {'$match': {
                    'tenant_id': self.tenant_id,
                    'metadata.propagated': True
                }},
                {'$group': {
                    '_id': '$classification',
                    'count': {'$sum': 1},
                    'examples': {'$push': '$session_id'}
                }}
            ]
            
            by_classification = list(self.collection.aggregate(pipeline))
            
            return {
                'total_propagated': propagated_count,
                'by_classification': {item['_id']: {
                    'count': item['count'],
                    'example_sessions': item['examples'][:3]
                } for item in by_classification},
                'sample_documents': sample_propagated
            }
            
        except Exception as e:
            print(f"❌ Errore conteggio propagati: {e}")
            return {}
    
    def analyze_collection(self) -> dict:
        """
        Analisi completa della collezione
        
        Returns:
            dict: Tutte le statistiche
            
        Data ultima modifica: 2025-09-07
        """
        print("🔍 AVVIO ANALISI COLLEZIONE HUMANITAS")
        print("="*60)
        
        if not self.connect_mongodb():
            return {'error': 'Connessione MongoDB fallita'}
        
        try:
            # Statistiche generali
            collection_stats = self.get_collection_stats()
            print(f"📊 Documenti totali: {collection_stats.get('total_documents', 0)}")
            print(f"👤 Documenti Humanitas: {collection_stats.get('tenant_documents', 0)}")
            
            # Analisi dettagliate
            representatives = self.count_representatives()
            outliers = self.count_outliers()
            propagated = self.count_propagated()
            
            # Report finale
            print("\n" + "🏁" + "="*60)
            print("🏁 REPORT FINALE ANALISI HUMANITAS")
            print("🏁" + "="*60)
            
            print(f"\n📊 RAPPRESENTANTI:")
            print(f"   🎯 Totale (metadata.is_representative): {representatives.get('total_by_metadata', 0)}")
            print(f"   🎯 Totale (metadata.representative): {representatives.get('total_by_field', 0)}")
            
            print(f"\n🔍 OUTLIER:")
            print(f"   🚨 Totale (classification_type): {outliers.get('total_by_type', 0)}")
            print(f"   🚨 Totale (metadata.outlier): {outliers.get('total_by_metadata', 0)}")
            
            print(f"\n🔄 PROPAGATI:")
            print(f"   📈 Totale (metadata.propagated): {propagated.get('total_propagated', 0)}")
            
            # Statistiche per classificazione
            print(f"\n📋 DISTRIBUZIONE PER CLASSIFICAZIONE:")
            
            all_classifications = set()
            if representatives.get('by_classification'):
                all_classifications.update(representatives['by_classification'].keys())
            if outliers.get('by_classification'):
                all_classifications.update(outliers['by_classification'].keys())
            if propagated.get('by_classification'):
                all_classifications.update(propagated['by_classification'].keys())
            
            for classification in sorted(all_classifications):
                repr_count = representatives.get('by_classification', {}).get(classification, {}).get('count', 0)
                outl_count = outliers.get('by_classification', {}).get(classification, {}).get('count', 0)
                prop_count = propagated.get('by_classification', {}).get(classification, {}).get('count', 0)
                
                print(f"   📌 {classification}:")
                print(f"      🎯 Rappresentanti: {repr_count}")
                print(f"      🚨 Outlier: {outl_count}")
                print(f"      📈 Propagati: {prop_count}")
            
            return {
                'success': True,
                'tenant_id': self.tenant_id,
                'tenant_name': self.tenant_name,
                'collection_name': self.collection_name,
                'analysis_date': datetime.now().isoformat(),
                'collection_stats': collection_stats,
                'representatives': representatives,
                'outliers': outliers,
                'propagated': propagated,
                'summary': {
                    'total_representatives': representatives.get('total_by_metadata', 0),
                    'total_outliers': outliers.get('total_by_type', 0),
                    'total_propagated': propagated.get('total_propagated', 0),
                    'total_documents': collection_stats.get('tenant_documents', 0)
                }
            }
            
        except Exception as e:
            print(f"❌ Errore analisi: {e}")
            return {'error': str(e)}
        
        finally:
            if self.mongo_client:
                self.mongo_client.close()
                print("🔐 Connessione MongoDB chiusa")

def main():
    """
    Funzione principale per eseguire l'analisi
    
    Data ultima modifica: 2025-09-07
    """
    analyzer = HumanitasMongoAnalyzer()
    result = analyzer.analyze_collection()
    
    if result.get('success'):
        print("\n✅ ANALISI COMPLETATA CON SUCCESSO!")
        
        # Salva risultati su file
        output_file = f"/home/ubuntu/classificatore/humanitas_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Risultati salvati in: {output_file}")
        
    else:
        print(f"❌ ANALISI FALLITA: {result.get('error')}")

if __name__ == "__main__":
    main()
