#!/usr/bin/env python3
"""
Migration Script: Da Collection Singola a Collection per Tenant

Migra i dati da:
- client_session_classifications (collection unica)
A:
- {tenant}_classifications (collection separate per tenant)

PRESERVA TUTTI I DATI ESISTENTI
NON MODIFICA LA COLLECTION ORIGINALE (mantiene backup)

Autore: Sistema di classificazione
Data: 2025-08-21
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import json

# Aggiungi il path root del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from mongo_classification_reader import MongoClassificationReader

class TenantCollectionMigrator:
    """Migra da collection singola a collection per tenant"""
    
    def __init__(self):
        self.mongodb_url = "mongodb://localhost:27017/"
        self.database_name = "classificazione_discussioni"
        self.source_collection = "client_session_classifications"
        self.client = None
        self.db = None
        
    def connect(self) -> bool:
        """Connessione a MongoDB"""
        try:
            self.client = MongoClient(self.mongodb_url)
            self.db = self.client[self.database_name]
            return True
        except Exception as e:
            print(f"❌ Errore connessione MongoDB: {e}")
            return False
    
    def get_existing_tenants(self) -> List[str]:
        """Recupera tutti i tenant dalla collection esistente"""
        try:
            source_coll = self.db[self.source_collection]
            
            # Trova tutti i valori distinti del campo 'client'
            tenants = source_coll.distinct("client")
            tenants = [t for t in tenants if t and t.strip()]  # Rimuovi valori vuoti
            
            print(f"📋 Tenant trovati nella collection esistente: {tenants}")
            return tenants
            
        except Exception as e:
            print(f"❌ Errore nel recupero tenant: {e}")
            return []
    
    def check_existing_tenant_collections(self) -> Dict[str, bool]:
        """Verifica se esistono già collection per i tenant"""
        try:
            existing_collections = self.db.list_collection_names()
            tenant_collections = {}
            
            for collection_name in existing_collections:
                if collection_name.endswith("_classifications"):
                    tenant_name = collection_name.replace("_classifications", "")
                    if tenant_name != "client_session":  # Esclude la collection originale
                        tenant_collections[tenant_name] = True
            
            if tenant_collections:
                print(f"⚠️  Collection tenant esistenti: {list(tenant_collections.keys())}")
            else:
                print("ℹ️  Nessuna collection tenant esistente")
                
            return tenant_collections
            
        except Exception as e:
            print(f"❌ Errore nel controllo collection esistenti: {e}")
            return {}
    
    def migrate_tenant_data(self, tenant_name: str) -> Dict[str, Any]:
        """Migra i dati di un singolo tenant"""
        try:
            source_coll = self.db[self.source_collection]
            target_collection_name = f"{tenant_name}_classifications"
            target_coll = self.db[target_collection_name]
            
            print(f"\n🔄 Migrazione tenant: {tenant_name}")
            print(f"   📂 Da: {self.source_collection}")
            print(f"   📂 A: {target_collection_name}")
            
            # Query per documenti del tenant
            query = {"client": tenant_name}
            documents = list(source_coll.find(query))
            
            if not documents:
                print(f"   ℹ️  Nessun documento trovato per {tenant_name}")
                return {"migrated": 0, "errors": 0, "tenant": tenant_name}
            
            print(f"   📊 Documenti da migrare: {len(documents)}")
            
            # Modifica documenti: rimuovi il campo 'client' (non più necessario)
            migrated_docs = []
            for doc in documents:
                # Crea una copia del documento
                new_doc = doc.copy()
                
                # Rimuovi il campo 'client' (ridondante nella collection del tenant)
                if 'client' in new_doc:
                    del new_doc['client']
                
                # Aggiungi metadata di migrazione
                new_doc['migration_metadata'] = {
                    'migrated_at': datetime.now().isoformat(),
                    'source_collection': self.source_collection,
                    'original_client_field': tenant_name
                }
                
                migrated_docs.append(new_doc)
            
            # Inserisci nella nuova collection
            if migrated_docs:
                result = target_coll.insert_many(migrated_docs, ordered=False)
                migrated_count = len(result.inserted_ids)
                print(f"   ✅ Migrati {migrated_count} documenti")
                
                # Crea indici per performance
                self.create_tenant_indexes(target_collection_name)
                
                return {
                    "migrated": migrated_count,
                    "errors": 0,
                    "tenant": tenant_name,
                    "target_collection": target_collection_name
                }
            else:
                print(f"   ⚠️  Nessun documento da inserire per {tenant_name}")
                return {"migrated": 0, "errors": 0, "tenant": tenant_name}
                
        except Exception as e:
            print(f"   ❌ Errore migrazione {tenant_name}: {e}")
            return {"migrated": 0, "errors": 1, "tenant": tenant_name, "error": str(e)}
    
    def create_tenant_indexes(self, collection_name: str):
        """Crea indici per performance nella collection del tenant"""
        try:
            coll = self.db[collection_name]
            
            # Indici per query comuni
            indexes = [
                ("session_id", 1),
                ("review_status", 1),
                ("classified_at", -1),
                ("confidence", -1),
                [("review_status", 1), ("classified_at", -1)]  # Indice composto
            ]
            
            for index in indexes:
                if isinstance(index, list):
                    coll.create_index(index)
                else:
                    coll.create_index([index])
            
            print(f"   📋 Indici creati per {collection_name}")
            
        except Exception as e:
            print(f"   ⚠️  Errore creazione indici: {e}")
    
    def verify_migration(self, tenant_name: str) -> bool:
        """Verifica che la migrazione sia andata a buon fine"""
        try:
            source_coll = self.db[self.source_collection]
            target_collection_name = f"{tenant_name}_classifications"
            target_coll = self.db[target_collection_name]
            
            # Conta documenti originali
            original_count = source_coll.count_documents({"client": tenant_name})
            
            # Conta documenti migrati
            migrated_count = target_coll.count_documents({})
            
            print(f"   🔍 Verifica {tenant_name}: {original_count} → {migrated_count}")
            
            if original_count == migrated_count:
                print(f"   ✅ Migrazione verificata per {tenant_name}")
                return True
            else:
                print(f"   ❌ Discrepanza count per {tenant_name}")
                return False
                
        except Exception as e:
            print(f"   ❌ Errore verifica {tenant_name}: {e}")
            return False
    
    def run_migration(self, backup_original: bool = True) -> Dict[str, Any]:
        """Esegue la migrazione completa"""
        print("🚀 AVVIO MIGRAZIONE TENANT COLLECTIONS")
        print("=" * 60)
        
        if not self.connect():
            return {"success": False, "error": "Connessione MongoDB fallita"}
        
        # 1. Trova tenant esistenti
        tenants = self.get_existing_tenants()
        if not tenants:
            print("⚠️  Nessun tenant trovato da migrare")
            return {"success": True, "message": "Nessun dato da migrare"}
        
        # 2. Controlla collection esistenti
        existing = self.check_existing_tenant_collections()
        
        # 3. Backup collection originale se richiesto
        if backup_original:
            self.backup_original_collection()
        
        # 4. Migra ogni tenant
        migration_results = []
        total_migrated = 0
        total_errors = 0
        
        for tenant in tenants:
            if tenant in existing:
                print(f"⏭️  Saltato {tenant} (collection già esistente)")
                continue
                
            result = self.migrate_tenant_data(tenant)
            migration_results.append(result)
            total_migrated += result.get("migrated", 0)
            total_errors += result.get("errors", 0)
            
            # Verifica migrazione
            if result.get("migrated", 0) > 0:
                self.verify_migration(tenant)
        
        # 5. Riepilogo
        print("\n" + "=" * 60)
        print("📊 RIEPILOGO MIGRAZIONE:")
        print(f"   📦 Tenant processati: {len(tenants)}")
        print(f"   ✅ Documenti migrati: {total_migrated}")
        print(f"   ❌ Errori: {total_errors}")
        
        if total_errors == 0:
            print("🎉 Migrazione completata con successo!")
        else:
            print("⚠️  Migrazione completata con errori")
        
        return {
            "success": total_errors == 0,
            "tenants_processed": len(tenants),
            "total_migrated": total_migrated,
            "total_errors": total_errors,
            "results": migration_results
        }
    
    def backup_original_collection(self):
        """Crea backup della collection originale"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self.source_collection}_backup_{timestamp}"
            
            print(f"💾 Creazione backup: {backup_name}")
            
            # Copia collection (pipeline aggregation)
            source_coll = self.db[self.source_collection]
            backup_coll = self.db[backup_name]
            
            # Copia tutti i documenti
            documents = list(source_coll.find())
            if documents:
                backup_coll.insert_many(documents)
                print(f"   ✅ Backup creato: {len(documents)} documenti")
            else:
                print(f"   ℹ️  Collection originale vuota, nessun backup necessario")
                
        except Exception as e:
            print(f"   ⚠️  Errore creazione backup: {e}")
    
    def disconnect(self):
        """Chiude connessione MongoDB"""
        if self.client:
            self.client.close()


def main():
    """Punto di ingresso principale"""
    print("🔄 TENANT COLLECTION MIGRATION TOOL")
    print("Migra da collection singola a collection per tenant")
    print("=" * 60)
    
    migrator = TenantCollectionMigrator()
    
    try:
        # Esegui migrazione con backup automatico
        result = migrator.run_migration(backup_original=True)
        
        if result.get("success"):
            print(f"\n✅ MIGRAZIONE COMPLETATA!")
            print(f"   📊 {result.get('total_migrated', 0)} documenti migrati")
            print(f"   🏢 {result.get('tenants_processed', 0)} tenant processati")
        else:
            print(f"\n❌ MIGRAZIONE FALLITA!")
            if "error" in result:
                print(f"   Errore: {result['error']}")
                
    except KeyboardInterrupt:
        print("\n⏹️  Migrazione interrotta dall'utente")
    except Exception as e:
        print(f"\n💥 Errore imprevisto: {e}")
    finally:
        migrator.disconnect()
    
    return result.get("success", False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
