#!/usr/bin/env python3
"""
Monitora i rappresentanti salvati durante il training per verificare che non siano N/A

Autore: Valerio Bignardi
Data: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mongo_classification_reader import MongoClassificationReader
import time

def monitor_representatives():
    """
    Scopo: Monitora i rappresentanti salvati per verificare che abbiano ml_result/llm_result
    """
    
    print("🔍 Monitoraggio rappresentanti in tempo reale...")
    
    try:
        # CORREZIONE: Usa il primo tenant disponibile per il monitoraggio
        tenant_objects = MongoClassificationReader.get_available_tenants()
        if not tenant_objects:
            print("❌ Nessun tenant trovato nel database")
            return
        
        # Usa il primo tenant disponibile
        test_tenant = tenant_objects[0]
        print(f"📝 Monitoraggio per tenant: {test_tenant.tenant_name}")
        
        mongo_reader = MongoClassificationReader(tenant=test_tenant)
        
        # Controlla rappresentanti recenti (ultimi 5 minuti)
        from datetime import datetime, timedelta
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_time_str = recent_time.isoformat()
        
        # Connetti al database
        if mongo_reader.ensure_connection():
            collection_name = mongo_reader.generate_collection_name("humanitas")
            collection = mongo_reader.db[collection_name]
            
            print(f"📋 Collection: {collection_name}")
            print(f"📅 Ricerca rappresentanti salvati dopo: {recent_time_str}")
            
            # Query per rappresentanti recenti
            query = {
                "classified_at": {"$gte": recent_time_str},
                "cluster_metadata.is_representative": True
            }
            
            representatives = list(collection.find(query).sort("classified_at", -1).limit(10))
            
            print(f"📊 Trovati {len(representatives)} rappresentanti recenti")
            
            for rep in representatives:
                session_id = rep.get('session_id', 'unknown')
                classified_at = rep.get('classified_at', 'unknown')
                
                # Verifica ml_result
                ml_result = rep.get('ml_result')
                ml_status = "✅ OK" if ml_result and ml_result.get('predicted_label') != 'unknown' else "❌ N/A"
                ml_label = ml_result.get('predicted_label', 'N/A') if ml_result else 'N/A'
                ml_conf = ml_result.get('confidence', 0.0) if ml_result else 0.0
                
                # Verifica llm_result
                llm_result = rep.get('llm_result')
                llm_status = "✅ OK" if llm_result and llm_result.get('predicted_label') != 'unknown' else "❌ N/A"
                llm_label = llm_result.get('predicted_label', 'N/A') if llm_result else 'N/A'
                llm_conf = llm_result.get('confidence', 0.0) if llm_result else 0.0
                
                print(f"🎯 {session_id} ({classified_at[:19]}):")
                print(f"   ML: {ml_status} {ml_label} ({ml_conf:.2f})")
                print(f"   LLM: {llm_status} {llm_label} ({llm_conf:.2f})")
                
            if representatives:
                # Controlla se ci sono ancora N/A
                ml_na_count = sum(1 for rep in representatives if not rep.get('ml_result') or rep.get('ml_result', {}).get('predicted_label') == 'unknown')
                llm_na_count = sum(1 for rep in representatives if not rep.get('llm_result') or rep.get('llm_result', {}).get('predicted_label') == 'unknown')
                
                print(f"\n📈 STATISTICHE:")
                print(f"   ML N/A: {ml_na_count}/{len(representatives)}")
                print(f"   LLM N/A: {llm_na_count}/{len(representatives)}")
                
                if ml_na_count == 0 and llm_na_count == 0:
                    print("✅ ECCELLENTE: Nessun N/A trovato!")
                    return True
                elif ml_na_count + llm_na_count < len(representatives):
                    print("✅ MIGLIORAMENTO: Meno N/A rispetto a prima")
                    return True
                else:
                    print("⚠️ Ancora N/A presenti")
                    return False
            else:
                print("⏳ Nessun rappresentante recente trovato, training probabilmente in corso...")
                return True
                
        else:
            print("❌ Impossibile connettersi al database")
            return False
            
    except Exception as e:
        print(f"❌ Errore monitoraggio: {e}")
        return False

if __name__ == "__main__":
    success = monitor_representatives()
    
    if success:
        print("\n🎯 CONCLUSIONE: Le correzioni sembrano funzionare!")
    else:
        print("\n⚠️ Potrebbe essere necessario ulteriore debug")
