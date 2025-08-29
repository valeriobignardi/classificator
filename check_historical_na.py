#!/usr/bin/env python3
"""
Verifica rappresentanti più vecchi per confronto N/A

Autore: Valerio Bignardi
Data: 2025-08-28
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mongo_classification_reader import MongoClassificationReader
from datetime import datetime, timedelta

def check_old_representatives():
    """
    Scopo: Confronta rappresentanti vecchi per vedere se mostravano N/A
    """
    
    print("🔍 Verifica rappresentanti storici...")
    
    try:
        mongo_reader = MongoClassificationReader()
        
        if mongo_reader.ensure_connection():
            collection_name = mongo_reader.generate_collection_name("humanitas")
            collection = mongo_reader.db[collection_name]
            
            print(f"📋 Collection: {collection_name}")
            
            # Query per rappresentanti storici
            query = {
                "cluster_metadata.is_representative": True
            }
            
            representatives = list(collection.find(query).sort("classified_at", -1).limit(20))
            
            print(f"📊 Trovati {len(representatives)} rappresentanti storici")
            
            na_count = 0
            non_na_count = 0
            
            for rep in representatives[:10]:  # Mostra solo i primi 10
                session_id = rep.get('session_id', 'unknown')[:20] + "..."
                classified_at = rep.get('classified_at', 'unknown')[:19]
                
                # Verifica ml_result
                ml_result = rep.get('ml_result')
                if ml_result is None:
                    ml_status = "❌ NULL"
                    na_count += 1
                elif ml_result.get('predicted_label') in [None, 'unknown', 'N/A']:
                    ml_status = "❌ N/A"
                    na_count += 1
                else:
                    ml_status = "✅ OK"
                    non_na_count += 1
                
                ml_label = ml_result.get('predicted_label', 'NULL') if ml_result else 'NULL'
                ml_conf = ml_result.get('confidence', 0.0) if ml_result else 0.0
                
                # Verifica llm_result
                llm_result = rep.get('llm_result')
                if llm_result is None:
                    llm_status = "❌ NULL"
                elif llm_result.get('predicted_label') in [None, 'unknown', 'N/A']:
                    llm_status = "❌ N/A"
                else:
                    llm_status = "✅ OK"
                
                llm_label = llm_result.get('predicted_label', 'NULL') if llm_result else 'NULL'
                llm_conf = llm_result.get('confidence', 0.0) if llm_result else 0.0
                
                print(f"🎯 {session_id} ({classified_at}):")
                print(f"   ML: {ml_status} {ml_label} ({ml_conf:.2f})")
                print(f"   LLM: {llm_status} {llm_label} ({llm_conf:.2f})")
                
            print(f"\n📈 STATISTICHE STORICHE:")
            print(f"   Rappresentanti con N/A: {na_count}")
            print(f"   Rappresentanti senza N/A: {non_na_count}")
            
            if na_count > 0:
                print(f"✅ CONFERMA PROBLEMA: {na_count} rappresentanti mostravano N/A")
                print("💡 Le correzioni applicate dovrebbero risolvere questo problema")
                return True
            else:
                print("⚠️ Non trovati rappresentanti con N/A storici")
                return False
                
        else:
            print("❌ Impossibile connettersi al database")
            return False
            
    except Exception as e:
        print(f"❌ Errore verifica: {e}")
        return False

if __name__ == "__main__":
    success = check_old_representatives()
    
    if success:
        print("\n🎯 PROBLEMA CONFERMATO: I rappresentanti storici avevano N/A")
        print("✅ Le correzioni implementate dovrebbero risolverlo nel prossimo training")
    else:
        print("\n⚠️ Problema N/A non confermato nei dati storici")
