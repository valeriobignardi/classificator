#!/usr/bin/env python3
"""
Test della correzione della propagazione: NON deve toccare rappresentanti e outliers

Autore: Valerio Bignardi  
Data: 2025-09-04
"""

import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymongo

def test_propagation_fix():
    """
    Testa che la propagazione non tocchi rappresentanti e outliers
    """
    try:
        print("🧪 TEST CORREZIONE PROPAGAZIONE")
        
        # Connetti a MongoDB
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['classificazioni']
        collection = db['humanitas_015007d9-d413-11ef-86a5-96000228e7fe']
        
        print(f"📊 PRIMA DELLA CORREZIONE:")
        
        # Conta documenti per classified_by
        pipeline = [
            {'$group': {
                '_id': '$classified_by',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}}
        ]
        
        results = list(collection.aggregate(pipeline))
        print(f"   DISTRIBUZIONE classified_by:")
        for result in results:
            classified_by = result['_id'] or 'NULL'
            count = result['count']
            print(f"     {classified_by}: {count} documenti")
        
        # Verifica se ci sono rappresentanti con classified_by corretto
        supervised_docs = collection.count_documents({
            'classified_by': 'supervised_training_pipeline'
        })
        
        print(f"\n📋 ANALISI SPECIFICA:")
        print(f"   🎓 Documenti da supervised_training_pipeline: {supervised_docs}")
        
        if supervised_docs == 0:
            print(f"   ❌ PROBLEMA: Nessun documento con classified_by='supervised_training_pipeline'")
            print(f"   💡 Significa che la propagazione ha sovrascritto TUTTO")
            print(f"   🔧 La correzione dovrebbe risolvere questo!")
        else:
            print(f"   ✅ BENE: Ci sono documenti con classified_by corretto")
            
        # Conta rappresentanti reali
        representatives = collection.count_documents({
            'metadata.is_representative': True
        })
        
        print(f"   👑 Documenti con is_representative=True: {representatives}")
        
        if representatives == 0:
            print(f"   ❌ PROBLEMA: Nessun rappresentante!")
            print(f"   💡 I rappresentanti sono stati sovrascritti dalla propagazione")
        else:
            print(f"   ✅ BENE: Ci sono rappresentanti salvati")
            
        # Test: cerca documenti che dovrebbero essere rappresentanti
        # (based on review_reason)
        potential_reps = collection.count_documents({
            'review_reason': 'supervised_training_representative'
        })
        
        print(f"   🔍 Documenti con review_reason='supervised_training_representative': {potential_reps}")
        
        print(f"\n🎯 RISULTATO ATTESO DOPO CORREZIONE:")
        print(f"   ✅ Rappresentanti: classified_by='supervised_training_pipeline' + is_representative=True")
        print(f"   ✅ Outliers: classified_by='supervised_training_pipeline' + cluster_id=-1") 
        print(f"   ✅ Propagati: classified_by='cluster_propagation' + is_representative=False")
        print(f"   ❌ NESSUN documento rappresentante/outlier dovrebbe avere classified_by='cluster_propagation'")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_propagation_fix()
