#!/usr/bin/env python3
"""
Verifica struttura tabella clustering_test_results e analisi risultati
Autore: Valerio Bignardi
Data: 2025-08-28
"""

from Database.clustering_results_db import ClusteringResultsDB

def inspect_clustering_table():
    """
    Ispeziona la struttura della tabella clustering_test_results
    """
    results_db = ClusteringResultsDB()
    
    if not results_db.connect():
        print("❌ Impossibile connettersi al database")
        return
    
    try:
        cursor = results_db.connection.cursor()
        
        # Mostra struttura tabella
        print("📋 STRUTTURA TABELLA clustering_test_results:")
        print("=" * 60)
        cursor.execute("DESCRIBE clustering_test_results")
        columns = cursor.fetchall()
        
        for column in columns:
            field_name = column[0]
            field_type = column[1]
            null_allowed = column[2]
            key_type = column[3] if column[3] else ''
            default_val = column[4] if column[4] else ''
            extra = column[5] if column[5] else ''
            
            print(f"🔹 {field_name:20} | {field_type:15} | NULL: {null_allowed:3} | {key_type:3} | Default: {default_val}")
        
        # Test query base
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        
        print(f"\n🔍 DATI PRESENTI per tenant {tenant_id}:")
        print("=" * 80)
        
        cursor.execute("""
            SELECT * FROM clustering_test_results 
            WHERE tenant_id = %s 
            ORDER BY version_number
            LIMIT 3
        """, (tenant_id,))
        
        results = cursor.fetchall()
        
        if results:
            print(f"📊 Trovati {len(results)} record (mostro i primi 3):")
            
            # Ottieni nomi colonne
            cursor.execute("SHOW COLUMNS FROM clustering_test_results")
            column_names = [col[0] for col in cursor.fetchall()]
            
            for i, result in enumerate(results):
                print(f"\n🔸 Record {i+1}:")
                for j, value in enumerate(result):
                    if j < len(column_names):
                        print(f"   {column_names[j]:20} = {value}")
        else:
            print("❌ Nessun dato trovato per questo tenant")
            
        # Conta totale record per tenant
        cursor.execute("""
            SELECT COUNT(*) FROM clustering_test_results 
            WHERE tenant_id = %s
        """, (tenant_id,))
        
        total_count = cursor.fetchone()[0]
        print(f"\n📈 TOTALE RECORD per tenant: {total_count}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        results_db.disconnect()

if __name__ == "__main__":
    inspect_clustering_table()
