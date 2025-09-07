#!/usr/bin/env python3
"""
Script per correggere i parametri di clustering direttamente nel database MySQL
Autore: Valerio Bignardi
Data: 2025-01-17

SOLUZIONE ALTERNATIVA: Accesso diretto al database MySQL per correggere i parametri
"""

import sys
import os
import yaml
import mysql.connector
from datetime import datetime

# Aggiungi il path del progetto
sys.path.append('/home/ubuntu/classificatore')

def get_database_config():
    """
    Carica la configurazione del database MySQL dal config.yaml
    """
    config_path = '/home/ubuntu/classificatore/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config['database']

def fix_clustering_parameters_direct():
    """
    Corregge i parametri di clustering accedendo direttamente al database MySQL
    """
    print("🔧 CORREZIONE DIRETTA PARAMETRI CLUSTERING - HUMANITAS")
    print("="*70)
    
    tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
    
    # Parametri corretti (meno restrittivi)
    optimal_params = {
        'min_cluster_size': 5,        # Era 13 → Ridotto del 62%
        'min_samples': 3,             # Era 16 → Ridotto dell'81%
        'cluster_selection_epsilon': 0.1,  # Era 0.28 → Ridotto del 64%
        'alpha': 1.0,                 # Era 0.4 → Aumentato del 150%
        'cluster_selection_method': 'eom'  # Mantiene eom
    }
    
    print("📊 PARAMETRI ATTUALI → NUOVI:")
    print("   - min_cluster_size: 13 → 5 (⬇ 62%)")
    print("   - min_samples: 16 → 3 (⬇ 81%)")  
    print("   - cluster_selection_epsilon: 0.28 → 0.1 (⬇ 64%)")
    print("   - alpha: 0.4 → 1.0 (⬆ 150%)")
    print("   - cluster_selection_method: eom → eom (✅ OK)")
    
    try:
        # Connessione al database
        db_config = get_database_config()
        print(f"\\n🔌 Connessione al database MySQL...")
        print(f"   Host: {db_config['host']}")
        print(f"   Database: {db_config['database']}")
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Trova il record del tenant
        print(f"\\n🔍 Ricerca tenant {tenant_id}...")
        
        select_query = """
        SELECT id, tenant_id, tenant_name, config_clustering 
        FROM engines 
        WHERE tenant_id = %s
        """
        cursor.execute(select_query, (tenant_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"❌ Tenant {tenant_id} non trovato nel database!")
            return False
        
        record_id, db_tenant_id, tenant_name, current_config = result
        print(f"✅ Tenant trovato: {tenant_name} (ID: {record_id})")
        
        # Parse configurazione esistente
        if current_config:
            import json
            config_dict = json.loads(current_config)
        else:
            config_dict = {}
        
        print(f"📋 Configurazione attuale: {len(config_dict)} parametri")
        
        # Aggiorna i parametri
        config_dict.update(optimal_params)
        new_config_json = json.dumps(config_dict, indent=2)
        
        print(f"📋 Nuova configurazione: {len(config_dict)} parametri")
        
        # Backup del record originale
        backup_query = """
        INSERT INTO engines_backup 
        (original_id, tenant_id, tenant_name, config_clustering, backup_date, reason)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        backup_data = (
            record_id, 
            db_tenant_id, 
            tenant_name, 
            current_config,
            datetime.now(),
            "Backup before clustering parameters fix"
        )
        
        # Crea tabella backup se non esiste
        create_backup_table = """
        CREATE TABLE IF NOT EXISTS engines_backup (
            id INT AUTO_INCREMENT PRIMARY KEY,
            original_id INT,
            tenant_id VARCHAR(255),
            tenant_name VARCHAR(255),
            config_clustering TEXT,
            backup_date DATETIME,
            reason TEXT
        )
        """
        cursor.execute(create_backup_table)
        
        try:
            cursor.execute(backup_query, backup_data)
            print("✅ Backup creato nella tabella engines_backup")
        except Exception as e:
            print(f"⚠️ Warning backup: {e}")
        
        # Aggiorna il record principale
        update_query = """
        UPDATE engines 
        SET config_clustering = %s, updated_at = %s
        WHERE tenant_id = %s
        """
        
        cursor.execute(update_query, (new_config_json, datetime.now(), tenant_id))
        
        # Commit delle modifiche
        conn.commit()
        
        print("\\n✅ PARAMETRI AGGIORNATI CON SUCCESSO!")
        print(f"📊 Record ID: {record_id}")
        print(f"📊 Parametri modificati: {len(optimal_params)}")
        print(f"📊 Timestamp: {datetime.now().isoformat()}")
        
        print("\\n🎯 EFFETTI ATTESI:")
        print("   • min_cluster_size ridotto: più cluster piccoli accettati")
        print("   • min_samples ridotto: formazione cluster più permissiva")  
        print("   • epsilon ridotto: soglia distanza più bassa")
        print("   • alpha aumentato: meno sensibile al rumore")
        
        print("\\n🚀 PROSSIMI PASSI:")
        print("1. 🔄 Riavvia la pipeline di clustering")
        print("2. 🧠 Esegui clustering sui 1360 documenti Humanitas")
        print("3. ✅ Verifica formazione cluster validi (non tutti outlier)")
        print("4. 👥 Testa selezione rappresentanti (dovrebbe restituire > 0)")
        print("5. 📊 Monitora qualità clustering con nuovi parametri")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Errore aggiornamento database: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """
    Verifica che i parametri siano stati aggiornati correttamente
    """
    print("\\n🔍 VERIFICA APPLICAZIONE PARAMETRI...")
    
    try:
        from Utils.tenant_config_helper import get_all_clustering_parameters_for_tenant
        
        tenant_id = "015007d9-d413-11ef-86a5-96000228e7fe"
        params = get_all_clustering_parameters_for_tenant(tenant_id)
        
        print("📊 PARAMETRI VERIFICATI:")
        expected = {
            'min_cluster_size': 5,
            'min_samples': 3,
            'cluster_selection_epsilon': 0.1,
            'alpha': 1.0,
            'cluster_selection_method': 'eom'
        }
        
        all_correct = True
        for param, expected_value in expected.items():
            actual_value = params.get(param)
            status = "✅" if actual_value == expected_value else "❌"
            print(f"   {status} {param}: {actual_value} (atteso: {expected_value})")
            if actual_value != expected_value:
                all_correct = False
        
        if all_correct:
            print("\\n🎉 TUTTI I PARAMETRI SONO CORRETTI!")
            return True
        else:
            print("\\n⚠️ ALCUNI PARAMETRI NON SONO CORRETTI")
            return False
            
    except Exception as e:
        print(f"❌ Errore verifica: {e}")
        return False

if __name__ == "__main__":
    print("🔧 AVVIO CORREZIONE DIRETTA PARAMETRI CLUSTERING")
    
    success = fix_clustering_parameters_direct()
    
    if success:
        # Verifica immediata
        verify_success = verify_fix()
        
        if verify_success:
            print("\\n🎉 CORREZIONE COMPLETATA E VERIFICATA!")
            print("📈 I parametri clustering sono ora ottimizzati per 1360 documenti")
            print("🎯 Il problema dei 0 rappresentanti dovrebbe essere risolto")
        else:
            print("\\n⚠️ Correzione applicata ma verifica fallita")
    else:
        print("\\n❌ CORREZIONE FALLITA!")
        sys.exit(1)
