#!/usr/bin/env python3
"""
Patch per aggiungere logging dettagliato al RemoteTagSyncService

Scopo: Tracciare se e come viene chiamato il RemoteTagSyncService durante il training
Autore: Valerio Bignardi  
Data: 2025-01-15
"""

import os
import shutil
from datetime import datetime

def backup_and_patch_remote_sync():
    """Crea backup e aggiunge logging dettagliato"""
    
    remote_sync_file = "/home/ubuntu/classificatore/Database/remote_tag_sync.py"
    backup_file = f"/home/ubuntu/classificatore/backup/remote_tag_sync_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    # Crea directory backup se non esiste
    os.makedirs(os.path.dirname(backup_file), exist_ok=True)
    
    # Backup del file originale
    shutil.copy2(remote_sync_file, backup_file)
    print(f"✅ Backup creato: {backup_file}")
    
    # Leggi il contenuto attuale
    with open(remote_sync_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aggiungi import datetime se non presente
    if 'from datetime import datetime' not in content:
        content = content.replace(
            'from typing import List, Dict, Any, Optional',
            'from typing import List, Dict, Any, Optional\nfrom datetime import datetime'
        )
    
    # Patch del metodo sync_session_tags per aggiungere logging
    old_sync_method = '''    def sync_session_tags(self, tenant: Tenant, documenti: List[Any]) -> Dict[str, Any]:
        """
        Upsert tags and session→tag rows for a batch of classified documents.

        Args:
            tenant: Tenant object with tenant_slug/tenant_id
            documenti: List of DocumentoProcessing (or objects exposing same attributes)

        Returns:
            Summary dict with counters and optional error
        """
        tenant_slug = getattr(tenant, 'tenant_slug', None) or getattr(tenant, 'tenant_name', None)
        if not tenant_slug:
            return {'success': False, 'error': 'tenant_slug non disponibile'}'''
    
    new_sync_method = '''    def sync_session_tags(self, tenant: Tenant, documenti: List[Any]) -> Dict[str, Any]:
        """
        Upsert tags and session→tag rows for a batch of classified documents.

        Args:
            tenant: Tenant object with tenant_slug/tenant_id
            documenti: List of DocumentoProcessing (or objects exposing same attributes)

        Returns:
            Summary dict with counters and optional error
        """
        # 🚨 LOGGING AGGIUNTO PER DEBUG
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\\n🔍 [{timestamp}] RemoteTagSyncService.sync_session_tags() CHIAMATO")
        print(f"   📋 Tenant: {getattr(tenant, 'tenant_slug', 'N/A')} (ID: {getattr(tenant, 'tenant_id', 'N/A')})")
        print(f"   📊 Documenti ricevuti: {len(documenti) if documenti else 0}")
        
        tenant_slug = getattr(tenant, 'tenant_slug', None) or getattr(tenant, 'tenant_name', None)
        if not tenant_slug:
            error_msg = 'tenant_slug non disponibile'
            print(f"   ❌ ERRORE: {error_msg}")
            return {'success': False, 'error': error_msg}
        
        print(f"   🎯 Target schema: {tenant_slug}")'''
    
    content = content.replace(old_sync_method, new_sync_method)
    
    # Patch per aggiungere logging al termine del sync
    old_return = '''            return {
                'success': True,
                'tag_inserts': tag_inserts,
                'tag_updates': tag_updates,
                'session_inserts': session_inserts,
                'session_updates': session_updates,
            }'''
    
    new_return = '''            result = {
                'success': True,
                'tag_inserts': tag_inserts,
                'tag_updates': tag_updates,
                'session_inserts': session_inserts,
                'session_updates': session_updates,
            }
            
            # 🚨 LOGGING RISULTATO
            print(f"   ✅ SYNC COMPLETATO: {result}")
            return result'''
    
    content = content.replace(old_return, new_return)
    
    # Patch per logging errori
    old_error_return = '''        except Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            return {'success': False, 'error': f"Errore durante sync remoto: {e}"}'''
    
    new_error_return = '''        except Error as e:
            try:
                conn.rollback()
            except Exception:
                pass
            error_result = {'success': False, 'error': f"Errore durante sync remoto: {e}"}
            print(f"   ❌ SYNC FALLITO: {error_result}")
            return error_result'''
    
    content = content.replace(old_error_return, new_error_return)
    
    # Salva il file patchato
    with open(remote_sync_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ RemoteTagSyncService patchato con logging dettagliato")
    print(f"   Ora tutte le chiamate al sync verranno loggate sulla console")
    print(f"   Per ripristinare: cp {backup_file} {remote_sync_file}")

if __name__ == '__main__':
    print("🔧 Patch RemoteTagSyncService per debugging")
    print("=" * 50)
    backup_and_patch_remote_sync()
    print("\\n🎯 Ora esegui il training supervisionado e osserva i log!")