#!/usr/bin/env python3
"""
Script per normalizzare i tag a MAIUSCOLO in MongoDB

Problema: L'algoritmo crea duplicati con tag minuscoli invece di mantenere i MAIUSCOLI
Soluzione: Trasforma tutti i campi classification a MAIUSCOLO

Database: classificazioni
Collection: humanitas_015007d9-d413-11ef-86a5-96000228e7fe
Campo target: classification

Autore: Valerio Bignardi
Data creazione: 2025-11-05
"""

import sys
import os
from pymongo import MongoClient
from datetime import datetime

# Aggiungi path per importare config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def connect_to_mongodb():
    """
    Connette a MongoDB usando configurazione standard
    
    Returns:
        MongoClient: Client MongoDB connesso
    """
    try:
        # Leggi configurazione MongoDB
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        mongo_config = config.get('mongodb', {})
        host = mongo_config.get('host', 'localhost')
        port = mongo_config.get('port', 27017)
        
        print(f"🔗 Connessione a MongoDB: {host}:{port}")
        client = MongoClient(host, port, serverSelectionTimeoutMS=5000)
        
        # Verifica connessione
        client.admin.command('ping')
        print(f"✅ Connesso a MongoDB con successo")
        
        return client
        
    except Exception as e:
        print(f"❌ Errore connessione MongoDB: {e}")
        sys.exit(1)

def analyze_lowercase_tags(collection):
    """
    Analizza quanti documenti hanno tag minuscoli
    
    Args:
        collection: Collection MongoDB
        
    Returns:
        dict: Statistiche sui tag minuscoli
    """
    print(f"\n📊 ANALISI TAG MINUSCOLI")
    print(f"=" * 80)
    
    # Trova tutti i documenti con classification non-null
    total_docs = collection.count_documents({})
    docs_with_classification = collection.count_documents({'classification': {'$exists': True, '$ne': None}})
    
    print(f"📋 Documenti totali: {total_docs}")
    print(f"📋 Documenti con classification: {docs_with_classification}")
    
    # Trova documenti con tag minuscoli (contengono almeno una lettera minuscola)
    # MongoDB non ha regex nativo per questo, quindi facciamo in Python
    lowercase_tags = {}
    mixed_case_count = 0
    uppercase_count = 0
    
    for doc in collection.find({'classification': {'$exists': True, '$ne': None}}, 
                               {'_id': 1, 'classification': 1, 'session_id': 1}):
        tag = doc.get('classification', '')
        if not tag:
            continue
            
        if tag != tag.upper():
            # Ha lettere minuscole
            mixed_case_count += 1
            uppercase_version = tag.upper()
            
            if uppercase_version not in lowercase_tags:
                lowercase_tags[uppercase_version] = {
                    'lowercase_variants': set(),
                    'count': 0
                }
            
            lowercase_tags[uppercase_version]['lowercase_variants'].add(tag)
            lowercase_tags[uppercase_version]['count'] += 1
        else:
            uppercase_count += 1
    
    print(f"\n📊 RISULTATI:")
    print(f"   ✅ Tag già MAIUSCOLI: {uppercase_count}")
    print(f"   ⚠️  Tag con minuscole: {mixed_case_count}")
    
    if lowercase_tags:
        print(f"\n🔍 DETTAGLIO TAG DA CORREGGERE:")
        print(f"-" * 80)
        for uppercase_tag, info in sorted(lowercase_tags.items()):
            print(f"\n   🏷️  {uppercase_tag} (versione corretta)")
            for variant in sorted(info['lowercase_variants']):
                print(f"      ❌ '{variant}' → trovato {info['count']} volte")
    
    return {
        'total': total_docs,
        'with_classification': docs_with_classification,
        'uppercase': uppercase_count,
        'mixed_case': mixed_case_count,
        'lowercase_tags': lowercase_tags
    }

def fix_lowercase_tags(collection, dry_run=True):
    """
    Corregge tutti i tag minuscoli a MAIUSCOLO
    
    Args:
        collection: Collection MongoDB
        dry_run: Se True, simula senza modificare (default: True)
        
    Returns:
        dict: Statistiche delle modifiche
    """
    mode_str = "🔍 MODALITÀ SIMULAZIONE (DRY RUN)" if dry_run else "🔧 MODALITÀ MODIFICA EFFETTIVA"
    print(f"\n{mode_str}")
    print(f"=" * 80)
    
    updated_count = 0
    errors = []
    updates_by_tag = {}
    
    # Trova tutti i documenti con classification
    cursor = collection.find({'classification': {'$exists': True, '$ne': None}})
    
    for doc in cursor:
        old_tag = doc.get('classification', '')
        if not old_tag:
            continue
        
        new_tag = old_tag.upper()
        
        # Se diverso, aggiorna
        if old_tag != new_tag:
            session_id = doc.get('session_id', 'unknown')
            
            if dry_run:
                print(f"   📝 Simulazione: '{old_tag}' → '{new_tag}' (session: {session_id})")
            else:
                try:
                    result = collection.update_one(
                        {'_id': doc['_id']},
                        {'$set': {'classification': new_tag}}
                    )
                    
                    if result.modified_count > 0:
                        print(f"   ✅ Aggiornato: '{old_tag}' → '{new_tag}' (session: {session_id})")
                        updated_count += 1
                        
                        # Traccia statistiche
                        if new_tag not in updates_by_tag:
                            updates_by_tag[new_tag] = []
                        updates_by_tag[new_tag].append(old_tag)
                    else:
                        print(f"   ⚠️  Nessuna modifica per session {session_id}")
                        
                except Exception as e:
                    error_msg = f"Errore aggiornamento session {session_id}: {e}"
                    print(f"   ❌ {error_msg}")
                    errors.append(error_msg)
    
    if not dry_run:
        print(f"\n📊 RIEPILOGO MODIFICHE:")
        print(f"-" * 80)
        print(f"   ✅ Documenti aggiornati: {updated_count}")
        print(f"   ❌ Errori: {len(errors)}")
        
        if updates_by_tag:
            print(f"\n🏷️  TAG CORRETTI:")
            for tag, old_variants in sorted(updates_by_tag.items()):
                unique_variants = set(old_variants)
                print(f"      {tag}: {len(old_variants)} documenti ({len(unique_variants)} varianti)")
                for variant in sorted(unique_variants):
                    count = old_variants.count(variant)
                    print(f"         - '{variant}' ({count}x)")
    
    return {
        'updated': updated_count,
        'errors': len(errors),
        'error_messages': errors,
        'updates_by_tag': updates_by_tag
    }

def verify_fix(collection):
    """
    Verifica che tutti i tag siano ora MAIUSCOLI
    
    Args:
        collection: Collection MongoDB
        
    Returns:
        bool: True se tutti i tag sono MAIUSCOLI
    """
    print(f"\n✅ VERIFICA POST-CORREZIONE")
    print(f"=" * 80)
    
    remaining_lowercase = 0
    
    for doc in collection.find({'classification': {'$exists': True, '$ne': None}}, 
                               {'classification': 1, 'session_id': 1}):
        tag = doc.get('classification', '')
        if tag and tag != tag.upper():
            remaining_lowercase += 1
            print(f"   ⚠️  Ancora minuscolo: '{tag}' (session: {doc.get('session_id', 'unknown')})")
    
    if remaining_lowercase == 0:
        print(f"   🎉 TUTTI I TAG SONO ORA MAIUSCOLI!")
        return True
    else:
        print(f"   ⚠️  Rimangono {remaining_lowercase} tag con minuscole")
        return False

def main():
    """Main function"""
    print(f"\n{'=' * 80}")
    print(f"🔧 SCRIPT NORMALIZZAZIONE TAG MAIUSCOLO - MONGODB")
    print(f"{'=' * 80}")
    print(f"📅 Data esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Database: classificazioni")
    print(f"📦 Collection: humanitas_015007d9-d413-11ef-86a5-96000228e7fe")
    print(f"{'=' * 80}\n")
    
    # Connetti a MongoDB
    client = connect_to_mongodb()
    db = client['classificazioni']
    collection = db['humanitas_015007d9-d413-11ef-86a5-96000228e7fe']
    
    try:
        # FASE 1: Analisi
        stats = analyze_lowercase_tags(collection)
        
        if stats['mixed_case'] == 0:
            print(f"\n✅ NESSUN TAG DA CORREGGERE - Tutti già MAIUSCOLI!")
            return
        
        # FASE 2: Dry run (simulazione)
        print(f"\n{'=' * 80}")
        print(f"⚠️  TROVATI {stats['mixed_case']} TAG DA CORREGGERE")
        print(f"{'=' * 80}")
        
        input(f"\n⏸️  Premi INVIO per vedere la SIMULAZIONE delle modifiche...")
        fix_lowercase_tags(collection, dry_run=True)
        
        # FASE 3: Conferma utente
        print(f"\n{'=' * 80}")
        risposta = input(f"\n❓ Vuoi APPLICARE le modifiche al database? (sì/no): ").strip().lower()
        
        if risposta not in ['sì', 'si', 's', 'yes', 'y']:
            print(f"\n❌ Operazione ANNULLATA dall'utente")
            print(f"ℹ️  Nessuna modifica è stata applicata al database")
            return
        
        # FASE 4: Applicazione modifiche
        print(f"\n{'=' * 80}")
        print(f"🚀 APPLICAZIONE MODIFICHE IN CORSO...")
        print(f"{'=' * 80}")
        
        result = fix_lowercase_tags(collection, dry_run=False)
        
        # FASE 5: Verifica finale
        if result['updated'] > 0:
            verify_fix(collection)
        
        # FASE 6: Report finale
        print(f"\n{'=' * 80}")
        print(f"📊 REPORT FINALE")
        print(f"{'=' * 80}")
        print(f"   ✅ Documenti aggiornati: {result['updated']}")
        print(f"   ❌ Errori: {result['errors']}")
        
        if result['errors'] > 0:
            print(f"\n⚠️  ERRORI RISCONTRATI:")
            for error in result['error_messages']:
                print(f"      - {error}")
        
        print(f"\n{'=' * 80}")
        print(f"✅ SCRIPT COMPLETATO")
        print(f"{'=' * 80}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Script interrotto dall'utente")
        print(f"ℹ️  Alcune modifiche potrebbero essere state applicate")
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        print(f"\n🔌 Connessione MongoDB chiusa")

if __name__ == '__main__':
    main()
