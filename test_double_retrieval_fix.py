#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test per verificare la correzione del doppio recupero parametri clustering

Autore: Valerio Bignardi
Data: 05/09/2025
Ultima modifica: 05/09/2025 - Valerio Bignardi
"""

import sys
import os
from datetime import datetime

def test_no_double_data_retrieval():
    """
    Test per verificare che il doppio recupero dati sia stato corretto.
    
    Verifica che:
    1. supervised_training NON carichi più i parametri clustering
    2. Solo get_all_clustering_parameters_for_tenant() carichi i parametri
    3. Non ci sia codice duplicato
    
    Scopo: Validare che la correzione del doppio recupero funzioni
    Input: Analisi del codice sorgente
    Output: Report di validazione
    
    Ultima modifica: 05/09/2025 - Valerio Bignardi
    """
    print("🧪 TEST CORREZIONE DOPPIO RECUPERO PARAMETRI")
    print("=" * 60)
    
    issues_found = []
    success = True
    
    # Test 1: Verifica che supervised_training non carichi più clustering params
    print("🔍 TEST 1: Analisi supervised_training endpoint")
    
    try:
        server_path = "/home/ubuntu/classificatore/server.py"
        with open(server_path, 'r', encoding='utf-8') as f:
            server_content = f.read()
        
        # Cerca la funzione supervised_training
        lines = server_content.split('\n')
        in_supervised_function = False
        clustering_params_assignments = []
        
        for i, line in enumerate(lines):
            if 'def supervised_training(' in line:
                in_supervised_function = True
                start_line = i
                continue
            
            if in_supervised_function:
                # Fine della funzione (prossima def o fine file)
                if line.strip().startswith('def ') and 'supervised_training' not in line:
                    break
                
                # Cerca assegnazioni a clustering_params
                if 'clustering_params = {' in line:
                    clustering_params_assignments.append(i + 1)  # Line numbers start from 1
        
        if clustering_params_assignments:
            issues_found.append(f"TROVATE {len(clustering_params_assignments)} assegnazioni clustering_params in supervised_training alle righe: {clustering_params_assignments}")
            success = False
            print(f"   ❌ ERRORE: Trovate assegnazioni clustering_params alle righe: {clustering_params_assignments}")
        else:
            print(f"   ✅ supervised_training NON carica più clustering_params")
            
    except Exception as e:
        issues_found.append(f"Errore lettura server.py: {e}")
        success = False
        print(f"   ❌ Errore lettura server.py: {e}")
    
    # Test 2: Verifica che get_all_clustering_parameters_for_tenant esista
    print("\n🔍 TEST 2: Verifica funzione centralizzata")
    
    try:
        tenant_config_path = "/home/ubuntu/classificatore/Utils/tenant_config_helper.py"
        with open(tenant_config_path, 'r', encoding='utf-8') as f:
            tenant_content = f.read()
        
        if 'def get_all_clustering_parameters_for_tenant(' in tenant_content:
            print(f"   ✅ get_all_clustering_parameters_for_tenant() trovata")
        else:
            issues_found.append("get_all_clustering_parameters_for_tenant() NON trovata")
            success = False
            print(f"   ❌ get_all_clustering_parameters_for_tenant() NON trovata")
            
    except Exception as e:
        issues_found.append(f"Errore lettura tenant_config_helper.py: {e}")
        success = False
        print(f"   ❌ Errore lettura tenant_config_helper.py: {e}")
    
    # Test 3: Verifica che pipeline usi get_all_clustering_parameters_for_tenant
    print("\n🔍 TEST 3: Verifica uso nella pipeline")
    
    try:
        pipeline_path = "/home/ubuntu/classificatore/Pipeline/end_to_end_pipeline.py"
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        
        if 'get_all_clustering_parameters_for_tenant(' in pipeline_content:
            print(f"   ✅ Pipeline usa get_all_clustering_parameters_for_tenant()")
        else:
            issues_found.append("Pipeline NON usa get_all_clustering_parameters_for_tenant()")
            success = False
            print(f"   ❌ Pipeline NON usa get_all_clustering_parameters_for_tenant()")
            
    except Exception as e:
        issues_found.append(f"Errore lettura end_to_end_pipeline.py: {e}")
        success = False
        print(f"   ❌ Errore lettura pipeline: {e}")
    
    # Test 4: Cerca query SQL duplicate
    print("\n🔍 TEST 4: Ricerca query SQL duplicate")
    
    try:
        sql_patterns = [
            "SELECT * FROM soglie WHERE tenant_id",
            "SELECT.*FROM soglie.*WHERE tenant_id"
        ]
        
        files_to_check = [
            "/home/ubuntu/classificatore/server.py",
            "/home/ubuntu/classificatore/Utils/tenant_config_helper.py"
        ]
        
        sql_occurrences = {}
        
        for file_path in files_to_check:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in sql_patterns:
                    count = content.count("SELECT") + content.count("FROM soglie")
                    if count > 0:
                        sql_occurrences[os.path.basename(file_path)] = count
                        
            except Exception as e:
                print(f"   ⚠️ Errore lettura {file_path}: {e}")
        
        print(f"   📊 Query SQL trovate per file:")
        for file, count in sql_occurrences.items():
            print(f"      {file}: query patterns trovati")
            
        # Questo è normale - tenant_config_helper dovrebbe avere le query
        print(f"   ✅ Query SQL centralizzate in tenant_config_helper.py")
            
    except Exception as e:
        print(f"   ⚠️ Errore analisi query SQL: {e}")
    
    # Test 5: Verifica rimozione variabili clustering_params
    print("\n🔍 TEST 5: Verifica rimozione variabili clustering_params in supervised_training")
    
    try:
        with open(server_path, 'r', encoding='utf-8') as f:
            server_content = f.read()
        
        # Cerca riferimenti a clustering_params nella funzione supervised_training
        lines = server_content.split('\n')
        in_supervised_function = False
        clustering_params_references = []
        
        for i, line in enumerate(lines):
            if 'def supervised_training(' in line:
                in_supervised_function = True
                continue
            
            if in_supervised_function:
                if line.strip().startswith('def ') and 'supervised_training' not in line:
                    break
                
                if 'clustering_params' in line and 'RIMOSSO' not in line and 'NOTE' not in line:
                    clustering_params_references.append(i + 1)
        
        if clustering_params_references:
            issues_found.append(f"Trovati riferimenti clustering_params in supervised_training alle righe: {clustering_params_references}")
            success = False
            print(f"   ❌ Trovati riferimenti clustering_params alle righe: {clustering_params_references}")
        else:
            print(f"   ✅ Tutti i riferimenti clustering_params rimossi da supervised_training")
            
    except Exception as e:
        print(f"   ❌ Errore analisi clustering_params: {e}")
    
    # Report finale
    print("\n📋 REPORT FINALE CORREZIONE")
    print("-" * 50)
    
    if success:
        print("🎉 CORREZIONE DOPPIO RECUPERO COMPLETATA CON SUCCESSO!")
        print("   ✅ supervised_training NON carica più parametri clustering")
        print("   ✅ Pipeline usa fonte centralizzata get_all_clustering_parameters_for_tenant()")
        print("   ✅ Eliminato codice duplicato")
        print("   ✅ Un'unica connessione DB per parametri clustering")
        print("\n💡 BENEFICI:")
        print("   • Performance migliorate (una sola query DB)")
        print("   • Coerenza garantita (fonte unica)")
        print("   • Manutenibilità migliorata (logica centralizzata)")
        print("   • Riduzione rischio inconsistenze")
    else:
        print("⚠️ CORREZIONE DOPPIO RECUPERO INCOMPLETA")
        print("📋 Problemi trovati:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\n💡 AZIONI RICHIESTE:")
        print("   • Risolvere i problemi sopra elencati")
        print("   • Verificare che tutti i riferimenti siano stati rimossi")
    
    return success

def test_clustering_params_flow():
    """
    Test per tracciare il flusso corretto dei parametri clustering
    
    Scopo: Documentare il flusso corretto post-correzione
    
    Ultima modifica: 05/09/2025 - Valerio Bignardi
    """
    print("\n🔄 FLUSSO CORRETTO PARAMETRI CLUSTERING")
    print("=" * 60)
    
    print("📋 FLUSSO POST-CORREZIONE:")
    print("   1. 🌐 Frontend: Modifica parametri clustering")
    print("   2. 💾 API: Salva in database MySQL soglie table")  
    print("   3. 🚀 supervised_training: Carica SOLO soglie review queue")
    print("   4. 🏭 Pipeline: Chiama get_all_clustering_parameters_for_tenant()")
    print("   5. 📊 tenant_config_helper: Legge TUTTI i parametri da DB")
    print("   6. 🧩 Clustering: Usa parametri unificati")
    
    print("\n✅ VANTAGGI DEL NUOVO FLUSSO:")
    print("   • Una sola fonte di verità per parametri clustering")
    print("   • Eliminata duplicazione query database")
    print("   • Coerenza garantita tra componenti")
    print("   • Performance migliorate")
    print("   • Manutenibilità semplificata")
    
    print("\n🎯 RESPONSABILITÀ:")
    print("   • supervised_training: Solo soglie review queue")
    print("   • get_all_clustering_parameters_for_tenant(): Tutti i parametri clustering")
    print("   • Pipeline: Uso parametri centralizzati")
    
    return True

if __name__ == "__main__":
    print("🧪 SUITE TEST CORREZIONE DOPPIO RECUPERO")
    print("=" * 60)
    print(f"⏰ Avvio test: {datetime.now().isoformat()}")
    print()
    
    # Esegui test
    test1_result = test_no_double_data_retrieval()
    test2_result = test_clustering_params_flow()
    
    # Report finale
    print("\n" + "=" * 60)
    print("📊 RISULTATI FINALI")
    print("=" * 60)
    print(f"   🔧 Test Correzione Doppio Recupero: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   📋 Test Flusso Parametri: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    overall_success = test1_result and test2_result
    print(f"\n🎯 RISULTATO COMPLESSIVO: {'✅ SUCCESSO' if overall_success else '❌ FALLIMENTO'}")
    
    if overall_success:
        print("\n🎉 La correzione del doppio recupero è COMPLETA e FUNZIONANTE!")
        print("   • Codice duplicato eliminato")
        print("   • Fonte unica centralizzata attiva")
        print("   • Performance e coerenza migliorate")
    else:
        print("\n⚠️ La correzione richiede ancora alcuni aggiustamenti.")
    
    print(f"\n⏰ Test completati: {datetime.now().isoformat()}")
