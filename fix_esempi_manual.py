#!/usr/bin/env python3
"""
Fix manuale degli esempi nel database TAG

Autore: Valerio Bignardi
Data: 2025-09-01
"""

import mysql.connector

# Esempi corretti da inserire manualmente
ESEMPI_CORRETTI = {
    'indicazioni_stradali': '''user: "Come faccio a raggiungervi? Parto da Milano via corso buenos aires 2"
assistant: {"confidence": 1, "motivation": "Informazioni su come raggiungere la struttura", "predicted_label": "indicazioni_stradali"}''',

    'cambio_email': '''user: "Devo cambiare la mail con cui mi sono registrato al portale"
assistant: {"confidence": 1, "motivation": "Cambio delle informazioni anagrafiche richiesto", "predicted_label": "cambio_anagrafica"}''',
    
    'Problema_accesso_portale': '''user: "Quando inserisco il CF dice che non è valido"
assistant: {"confidence": 1, "motivation": "Problema tecnico di accesso al portale online", "predicted_label": "problema_accesso_portale"}''',
    
    'accesso_portale': '''user: "Non mi fa registrare al portale, dice codice fiscale errato"
assistant: {"confidence": 1, "motivation": "Problema tecnico di accesso al portale online", "predicted_label": "problema_accesso_portale"}''',
    
    'ID_Referto': '''user: "Non trovo l'id per scaricare il referto"
assistant: {"confidence": 1, "motivation": "Problema ID per ritiri referti e cartella clinica", "predicted_label": "problema_ritiro_referti_cartella_clinica"}''',
    
    'convenzioni_viaggio': '''user: "Ci sono convenzioni con hotel per assistere un ricoverato?"
assistant: {"confidence": 1, "motivation": "Richiesta info convenzioni viaggio", "predicted_label": "convenzioni_viaggio"}'''
}

def fix_examples():
    """Fix manuale degli esempi"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Valerio220693!',
            database='TAG'
        )
        
        cursor = connection.cursor()
        tenant_id = '015007d9-d413-11ef-86a5-96000228e7fe'
        
        print("🔧 Correzione manuale degli esempi...")
        
        for esempio_name, contenuto_corretto in ESEMPI_CORRETTI.items():
            cursor.execute("""
                UPDATE esempi 
                SET esempio_content = %s 
                WHERE tenant_id = %s AND esempio_name = %s
            """, (contenuto_corretto, tenant_id, esempio_name))
            
            print(f"✅ Corretto: {esempio_name}")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("\\n✅ Correzione completata!")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    fix_examples()
