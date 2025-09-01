#!/usr/bin/env python3
"""
Fix degli esempi nel database TAG per correggere JSON malformati

Autore: Valerio Bignardi
Data: 2025-09-01
Scopo: Corregge gli esempi con JSON malformati nel database TAG
"""

import mysql.connector
import json
import re
from typing import Dict, List, Tuple


class EsempiDatabaseFixer:
    """
    Classe per correggere gli esempi con JSON malformati nel database TAG
    
    Scopo:
    - Connessione al database TAG
    - Lettura degli esempi corrotti
    - Correzione automatica dei JSON
    - Aggiornamento nel database
    """
    
    def __init__(self):
        """Inizializza la connessione al database"""
        self.connection = None
        self.tenant_id = '015007d9-d413-11ef-86a5-96000228e7fe'
        
    def connect(self) -> bool:
        """
        Stabilisce connessione al database TAG
        
        Returns:
            bool: True se connessione riuscita
        """
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='Valerio220693!',
                database='TAG'
            )
            print("✅ Connesso al database TAG")
            return True
        except Exception as e:
            print(f"❌ Errore connessione: {e}")
            return False
    
    def get_broken_examples(self) -> List[Tuple]:
        """
        Recupera tutti gli esempi dal database
        
        Returns:
            Lista di tuple (esempio_name, esempio_content)
        """
        if not self.connection:
            return []
            
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT esempio_name, esempio_content 
            FROM esempi 
            WHERE tenant_id = %s
        """, (self.tenant_id,))
        
        results = cursor.fetchall()
        cursor.close()
        
        print(f"📥 Recuperati {len(results)} esempi dal database")
        return results
    
    def fix_json_format(self, content: str) -> str:
        """
        Corregge il formato JSON nell'esempio
        
        Args:
            content: Contenuto dell'esempio da correggere
            
        Returns:
            Contenuto corretto con JSON valido
        """
        # Prova prima con \n letterale, poi con newline reale
        if '\\nassistant: ' in content:
            parts = content.split('\\nassistant: ')
        elif '\nassistant: ' in content:
            parts = content.split('\nassistant: ')
        else:
            print(f"⚠️ Formato non riconosciuto: {content[:50]}...")
            return content
            
        if len(parts) != 2:
            print(f"⚠️ Formato non riconosciuto: {content[:50]}...")
            return content
        
        user_part = parts[0]
        assistant_part = parts[1]
        
        # Rimuove eventuali newline alla fine
        assistant_part = assistant_part.strip()
        
        # Corregge problemi comuni nel JSON
        fixed_json = self._fix_common_json_issues(assistant_part)
        
        # Ricompone il contenuto con il separatore corretto
        separator = '\\nassistant: ' if '\\nassistant: ' in content else '\nassistant: '
        return f"{user_part}{separator}{fixed_json}"
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """
        Corregge problemi comuni nei JSON degli esempi
        
        Args:
            json_str: Stringa JSON da correggere
            
        Returns:
            JSON corretto
        """
        # Rimuove eventuali caratteri extra all'inizio
        json_str = json_str.strip()
        if json_str.startswith('{'):
            json_str = json_str[1:]  # Rimuove la prima parentesi graffa
            
        # Assicura che inizi con parentesi graffa
        if not json_str.startswith("'confidence'"):
            json_str = "{'confidence': 1, " + json_str
        else:
            json_str = "{" + json_str
            
        # Corregge virgolette mancanti nella motivation
        json_str = re.sub(r"'motivation':\s*([^'\"]+?),", r"'motivation': \"\\1\",", json_str)
        json_str = re.sub(r"'motivation':\s*\"([^\"]*?)\"([^,}]*),", r"'motivation': \"\\1\",", json_str)
        
        # Assicura che finisca correttamente
        if not json_str.endswith('}'):
            if json_str.endswith("'}"):
                pass  # È già corretto
            elif "'}" in json_str:
                # Trova l'ultima occorrenza e taglia lì
                last_quote_brace = json_str.rfind("'}")
                json_str = json_str[:last_quote_brace+2]
            else:
                # Aggiunge la chiusura mancante
                if json_str.endswith("'"):
                    json_str += "}"
                elif json_str.endswith("',"):
                    json_str = json_str[:-1] + "}"
                else:
                    json_str += "'}"
        
        # Pulizia finale
        json_str = json_str.replace(" '}", "}")
        json_str = re.sub(r'\s+', ' ', json_str)  # Rimuove spazi multipli
        
        return json_str
    
    def update_example(self, esempio_name: str, new_content: str):
        """
        Aggiorna un esempio nel database
        
        Args:
            esempio_name: Nome dell'esempio da aggiornare
            new_content: Nuovo contenuto corretto
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE esempi 
            SET esempio_content = %s 
            WHERE tenant_id = %s AND esempio_name = %s
        """, (new_content, self.tenant_id, esempio_name))
        
        self.connection.commit()
        cursor.close()
        print(f"✅ Aggiornato: {esempio_name}")
    
    def run_fix(self):
        """
        Esegue la correzione di tutti gli esempi
        """
        print("🔧 Avvio correzione esempi nel database TAG...")
        
        if not self.connect():
            return
        
        examples = self.get_broken_examples()
        
        print(f"\\n📝 Correzione di {len(examples)} esempi...")
        
        for esempio_name, esempio_content in examples:
            print(f"\\n🔍 Elaboro: {esempio_name}")
            print(f"   📥 Originale: {esempio_content[:100]}...")
            
            fixed_content = self.fix_json_format(esempio_content)
            print(f"   📤 Corretto:  {fixed_content[:100]}...")
            
            # Aggiorna nel database
            self.update_example(esempio_name, fixed_content)
        
        print("\\n✅ Correzione completata!")
        
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    fixer = EsempiDatabaseFixer()
    fixer.run_fix()
