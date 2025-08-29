#!/usr/bin/env python3
"""
File: tenant.py
Autore: GitHub Copilot             from TagDatabase.tag_database_connector import TagDatabaseConnector
            
            # Usa funzione bootstrap per risoluzione tenant
            tag_connector = TagDatabaseConnector.create_for_tenant_resolution()
            tag_connector.connetti()lerio Bignardi
Data creazione: 2025-08-26
Ultima modifica: 2025-08-26

Classe Tenant per centralizzare tutte le informazioni del tenant
ed eliminare le conversioni UUID ↔ slug sparse nel codice.

Scopo: 
- Risolvere UNA VOLTA le informazioni del tenant dal database
- Fornire un oggetto immutabile con tutte le info necessarie
- Eliminare performance waste da conversioni multiple
- Semplificare debugging e maintenance
"""

import sys
import os
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)  # Immutabile per evitare modifiche accidentali
class Tenant:
    """
    Classe immutabile che contiene tutte le informazioni di un tenant.
    
    Viene popolata UNA VOLTA all'inizializzazione e poi passata 
    a tutti i componenti che ne hanno bisogno.
    
    Attributi:
        tenant_id (str): UUID del tenant (es: '16c222a9-f293-11ef-9315-96000228e7fe')
        tenant_name (str): Nome leggibile (es: 'Wopta')
        tenant_slug (str): Slug per database (es: 'wopta')
        tenant_database (str): Nome database completo (es: 'wopta_16c222a9...')
        tenant_status (int): Status del tenant (1 = attivo)
    """
    tenant_id: str          # UUID univoco
    tenant_name: str        # Nome leggibile
    tenant_slug: str        # Slug per database/API
    tenant_database: str    # Database name completo  
    tenant_status: int      # Status (1 = attivo)
    
    @classmethod
    def from_uuid(cls, tenant_uuid: str) -> 'Tenant':
        """
        Crea un oggetto Tenant dal UUID risolvendo TUTTE le info dal database TAG LOCALE.
        
        Args:
            tenant_uuid: UUID del tenant
            
        Returns:
            Tenant: Oggetto tenant popolato
            
        Raises:
            ValueError: Se il tenant non viene trovato
            RuntimeError: Se ci sono errori di connessione database
        """
        if not cls._is_valid_uuid(tenant_uuid):
            raise ValueError(f"UUID non valido: {tenant_uuid}")
            
        try:
            # Import dinamico per evitare circular imports
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TagDatabase'))
            from tag_database_connector import TagDatabaseConnector
            
            # Usa funzione bootstrap per risoluzione tenant
            tag_connector = TagDatabaseConnector.create_for_tenant_resolution()
            tag_connector.connetti()
            
            query = """
            SELECT tenant_id, tenant_name, tenant_slug, is_active
            FROM tenants 
            WHERE tenant_id = %s AND is_active = 1
            """
            
            result = tag_connector.esegui_query(query, (tenant_uuid,))
            tag_connector.disconnetti()
            
            if not result or len(result) == 0:
                error_msg = f"❌ ERRORE: Tenant UUID '{tenant_uuid}' non trovato nel database TAG locale"
                print(error_msg)
                raise ValueError(error_msg)
                
            tenant_id, tenant_name, tenant_slug, is_active = result[0]
            
            # Il tenant_database corrisponde al tenant_slug per golvalerio
            tenant_database = tenant_slug
            
            print(f"✅ Tenant risolto (DB TAG locale): {tenant_name} ({tenant_slug}) UUID={tenant_id}")
            
            return cls(
                tenant_id=tenant_id,
                tenant_name=tenant_name, 
                tenant_slug=tenant_slug,
                tenant_database=tenant_database,
                tenant_status=is_active
            )
            
        except Exception as e:
            error_msg = f"❌ ERRORE risoluzione tenant {tenant_uuid}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    @classmethod  
    def from_slug(cls, tenant_slug: str) -> 'Tenant':
        """
        Crea un oggetto Tenant dal slug risolvendo TUTTE le info dal database TAG LOCALE.
        
        Args:
            tenant_slug: Slug del tenant (es: 'golvalerio')
            
        Returns:
            Tenant: Oggetto tenant popolato
        """
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TagDatabase'))
            from tag_database_connector import TagDatabaseConnector
            
            # Usa funzione bootstrap per risoluzione tenant
            tag_connector = TagDatabaseConnector.create_for_tenant_resolution()
            tag_connector.connetti()
            
            query = """
            SELECT tenant_id, tenant_name, tenant_slug, is_active
            FROM tenants 
            WHERE tenant_slug = %s AND is_active = 1
            """
            
            result = tag_connector.esegui_query(query, (tenant_slug,))
            tag_connector.disconnetti()
            
            if not result or len(result) == 0:
                error_msg = f"❌ ERRORE: Tenant slug '{tenant_slug}' non trovato nel database TAG locale"
                print(error_msg)
                raise ValueError(error_msg)
                
            tenant_id, tenant_name, tenant_database_name, is_active = result[0]
            
            # Il tenant_database corrisponde al tenant_slug per golvalerio 
            tenant_database = tenant_slug
            
            print(f"✅ Tenant risolto da slug (DB TAG locale): {tenant_name} ({tenant_slug}) UUID={tenant_id}")
            
            return cls(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                tenant_slug=tenant_slug, 
                tenant_database=tenant_database,
                tenant_status=is_active
            )
            
        except Exception as e:
            error_msg = f"❌ ERRORE risoluzione tenant slug {tenant_slug}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    @staticmethod
    def _is_valid_uuid(uuid_string: str) -> bool:
        """Verifica se una stringa è un UUID valido"""
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        return bool(re.match(uuid_pattern, uuid_string))
    
    def __str__(self) -> str:
        """Rappresentazione string leggibile"""
        return f"Tenant({self.tenant_name}/{self.tenant_slug})"
    
    def __repr__(self) -> str:
        """Rappresentazione per debugging"""
        return f"Tenant(id={self.tenant_id}, name={self.tenant_name}, slug={self.tenant_slug})"
