#!/usr/bin/env python3
"""
File: tenant.py
Autore: GitHub Copilot & Valerio Bignardi
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
        Crea un oggetto Tenant dal UUID risolvendo TUTTE le info dal database.
        
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
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MySql'))
            from connettore import MySqlConnettore
            
            remote = MySqlConnettore()
            
            query = """
            SELECT tenant_id, tenant_name, tenant_database, tenant_status
            FROM common.tenants 
            WHERE tenant_id = %s AND tenant_status = 1
            """
            
            result = remote.esegui_query(query, (tenant_uuid,))
            remote.disconnetti()
            
            if not result or len(result) == 0:
                raise ValueError(f"Tenant UUID '{tenant_uuid}' non trovato nel database")
                
            tenant_id, tenant_name, tenant_slug, tenant_status = result[0]
            
            # Genera database name completo
            tenant_database = f"{tenant_slug}_{tenant_id.replace('-', '_')}"
            
            print(f"🎯 Tenant risolto: {tenant_name} ({tenant_slug}) UUID={tenant_id}")
            
            return cls(
                tenant_id=tenant_id,
                tenant_name=tenant_name, 
                tenant_slug=tenant_slug,
                tenant_database=tenant_database,
                tenant_status=tenant_status
            )
            
        except Exception as e:
            print(f"❌ Errore risoluzione tenant {tenant_uuid}: {e}")
            raise RuntimeError(f"Impossibile risolvere tenant {tenant_uuid}: {e}")
    
    @classmethod  
    def from_slug(cls, tenant_slug: str) -> 'Tenant':
        """
        Crea un oggetto Tenant dal slug risolvendo TUTTE le info dal database.
        
        Args:
            tenant_slug: Slug del tenant (es: 'wopta')
            
        Returns:
            Tenant: Oggetto tenant popolato
        """
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MySql'))
            from connettore import MySqlConnettore
            
            remote = MySqlConnettore()
            
            query = """
            SELECT tenant_id, tenant_name, tenant_database, tenant_status
            FROM common.tenants 
            WHERE tenant_database = %s AND tenant_status = 1
            """
            
            result = remote.esegui_query(query, (tenant_slug,))
            remote.disconnetti()
            
            if not result or len(result) == 0:
                raise ValueError(f"Tenant slug '{tenant_slug}' non trovato nel database")
                
            tenant_id, tenant_name, tenant_database_name, tenant_status = result[0]
            
            # Genera database name completo
            tenant_database = f"{tenant_slug}_{tenant_id.replace('-', '_')}"
            
            print(f"🎯 Tenant risolto da slug: {tenant_name} ({tenant_slug}) UUID={tenant_id}")
            
            return cls(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                tenant_slug=tenant_slug, 
                tenant_database=tenant_database,
                tenant_status=tenant_status
            )
            
        except Exception as e:
            print(f"❌ Errore risoluzione tenant slug {tenant_slug}: {e}")
            raise RuntimeError(f"Impossibile risolvere tenant slug {tenant_slug}: {e}")
    
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
