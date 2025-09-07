#!/usr/bin/env python3
"""
File: mock_tenant_manager.py
Autore: Valerio Bignardi
Data: 2025-09-07
Descrizione: Mock TenantManager per test e compatibilità con codice esistente

Mock semplice per sostituire TenantManager quando non è disponibile
o per test che non necessitano del database reale.
"""

from typing import List, Optional
from Utils.tenant import Tenant


class MockTenantManager:
    """
    Mock del TenantManager per test e compatibilità
    
    Scopo: Fornire una implementazione base per test senza database
    Input: config_path opzionale (ignorato nel mock)
    Output: Oggetti Tenant mock per test
    Data ultima modifica: 2025-09-07
    """
    
    def __init__(self, config_path: str = None):
        """
        Inizializza il mock TenantManager
        
        Args:
            config_path: Path configurazione (ignorato nel mock)
        """
        self.config_path = config_path
        
        # Tenant di test hardcoded
        self._mock_tenants = [
            Tenant(
                tenant_id="16c222a9-f293-11ef-9315-96000228e7fe",
                tenant_name="Humanitas",
                tenant_slug="humanitas",
                tenant_database="humanitas",
                tenant_status=1
            ),
            Tenant(
                tenant_id="26c333b9-f394-11ef-9316-96000228e7ff",
                tenant_name="Wopta",
                tenant_slug="wopta", 
                tenant_database="wopta",
                tenant_status=1
            ),
            Tenant(
                tenant_id="36c444c9-f495-11ef-9317-96000228e800",
                tenant_name="TestTenant",
                tenant_slug="test",
                tenant_database="test",
                tenant_status=1
            )
        ]
    
    def get_all_tenants(self) -> List[Tenant]:
        """
        Restituisce tutti i tenant mock
        
        Scopo: Fornire lista tenant per test
        Input: None
        Output: Lista oggetti Tenant mock
        Data ultima modifica: 2025-09-07
        
        Returns:
            List[Tenant]: Lista tenant di test
        """
        return self._mock_tenants.copy()
    
    def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """
        Trova tenant per ID
        
        Scopo: Recuperare tenant specifico per ID
        Input: tenant_id (stringa o intero)
        Output: Oggetto Tenant o None se non trovato
        Data ultima modifica: 2025-09-07
        
        Args:
            tenant_id: ID del tenant (può essere stringa UUID o intero)
            
        Returns:
            Optional[Tenant]: Tenant trovato o None
        """
        # Supporta sia string UUID che integer ID
        if isinstance(tenant_id, int):
            # Se è un intero, mappa al primo tenant per semplicità
            if tenant_id == 1:
                return self._mock_tenants[0]  # Humanitas
            elif tenant_id == 2:
                return self._mock_tenants[1]  # Wopta
            elif tenant_id == 3:
                return self._mock_tenants[2]  # TestTenant
            else:
                return None
        
        # Se è string UUID, cerca per tenant_id
        for tenant in self._mock_tenants:
            if tenant.tenant_id == tenant_id:
                return tenant
        
        return None
    
    def get_tenant_by_slug(self, tenant_slug: str) -> Optional[Tenant]:
        """
        Trova tenant per slug
        
        Scopo: Recuperare tenant specifico per slug
        Input: tenant_slug
        Output: Oggetto Tenant o None se non trovato
        Data ultima modifica: 2025-09-07
        
        Args:
            tenant_slug: Slug del tenant
            
        Returns:
            Optional[Tenant]: Tenant trovato o None
        """
        for tenant in self._mock_tenants:
            if tenant.tenant_slug == tenant_slug:
                return tenant
        
        return None


# Alias per compatibilità
TenantManager = MockTenantManager
