import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { Tenant, TenantContextType, PromptStatus } from '../types/Tenant';
import { apiService } from '../services/apiService';

const TenantContext = createContext<TenantContextType | undefined>(undefined);

export const useTenant = (): TenantContextType => {
  const context = useContext(TenantContext);
  if (!context) {
    throw new Error('useTenant must be used within a TenantProvider');
  }
  return context;
};

interface TenantProviderProps {
  children: ReactNode;
}

export const TenantProvider: React.FC<TenantProviderProps> = ({ children }) => {
  const [selectedTenant, setSelectedTenant] = useState<Tenant | null>(null);
  const [availableTenants, setAvailableTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promptStatus, setPromptStatus] = useState<PromptStatus | null>(null);

  // Funzione per ricaricare la lista dei tenant
  const refreshTenants = useCallback(async () => {
    try {
      setError(null);
      
      const tenants = await apiService.getTenants();
      setAvailableTenants(tenants);
      
    } catch (err) {
      console.error('❌ [TenantContext] Errore in refreshTenants:', err);
      setError('Errore nel ricaricamento dei tenant');
    }
  }, []);

  // Funzione per ricaricare lo stato dei prompt
  const refreshPromptStatus = useCallback(async () => {
    if (!selectedTenant) {
      setPromptStatus(null);
      return;
    }

    try {
      const status = await apiService.checkPromptStatus(selectedTenant.tenant_id);
      setPromptStatus(status);
    } catch (err) {
      console.error('❌ [TenantContext] Errore in refreshPromptStatus:', err);
      // Impostiamo uno stato di fallback che blocca l'operazione
      setPromptStatus({
        canOperate: false,
        requiredPrompts: [
          {
            name: 'System Prompt',
            type: 'system',
            description: 'Prompt di sistema per la classificazione',
            exists: false
          },
          {
            name: 'User Template',
            type: 'user',
            description: 'Template per l\'input utente',
            exists: false
          }
        ],
        missingCount: 2
      });
    }
  }, [selectedTenant]);

  // Caricamento iniziale dei tenant
  useEffect(() => {
    const loadTenants = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const tenants = await apiService.getTenants();
        setAvailableTenants(tenants);
        
        // Auto-select first active tenant or first tenant
        const defaultTenant = tenants.find(t => t.is_active) || tenants[0];
        
        if (defaultTenant && !selectedTenant) {
          setSelectedTenant(defaultTenant);
        }
        
      } catch (err) {
        console.error('❌ [TenantContext] Errore in loadTenants:', err);
        setError('Errore nel caricamento dei tenant');
      } finally {
        setLoading(false);
      }
    };

    loadTenants();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Questa dipendenza vuota è intenzionale - vogliamo caricare i tenant solo al mount

  // Caricamento dello stato dei prompt quando cambia il tenant selezionato
  useEffect(() => {
    refreshPromptStatus();
  }, [refreshPromptStatus]);

  const contextValue: TenantContextType = {
    selectedTenant,
    availableTenants,
    setSelectedTenant,
    loading,
    error,
    promptStatus,
    refreshPromptStatus,
    refreshTenants
  };

  return (
    <TenantContext.Provider value={contextValue}>
      {children}
    </TenantContext.Provider>
  );
};
