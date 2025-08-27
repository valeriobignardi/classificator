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

  // Funzione per ricaricare lo stato dei prompt
  const refreshPromptStatus = useCallback(async () => {
    console.log('🔍 [DEBUG] TenantContext.refreshPromptStatus() - Avvio');
    console.log('🔍 [DEBUG] selectedTenant:', selectedTenant);
    
    if (!selectedTenant) {
      console.log('🔍 [DEBUG] Nessun tenant selezionato, imposto promptStatus null');
      setPromptStatus(null);
      return;
    }

    try {
      console.log(`🔍 [DEBUG] Chiamo apiService.checkPromptStatus(${selectedTenant.tenant_id})`);
      const status = await apiService.checkPromptStatus(selectedTenant.tenant_id);
      console.log('✅ [DEBUG] Ricevuto status dai prompt:', status);
      setPromptStatus(status);
    } catch (err) {
      console.error('❌ [DEBUG] Errore in refreshPromptStatus:', err);
      console.error('❌ [DEBUG] Tipo errore:', typeof err);
      console.error('❌ [DEBUG] Stack:', err instanceof Error ? err.stack : 'No stack');
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
      console.log('🔍 [DEBUG] TenantContext.loadTenants() - Avvio caricamento tenant');
      try {
        setLoading(true);
        setError(null);
        
        console.log('🔍 [DEBUG] Chiamo apiService.getTenants()...');
        const tenants = await apiService.getTenants();
        console.log('✅ [DEBUG] Ricevuti tenant:', tenants.length, 'elementi');
        console.log('✅ [DEBUG] Primi 3 tenant:', tenants.slice(0, 3));
        
        setAvailableTenants(tenants);
        
        // Auto-select first active tenant or first tenant
        const defaultTenant = tenants.find(t => t.is_active) || tenants[0];
        console.log('🔍 [DEBUG] Default tenant selezionato:', defaultTenant);
        
        if (defaultTenant && !selectedTenant) {
          console.log('🔍 [DEBUG] Imposto selectedTenant:', defaultTenant.nome);
          setSelectedTenant(defaultTenant);
        }
        
      } catch (err) {
        console.error('❌ [DEBUG] Errore in loadTenants:', err);
        console.error('❌ [DEBUG] Tipo errore:', typeof err);
        console.error('❌ [DEBUG] Stack:', err instanceof Error ? err.stack : 'No stack');
        setError('Errore nel caricamento dei tenant');
      } finally {
        console.log('🔍 [DEBUG] loadTenants completato, setLoading(false)');
        setLoading(false);
      }
    };

    console.log('🔍 [DEBUG] TenantContext useEffect - Avvio loadTenants');
    loadTenants();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Questa dipendenza vuota è intenzionale - vogliamo caricare i tenant solo al mount

  // Caricamento dello stato dei prompt quando cambia il tenant selezionato
  useEffect(() => {
    refreshPromptStatus();
  }, [selectedTenant]); // ✅ Dipende solo da selectedTenant, non da refreshPromptStatus

  const contextValue: TenantContextType = {
    selectedTenant,
    availableTenants,
    setSelectedTenant,
    loading,
    error,
    promptStatus,
    refreshPromptStatus
  };

  return (
    <TenantContext.Provider value={contextValue}>
      {children}
    </TenantContext.Provider>
  );
};
