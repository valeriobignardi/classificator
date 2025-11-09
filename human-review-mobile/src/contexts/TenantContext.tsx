import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { Tenant, TenantContextType, PromptStatus } from '../types/Tenant';

const TenantContext = createContext<TenantContextType | undefined>(undefined);

export const useTenant = (): TenantContextType => {
  const ctx = useContext(TenantContext);
  if (!ctx) throw new Error('useTenant must be used within a TenantProvider');
  return ctx;
};

export const TenantProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedTenant, setSelectedTenant] = useState<Tenant | null>(null);
  const [availableTenants, setAvailableTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [promptStatus, setPromptStatus] = useState<PromptStatus | null>(null);

  const refreshTenants = useCallback(async () => {
    setError(null);
    const tenants = await api.getTenants();
    setAvailableTenants(tenants);
    if (!selectedTenant && tenants.length > 0) {
      const def = tenants.find(t => t.is_active) || tenants[0];
      setSelectedTenant(def);
    }
  }, [selectedTenant]);

  const refreshPromptStatus = useCallback(async () => {
    // TODO: integrare endpoint prompt status quando disponibile lato mobile
    setPromptStatus(null);
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        await refreshTenants();
      } catch (e: any) {
        setError(e?.message || 'Errore caricamento tenants');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <TenantContext.Provider value={{
      selectedTenant,
      availableTenants,
      setSelectedTenant,
      loading,
      error,
      promptStatus,
      refreshPromptStatus,
      refreshTenants,
    }}>
      {children}
    </TenantContext.Provider>
  );
};

