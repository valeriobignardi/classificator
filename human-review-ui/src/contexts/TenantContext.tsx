import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Tenant, TenantContextType } from '../types/Tenant';
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
        console.error('Error loading tenants:', err);
        setError('Errore nel caricamento dei tenant');
      } finally {
        setLoading(false);
      }
    };

    loadTenants();
  }, []);

  const contextValue: TenantContextType = {
    selectedTenant,
    availableTenants,
    setSelectedTenant,
    loading,
    error
  };

  return (
    <TenantContext.Provider value={contextValue}>
      {children}
    </TenantContext.Provider>
  );
};
