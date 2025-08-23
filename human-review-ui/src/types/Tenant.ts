export interface Tenant {
  tenant_id: string;
  nome: string;
  is_active?: boolean;
}

export interface TenantContextType {
  selectedTenant: Tenant | null;
  availableTenants: Tenant[];
  setSelectedTenant: (tenant: Tenant | null) => void;
  loading: boolean;
  error: string | null;
}
