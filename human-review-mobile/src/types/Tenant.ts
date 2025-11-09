export interface Tenant {
  tenant_id: string;
  tenant_name: string;
  tenant_slug: string;
  is_active?: boolean;
}

export interface PromptStatus {
  canOperate: boolean;
  requiredPrompts: Array<{
    name: string;
    type: string;
    description: string;
    exists: boolean;
  }>;
  missingCount: number;
}

export interface TenantContextType {
  selectedTenant: Tenant | null;
  availableTenants: Tenant[];
  setSelectedTenant: (t: Tenant | null) => void;
  loading: boolean;
  error: string | null;
  promptStatus: PromptStatus | null;
  refreshPromptStatus: () => Promise<void>;
  refreshTenants: () => Promise<void>;
}

