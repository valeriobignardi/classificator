import React from 'react';
import {
  Drawer,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Chip,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
  IconButton,
  Divider
} from '@mui/material';
import {
  BusinessOutlined,
  ChevronLeft
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';

interface TenantSelectorProps {
  open: boolean;
  onToggle: () => void;
  drawerWidth?: number;
}

const TenantSelector: React.FC<TenantSelectorProps> = ({
  open,
  onToggle,
  drawerWidth = 280
}) => {
  const { selectedTenant, availableTenants, setSelectedTenant, loading, error } = useTenant();

  const handleSelectChange = (event: SelectChangeEvent<string>) => {
    const tenantId = event.target.value;
    const tenant = availableTenants.find(t => t.tenant_id === tenantId);
    if (tenant) {
      setSelectedTenant(tenant);
    }
  };

  return (
    <Drawer
      variant="persistent"
      open={open}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          border: '1px solid rgba(0, 0, 0, 0.12)',
          borderRight: '2px solid rgba(0, 0, 0, 0.12)',
        },
      }}
    >
      <Box sx={{ overflow: 'auto', display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header con pulsante di chiusura */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between', 
          p: 2,
          borderBottom: '1px solid rgba(0, 0, 0, 0.12)',
          backgroundColor: 'primary.main',
          color: 'white'
        }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <BusinessOutlined sx={{ mr: 1 }} />
            Seleziona Tenant
          </Typography>
          <IconButton 
            onClick={onToggle}
            size="small"
            sx={{ 
              color: 'white',
              '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' }
            }}
            title="Nascondi barra laterale"
          >
            <ChevronLeft />
          </IconButton>
        </Box>

        {/* Contenuto principale */}
        <Box sx={{ p: 2, flexGrow: 1 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {loading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress size={40} />
            </Box>
          ) : (
            <>
              {/* Menu a tendina principale */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Tenant Attivo:
                </Typography>
                <FormControl fullWidth>
                  <Select
                    value={selectedTenant?.tenant_id || ''}
                    onChange={handleSelectChange}
                    displayEmpty
                    size="medium"
                    sx={{
                      '& .MuiSelect-select': {
                        py: 1.5,
                      },
                    }}
                  >
                    <MenuItem value="">
                      <em>Seleziona un tenant</em>
                    </MenuItem>
                    {availableTenants.map((tenant) => (
                      <MenuItem 
                        key={tenant.tenant_id} 
                        value={tenant.tenant_id}
                        sx={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                          py: 1.5,
                        }}
                      >
                        <Box display="flex" alignItems="center" gap={1} width="100%">
                          <Typography variant="body1" fontWeight="medium">
                            {tenant.nome}
                          </Typography>
                          {tenant.is_active && (
                            <Chip
                              label="Attivo"
                              size="small"
                              color="success"
                              variant="outlined"
                            />
                          )}
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          ID: {tenant.tenant_id.substring(0, 8)}...
                        </Typography>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {availableTenants.length === 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                    Nessun tenant disponibile
                  </Typography>
                )}
              </Box>

              <Divider sx={{ mb: 2 }} />

              {/* Info tenant selezionato */}
              {selectedTenant && (
                <Box
                  sx={{
                    p: 2,
                    backgroundColor: 'success.lighter',
                    borderRadius: 1,
                    border: 1,
                    borderColor: 'success.main',
                  }}
                >
                  <Typography variant="subtitle2" gutterBottom sx={{ color: 'success.dark', fontWeight: 600 }}>
                    🏢 Tenant Corrente:
                  </Typography>
                  <Typography variant="h6" color="success.dark" fontWeight="bold" sx={{ mb: 0.5 }}>
                    {selectedTenant.nome}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    ID completo: {selectedTenant.tenant_id}
                  </Typography>
                  {selectedTenant.is_active && (
                    <Box sx={{ mt: 1 }}>
                      <Chip
                        label="✓ Stato: Attivo"
                        size="small"
                        color="success"
                        variant="filled"
                      />
                    </Box>
                  )}
                </Box>
              )}

              {!selectedTenant && (
                <Box
                  sx={{
                    p: 2,
                    backgroundColor: 'warning.lighter',
                    borderRadius: 1,
                    border: 1,
                    borderColor: 'warning.main',
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="body2" color="warning.dark">
                    ⚠️ Nessun tenant selezionato
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Seleziona un tenant dal menu a tendina
                  </Typography>
                </Box>
              )}
            </>
          )}
        </Box>

        {/* Footer informativo */}
        <Box sx={{ 
          p: 2, 
          borderTop: '1px solid rgba(0, 0, 0, 0.12)',
          backgroundColor: 'grey.50'
        }}>
          <Typography variant="caption" color="text.secondary" textAlign="center" display="block">
            🔄 Sistema Multi-Tenant
          </Typography>
          <Typography variant="caption" color="text.secondary" textAlign="center" display="block" sx={{ mt: 0.5 }}>
            Totale tenant: {availableTenants.length}
          </Typography>
        </Box>
      </Box>
    </Drawer>
  );
};

export default TenantSelector;
