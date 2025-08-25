import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Chip,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import {
  BusinessOutlined,
  CheckCircleOutlined,
  RadioButtonUncheckedOutlined
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { Tenant } from '../types/Tenant';

interface TenantSelectorProps {
  open: boolean;
  onClose: () => void;
  drawerWidth?: number;
}

const TenantSelector: React.FC<TenantSelectorProps> = ({
  open,
  onClose,
  drawerWidth = 280
}) => {
  const { selectedTenant, availableTenants, setSelectedTenant, loading, error } = useTenant();

  const handleTenantSelect = (tenant: Tenant) => {
    setSelectedTenant(tenant);
    onClose();
  };

  const handleSelectChange = (event: SelectChangeEvent<string>) => {
    const tenantId = event.target.value;
    const tenant = availableTenants.find(t => t.tenant_id === tenantId);
    if (tenant) {
      setSelectedTenant(tenant);
    }
  };

  return (
    <Drawer
      variant="temporary"
      open={open}
      onClose={onClose}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      <Box sx={{ overflow: 'auto', p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <BusinessOutlined sx={{ mr: 1 }} />
          Seleziona Tenant
        </Typography>

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
            {/* Selector compatto per mobile */}
            <Box sx={{ mb: 3, display: { xs: 'block', sm: 'none' } }}>
              <FormControl fullWidth>
                <Select
                  value={selectedTenant?.tenant_id || ''}
                  onChange={handleSelectChange}
                  displayEmpty
                  size="small"
                >
                  <MenuItem value="">
                    <em>Seleziona un tenant</em>
                  </MenuItem>
                  {availableTenants.map((tenant) => (
                    <MenuItem key={tenant.tenant_id} value={tenant.tenant_id}>
                      {tenant.nome} {tenant.is_active && '✓'}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* Lista dettagliata per desktop */}
            <List sx={{ display: { xs: 'none', sm: 'block' } }}>
              {availableTenants.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                  Nessun tenant disponibile
                </Typography>
              ) : (
                availableTenants.map((tenant) => (
                  <ListItem key={tenant.tenant_id} disablePadding>
                    <ListItemButton
                      onClick={() => handleTenantSelect(tenant)}
                      selected={selectedTenant?.tenant_id === tenant.tenant_id}
                      sx={{
                        borderRadius: 1,
                        mb: 0.5,
                        '&.Mui-selected': {
                          backgroundColor: 'primary.lighter',
                          '&:hover': {
                            backgroundColor: 'primary.light',
                          },
                        },
                      }}
                    >
                      <ListItemIcon>
                        {selectedTenant?.tenant_id === tenant.tenant_id ? (
                          <CheckCircleOutlined color="primary" />
                        ) : (
                          <RadioButtonUncheckedOutlined />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="subtitle2">
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
                        }
                        secondary={
                          <Typography variant="caption" color="text.secondary">
                            ID: {tenant.tenant_id.substring(0, 8)}...
                          </Typography>
                        }
                      />
                    </ListItemButton>
                  </ListItem>
                ))
              )}
            </List>

            {/* Info tenant selezionato */}
            {selectedTenant && (
              <Box
                sx={{
                  mt: 3,
                  p: 2,
                  backgroundColor: 'grey.50',
                  borderRadius: 1,
                  border: 1,
                  borderColor: 'divider',
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Tenant Corrente:
                </Typography>
                <Typography variant="body2" color="primary" fontWeight="medium">
                  {selectedTenant.nome}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {selectedTenant.tenant_id}
                </Typography>
              </Box>
            )}
          </>
        )}
      </Box>
    </Drawer>
  );
};

export default TenantSelector;
