import React, { useState } from 'react';
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
  Divider,
  Button,
  Snackbar
} from '@mui/material';
import {
  BusinessOutlined,
  ChevronLeft,
  Sync as SyncIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';

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
  const { selectedTenant, availableTenants, setSelectedTenant, loading, error, refreshTenants } = useTenant();
  
  // Stati per sincronizzazione
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const handleSelectChange = (event: SelectChangeEvent<string>) => {
    const tenantId = event.target.value;
    const tenant = availableTenants.find(t => t.tenant_id === tenantId);
    if (tenant) {
      setSelectedTenant(tenant);
    }
  };

  /**
   * Gestisce la sincronizzazione dei tenant dal server remoto
   * 
   * Autore: Valerio Bignardi
   * Data: 2025-08-27
   * Descrizione: Sincronizza tenant remoti e aggiorna la lista locale
   */
  const handleSyncTenants = async () => {
    setSyncing(true);
    setSyncResult(null);
    
    try {
      console.log('🔄 [SYNC] Avvio sincronizzazione tenant...');
      
      const result = await apiService.syncTenants();
      
      if (result.success) {
        console.log('✅ [SYNC] Sincronizzazione completata:', result);
        
        setSyncResult({
          type: 'success',
          message: `✅ Sincronizzazione completata! ${result.imported_count} nuovi tenant importati.`
        });
        
        // Aggiorna la lista dei tenant nel context
        if (refreshTenants) {
          await refreshTenants();
        }
        
      } else {
        console.error('❌ [SYNC] Sincronizzazione fallita:', result.error);
        
        setSyncResult({
          type: 'error',
          message: `❌ Errore sincronizzazione: ${result.error}`
        });
      }
      
    } catch (error) {
      console.error('💥 [SYNC] Errore durante sincronizzazione:', error);
      
      setSyncResult({
        type: 'error',
        message: `💥 Errore durante sincronizzazione: ${error instanceof Error ? error.message : String(error)}`
      });
      
    } finally {
      setSyncing(false);
    }
  };

  const handleCloseSyncResult = () => {
    setSyncResult(null);
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
              {/* NUOVO: Pulsante sincronizzazione tenant */}
              <Box sx={{ mb: 3 }}>
                <Button
                  variant="outlined"
                  color="primary"
                  fullWidth
                  startIcon={syncing ? <CircularProgress size={16} /> : <SyncIcon />}
                  onClick={handleSyncTenants}
                  disabled={syncing}
                  sx={{
                    py: 1.5,
                    borderRadius: 2,
                    textTransform: 'none',
                    fontWeight: 600,
                    '&:hover': {
                      backgroundColor: 'primary.lighter',
                    },
                  }}
                >
                  {syncing ? 'Sincronizzazione...' : '🔄 Sincronizza Nuovi Tenant'}
                </Button>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block', textAlign: 'center' }}>
                  Importa tenant dal server remoto
                </Typography>
              </Box>

              <Divider sx={{ mb: 2 }} />

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
                            {tenant.tenant_name}
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
                    {selectedTenant.tenant_name}
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

      {/* Snackbar per feedback sincronizzazione */}
      <Snackbar
        open={!!syncResult}
        autoHideDuration={6000}
        onClose={handleCloseSyncResult}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert 
          onClose={handleCloseSyncResult} 
          severity={syncResult?.type} 
          sx={{ width: '100%' }}
        >
          {syncResult?.message}
        </Alert>
      </Snackbar>
    </Drawer>
  );
};

export default TenantSelector;
