import React, { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress
} from '@mui/material';
import {
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import { PromptInfo } from '../types/Tenant';

interface PromptSetupGuardProps {
  children: React.ReactNode;
}

const PromptSetupGuard: React.FC<PromptSetupGuardProps> = ({ children }) => {
  const { selectedTenant, promptStatus, refreshPromptStatus } = useTenant();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [customization, setCustomization] = useState('');

  // Se non c'è tenant selezionato, mostra loader
  if (!selectedTenant) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  // Se non abbiamo caricato lo status, mostra loader
  if (!promptStatus) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Verifica configurazione prompt...</Typography>
      </Box>
    );
  }

  // Se il tenant può operare, mostra il contenuto normale
  if (promptStatus.canOperate) {
    return <>{children}</>;
  }

  // Altrimenti mostra la schermata di configurazione obbligatoria
  const handleCreatePrompts = async () => {
    if (!selectedTenant) return;

    setCreating(true);
    try {
      await apiService.createPromptFromTemplate(
        selectedTenant.tenant_id,
        selectedTenant.nome,
        {
          customize_prompts: true,
          system_customization: customization || `per ${selectedTenant.nome}`
        }
      );

      // Refresh status
      await refreshPromptStatus();
      setCreateDialogOpen(false);
      setCustomization('');
    } catch (error) {
      console.error('Error creating prompts:', error);
    } finally {
      setCreating(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: '800px', mx: 'auto' }}>
      <Alert severity="warning" sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          ⚠️ Configurazione Prompt Obbligatoria
        </Typography>
        <Typography>
          Il tenant <strong>{selectedTenant.nome}</strong> non ha i prompt LLM obbligatori configurati.
          È necessario configurarli prima di utilizzare le funzionalità di classificazione.
        </Typography>
      </Alert>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📋 Stato Prompt Richiesti
          </Typography>
          <List>
            {promptStatus.requiredPrompts.map((prompt: PromptInfo, index: number) => (
              <ListItem key={index}>
                <ListItemIcon>
                  {prompt.exists ? (
                    <CheckCircleIcon color="success" />
                  ) : (
                    <CancelIcon color="error" />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body1">
                        {prompt.name}
                      </Typography>
                      <Chip
                        label={prompt.name}
                        size="small"
                        color={prompt.exists ? "success" : "error"}
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={prompt.description}
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🚀 Configurazione Automatica
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Possiamo creare automaticamente i prompt necessari utilizzando il template di Humanitas
            come base, personalizzandolo per {selectedTenant.nome}.
          </Typography>

          <Button
            variant="contained"
            size="large"
            startIcon={<SettingsIcon />}
            onClick={() => setCreateDialogOpen(true)}
            sx={{ mr: 2 }}
          >
            Configura Prompt Automaticamente
          </Button>

          <Typography variant="caption" display="block" sx={{ mt: 1, color: 'text.secondary' }}>
            Questa operazione creerà i prompt di sistema e template utente basati su Humanitas
          </Typography>
        </CardContent>
      </Card>

      {/* Dialog per personalizzazione */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Configura Prompt per {selectedTenant.nome}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            I prompt verranno creati utilizzando il template di Humanitas come base.
            Puoi personalizzare la descrizione del dominio:
          </Typography>

          <TextField
            fullWidth
            label="Personalizzazione Sistema"
            placeholder={`per ${selectedTenant.nome}`}
            value={customization}
            onChange={(e) => setCustomization(e.target.value)}
            helperText="Es: 'per la compagnia assicurativa Alleanza', 'per l'ospedale San Raffaele', ecc."
            sx={{ mb: 2 }}
          />

          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Template Humanitas:</strong><br />
              • System Prompt: Classificatore esperto sanitario<br />
              • User Template: Esempi dinamici di classificazione<br />
              • Personalizzazione: Adattamento al dominio specifico
            </Typography>
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Annulla
          </Button>
          <Button
            onClick={handleCreatePrompts}
            variant="contained"
            disabled={creating}
            startIcon={creating ? <CircularProgress size={20} /> : <SettingsIcon />}
          >
            {creating ? 'Creazione...' : 'Crea Prompt'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PromptSetupGuard;
