import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
  Alert,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControlLabel,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  AutoFixHigh as AutoFixHighIcon,
  Psychology as PsychologyIcon,
  ModelTraining as ModelTrainingIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Schedule as ScheduleIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';

interface FineTuningInfo {
  client: string;
  has_finetuned_model: boolean;
  active_model?: string;
  created_at?: string;
  base_model?: string;
  training_samples?: number;
  validation_samples?: number;
  model_size_mb?: number;
  previous_models_count?: number;
}

interface FineTuningStatus {
  client: string;
  model_info: FineTuningInfo;
  current_model?: {
    current_model: string;
    is_finetuned: boolean;
    finetuning_enabled: boolean;
  };
  pipeline_active: boolean;
  finetuning_available: boolean;
}

interface FineTuningPanelProps {
  clientName: string;
}

const FineTuningPanel: React.FC<FineTuningPanelProps> = ({ clientName }) => {
  const [status, setStatus] = useState<FineTuningStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Dialog states
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [minConfidence, setMinConfidence] = useState(0.7);
  const [forceRetrain, setForceRetrain] = useState(false);

  useEffect(() => {
    loadFineTuningStatus();
  }, [clientName]);

  const loadFineTuningStatus = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.get(`/api/finetuning/${clientName}/status`);
      
      if (response.success) {
        setStatus(response);
      } else {
        setError(response.error || 'Errore nel caricamento stato fine-tuning');
      }
    } catch (err: any) {
      console.error('Errore caricamento fine-tuning status:', err);
      setError(err.message || 'Errore di rete');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateFineTuning = async () => {
    setCreating(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await apiService.post(`/api/finetuning/${clientName}/create`, {
        min_confidence: minConfidence,
        force_retrain: forceRetrain
      });
      
      if (response.success) {
        setSuccess(`✅ Fine-tuning completato! Modello: ${response.model_name}`);
        setCreateDialogOpen(false);
        await loadFineTuningStatus(); // Ricarica lo stato
      } else {
        setError(response.error || 'Errore durante fine-tuning');
      }
    } catch (err: any) {
      console.error('Errore fine-tuning:', err);
      setError(err.message || 'Errore di rete');
    } finally {
      setCreating(false);
    }
  };

  const handleSwitchModel = async (modelType: 'finetuned' | 'base') => {
    setSwitching(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await apiService.post(`/api/finetuning/${clientName}/switch`, {
        model_type: modelType
      });
      
      if (response.success) {
        setSuccess(`✅ Switch a ${modelType === 'finetuned' ? 'modello fine-tuned' : 'modello base'} completato`);
        await loadFineTuningStatus(); // Ricarica lo stato
      } else {
        setError(response.error || 'Errore durante switch modello');
      }
    } catch (err: any) {
      console.error('Errore switch modello:', err);
      setError(err.message || 'Errore di rete');
    } finally {
      setSwitching(false);
    }
  };

  const formatFileSize = (sizeInMB: number): string => {
    if (sizeInMB >= 1024) {
      return `${(sizeInMB / 1024).toFixed(1)} GB`;
    }
    return `${sizeInMB.toFixed(0)} MB`;
  };

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleString('it-IT');
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2}>
            <PsychologyIcon color="primary" />
            <Typography variant="h6">Fine-Tuning Status</Typography>
          </Box>
          <LinearProgress sx={{ mt: 2 }} />
        </CardContent>
      </Card>
    );
  }

  if (!status?.finetuning_available) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" gap={2} mb={2}>
            <PsychologyIcon color="disabled" />
            <Typography variant="h6">Fine-Tuning</Typography>
          </Box>
          <Alert severity="warning">
            Fine-tuning non disponibile su questo sistema
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const hasFineTunedModel = status?.model_info?.has_finetuned_model || false;
  const isUsingFineTuned = status?.current_model?.is_finetuned || false;
  const currentModel = status?.current_model?.current_model || 'N/A';

  return (
    <>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="between" mb={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <PsychologyIcon color="primary" />
              <Typography variant="h6">Fine-Tuning AI</Typography>
              <Chip 
                size="small"
                label={isUsingFineTuned ? "FINE-TUNED" : "BASE MODEL"}
                color={isUsingFineTuned ? "success" : "default"}
                icon={isUsingFineTuned ? <CheckCircleIcon /> : <ModelTrainingIcon />}
              />
            </Box>
            <Tooltip title="Ricarica stato">
              <IconButton onClick={loadFineTuningStatus} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Alerts */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}
          
          {success && (
            <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
              {success}
            </Alert>
          )}

          {/* Stato corrente */}
          <Box mb={3}>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              Modello Attuale
            </Typography>
            <Box display="flex" alignItems="center" gap={2} mb={1}>
              <Typography variant="body1" fontFamily="monospace">
                {currentModel}
              </Typography>
              {isUsingFineTuned && (
                <Chip size="small" label="Personalizzato" color="success" />
              )}
            </Box>
            <Typography variant="body2" color="textSecondary">
              Pipeline: {status?.pipeline_active ? 'Attiva' : 'Inattiva'}
            </Typography>
          </Box>

          {/* Informazioni modello fine-tuned */}
          {hasFineTunedModel && (
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle2">
                  📊 Dettagli Modello Fine-Tuned
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexWrap="wrap" gap={2}>
                  <Box flex="1" minWidth="200px" textAlign="center">
                    <TrendingUpIcon color="action" />
                    <Typography variant="h6">
                      {status.model_info.training_samples || 0}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Esempi Training
                    </Typography>
                  </Box>
                  <Box flex="1" minWidth="200px" textAlign="center">
                    <SpeedIcon color="action" />
                    <Typography variant="h6">
                      {status.model_info.validation_samples || 0}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Esempi Validazione
                    </Typography>
                  </Box>
                  <Box flex="1" minWidth="200px" textAlign="center">
                    <StorageIcon color="action" />
                    <Typography variant="h6">
                      {status.model_info.model_size_mb ? formatFileSize(status.model_info.model_size_mb) : 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Dimensione
                    </Typography>
                  </Box>
                  <Box flex="1" minWidth="200px" textAlign="center">
                    <ScheduleIcon color="action" />
                    <Typography variant="h6">
                      {status.model_info.previous_models_count || 0}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Versioni Precedenti
                    </Typography>
                  </Box>
                </Box>
                
                {status.model_info.created_at && (
                  <Box mt={2}>
                    <Typography variant="body2" color="textSecondary">
                      Creato: {formatDate(status.model_info.created_at)}
                    </Typography>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
          )}

          <Divider sx={{ my: 2 }} />

          {/* Azioni */}
          <Box display="flex" gap={2} flexWrap="wrap">
            {/* Crea/Rigenera modello fine-tuned */}
            <Button
              variant={hasFineTunedModel ? "outlined" : "contained"}
              color="primary"
              startIcon={<AutoFixHighIcon />}
              onClick={() => setCreateDialogOpen(true)}
              disabled={creating || switching}
            >
              {hasFineTunedModel ? 'Rigenera Fine-Tuning' : 'Crea Fine-Tuning'}
            </Button>

            {/* Switch modello */}
            {hasFineTunedModel && (
              <>
                <Button
                  variant="outlined"
                  color={isUsingFineTuned ? "primary" : "success"}
                  startIcon={<ModelTrainingIcon />}
                  onClick={() => handleSwitchModel(isUsingFineTuned ? 'base' : 'finetuned')}
                  disabled={creating || switching}
                >
                  {isUsingFineTuned ? 'Usa Modello Base' : 'Usa Fine-Tuned'}
                </Button>
              </>
            )}
          </Box>

          {(creating || switching) && (
            <Box mt={2}>
              <LinearProgress />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                {creating ? 'Creazione modello in corso...' : 'Switch modello in corso...'}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Dialog per creazione fine-tuning */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={2}>
            <AutoFixHighIcon color="primary" />
            {hasFineTunedModel ? 'Rigenera Fine-Tuning' : 'Crea Modello Fine-Tuned'}
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" paragraph>
            Il fine-tuning creerà un modello Mistral personalizzato utilizzando le decisioni di revisione umana 
            validate per questo cliente.
          </Typography>

          <TextField
            label="Confidence Minima"
            type="number"
            fullWidth
            margin="normal"
            value={minConfidence}
            onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
            inputProps={{ min: 0.0, max: 1.0, step: 0.1 }}
            helperText="Confidence minima per includere esempi nel training (0.0-1.0)"
          />

          {hasFineTunedModel && (
            <FormControlLabel
              control={
                <Switch
                  checked={forceRetrain}
                  onChange={(e) => setForceRetrain(e.target.checked)}
                />
              }
              label="Forza re-training (sostituisce modello esistente)"
            />
          )}

          {hasFineTunedModel && !forceRetrain && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              Esiste già un modello fine-tuned. Abilita "Forza re-training" per ricrearlo.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>
            Annulla
          </Button>
          <Button
            onClick={handleCreateFineTuning}
            variant="contained"
            disabled={creating || (hasFineTunedModel && !forceRetrain)}
            startIcon={creating ? undefined : <AutoFixHighIcon />}
          >
            {creating ? 'Creazione...' : (hasFineTunedModel ? 'Rigenera' : 'Crea')}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default FineTuningPanel;
