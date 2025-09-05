/**
 * File: ReviewQueueThresholdManager.tsx
 * Autore: Valerio Bignardi
 * Data: 03/09/2025
 * Descrizione: Componente React per la gestione delle soglie Review Queue
 * 
 * Storia aggiornamenti:
 * - 03/09/2025: Creazione componente per soglie Review Queue dinamiche
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Alert,
  Chip,
  Button,
  Slider,
  Card,
  CardContent,
  CircularProgress,
  Tooltip,
  IconButton,
  Paper,
  LinearProgress,
  TextField,
  Switch,
  FormControlLabel,
  FormGroup
} from '@mui/material';
import {
  Settings as SettingsIcon,
  RestartAlt as ResetIcon,
  Save as SaveIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Reviews as ReviewsIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';

interface ReviewQueueThreshold {
  value: number | boolean;
  default: number | boolean;
  min?: number;
  max?: number;
  step?: number;
  description: string;
  explanation: string;
  impact: string;
  recommendation?: string;
}

interface ReviewQueueThresholds {
  outlier_confidence_threshold: ReviewQueueThreshold;
  propagated_confidence_threshold: ReviewQueueThreshold;
  representative_confidence_threshold: ReviewQueueThreshold;
  minimum_consensus_threshold: ReviewQueueThreshold;
  enable_smart_review: ReviewQueueThreshold;
  max_pending_per_batch: ReviewQueueThreshold;
}

interface ThresholdsResponse {
  success: boolean;
  thresholds: Record<string, any>;
  tenant_id: string;
  config_source: 'default' | 'custom';
  last_updated: string;
}

/**
 * Componente per la gestione delle soglie Review Queue
 * 
 * Funzionalità:
 * - Visualizzazione soglie attuali con spiegazioni
 * - Modifica soglie con validazione in tempo reale
 * - Anteprima impatto delle modifiche
 * - Salvataggio e reset soglie
 * 
 * Props: Nessuna
 * 
 * Ultima modifica: 03/09/2025
 */
const ReviewQueueThresholdManager: React.FC = () => {
  const { selectedTenant } = useTenant();
  const [thresholds, setThresholds] = useState<ReviewQueueThresholds | null>(null);
  const [originalThresholds, setOriginalThresholds] = useState<ReviewQueueThresholds | null>(null);
  const [configSource, setConfigSource] = useState<'default' | 'custom'>('default');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  /**
   * Definizione parametri con metadati per l'interfaccia
   */
  const getThresholdDefinitions = useCallback((currentValues: Record<string, any>): ReviewQueueThresholds => {
    return {
      outlier_confidence_threshold: {
        value: currentValues.outlier_confidence_threshold ?? 0.7,
        default: 0.7,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        description: "Soglia Confidenza Outlier",
        explanation: "Outliers con confidenza sotto questa soglia vanno in Review Queue per supervisione umana",
        impact: "Valori più alti = più outliers in review, maggiore controllo qualità",
        recommendation: "0.7 è ottimale per bilanciare automazione e controllo qualità"
      },
      propagated_confidence_threshold: {
        value: currentValues.propagated_confidence_threshold ?? 0.8,
        default: 0.8,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        description: "Soglia Qualità Propagazione",
        explanation: "Soglia minima di confidenza per validare propagazioni durante training supervisionato. I propagati vengono sempre auto-classificati e NON vanno MAI automaticamente in review.",
        impact: "Valori più alti = propagazioni più rigorose durante il training, maggiore qualità",
        recommendation: "0.7-0.8 è ottimale per garantire buona qualità delle propagazioni"
      },
      representative_confidence_threshold: {
        value: currentValues.representative_confidence_threshold ?? 0.9,
        default: 0.9,
        min: 0.0,
        max: 1.0,
        step: 0.05,
        description: "Soglia Confidenza Rappresentanti",
        explanation: "Rappresentanti con confidenza sotto questa soglia vanno in review",
        impact: "Valori più alti = più rappresentanti in review, massima accuratezza",
        recommendation: "0.9 garantisce alta qualità per i rappresentanti cluster"
      },
      minimum_consensus_threshold: {
        value: currentValues.minimum_consensus_threshold ?? 3,
        default: 3,
        min: 1,
        max: 10,
        step: 1,
        description: "Consenso Minimo",
        explanation: "Numero minimo di validazioni concordi per auto-classificazione",
        impact: "Valori più alti = maggiore consenso richiesto, più conservativo",
        recommendation: "3 è bilanciato per la maggior parte dei casi"
      },
      enable_smart_review: {
        value: currentValues.enable_smart_review ?? true,
        default: true,
        description: "Abilita Review Intelligente",
        explanation: "Attiva la valutazione automatica intelligente per Review Queue",
        impact: "Se disabilitato, tutti i casi vengono auto-classificati senza review",
        recommendation: "Mantieni abilitato per controllo qualità ottimale"
      },
      max_pending_per_batch: {
        value: currentValues.max_pending_per_batch ?? 100,
        default: 100,
        min: 10,
        max: 1000,
        step: 10,
        description: "Max Casi Pending per Batch",
        explanation: "Numero massimo di casi inviati in review per singolo training",
        impact: "Limita il carico di lavoro umano per batch di training",
        recommendation: "100 è gestibile per review giornaliera"
      }
    };
  }, []);

  /**
   * Carica le soglie attuali per il tenant
   */
  const loadThresholds = useCallback(async () => {
    if (!selectedTenant?.tenant_id) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getReviewQueueThresholds(selectedTenant.tenant_id);

      if (response.success && response.thresholds) {
        const thresholdDefinitions = getThresholdDefinitions(response.thresholds);
        setThresholds(thresholdDefinitions);
        setOriginalThresholds(JSON.parse(JSON.stringify(thresholdDefinitions)));
        setConfigSource(response.config_source || 'default');
        setHasChanges(false);
        
        console.log('📊 [REVIEW-QUEUE] Soglie caricate:', response.thresholds);
      } else {
        setError('Errore caricamento soglie Review Queue');
      }
    } catch (err: any) {
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant?.tenant_id, getThresholdDefinitions]);

  /**
   * Salva le soglie modificate
   */
  const saveThresholds = async () => {
    if (!selectedTenant?.tenant_id || !thresholds || !hasChanges) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      // Prepara payload con solo i valori
      const payload = {
        thresholds: Object.entries(thresholds).reduce((acc, [key, threshold]) => {
          acc[key] = threshold.value;
          return acc;
        }, {} as Record<string, any>)
      };

      const response = await apiService.updateReviewQueueThresholds(
        selectedTenant.tenant_id,
        payload.thresholds
      );

      if (response.success) {
        setSuccess('Soglie Review Queue aggiornate con successo! Le modifiche saranno applicate al prossimo training.');
        setOriginalThresholds(JSON.parse(JSON.stringify(thresholds)));
        setHasChanges(false);
        setConfigSource('custom');
      } else {
        setError('Errore salvataggio soglie');
      }
    } catch (err: any) {
      setError(`Errore salvataggio: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Reset soglie ai valori default
   */
  const resetThresholds = async () => {
    if (!selectedTenant?.tenant_id) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await apiService.resetReviewQueueThresholds(selectedTenant.tenant_id);

      if (response.success) {
        setSuccess('Soglie ripristinate ai valori default!');
        await loadThresholds(); // Ricarica soglie
      } else {
        setError('Errore reset soglie');
      }
    } catch (err: any) {
      setError(`Errore reset: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Aggiorna una soglia specifica
   */
  const updateThreshold = (thresholdName: string, newValue: number | boolean) => {
    if (!thresholds) return;

    const updatedThresholds = {
      ...thresholds,
      [thresholdName]: {
        ...thresholds[thresholdName as keyof ReviewQueueThresholds],
        value: newValue
      }
    };

    setThresholds(updatedThresholds);

    // Controlla se ci sono cambiamenti
    const hasChanged = originalThresholds && Object.entries(updatedThresholds).some(
      ([key, threshold]) => threshold.value !== originalThresholds[key as keyof ReviewQueueThresholds]?.value
    );
    setHasChanges(!!hasChanged);

    // Validazione
    validateThreshold(thresholdName, newValue, thresholds[thresholdName as keyof ReviewQueueThresholds]);
  };

  /**
   * Valida una soglia singola
   */
  const validateThreshold = (thresholdName: string, value: number | boolean, thresholdDef: ReviewQueueThreshold) => {
    const errors = { ...validationErrors };
    delete errors[thresholdName];

    if (typeof value === 'number' && thresholdDef.min !== undefined && thresholdDef.max !== undefined) {
      if (value < thresholdDef.min || value > thresholdDef.max) {
        errors[thresholdName] = `Valore deve essere tra ${thresholdDef.min} e ${thresholdDef.max}`;
      }
    }

    setValidationErrors(errors);
  };

  /**
   * Genera il componente di controllo per una soglia
   */
  const renderThresholdControl = (thresholdName: string, threshold: ReviewQueueThreshold) => {
    const isNumeric = typeof threshold.value === 'number';
    const isBoolean = typeof threshold.value === 'boolean';
    const hasError = validationErrors[thresholdName];

    if (isBoolean) {
      return (
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                checked={threshold.value as boolean}
                onChange={(e) => updateThreshold(thresholdName, e.target.checked)}
                color={hasError ? "error" : "primary"}
              />
            }
            label={
              <Box>
                <Typography variant="body2" fontWeight="medium">
                  {threshold.description}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {threshold.value ? 'Attivato' : 'Disattivato'}
                  {threshold.value === threshold.default && ' (Default)'}
                </Typography>
              </Box>
            }
          />
        </FormGroup>
      );
    }

    if (isNumeric && threshold.min !== undefined && threshold.max !== undefined) {
      return (
        <Box>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Typography variant="body2" fontWeight="medium">
              {threshold.description}
            </Typography>
            <TextField
              size="small"
              type="number"
              value={threshold.value}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateThreshold(thresholdName, parseFloat(e.target.value))}
              inputProps={{
                min: threshold.min,
                max: threshold.max,
                step: threshold.step || 0.05
              }}
              sx={{ width: 100 }}
              error={!!hasError}
            />
          </Box>
          <Slider
            value={threshold.value as number}
            min={threshold.min}
            max={threshold.max}
            step={threshold.step || 0.05}
            onChange={(_, value) => updateThreshold(thresholdName, value as number)}
            valueLabelDisplay="auto"
            color={hasError ? "error" : "primary"}
            marks={[
              { value: threshold.min, label: threshold.min.toString() },
              { value: threshold.default as number, label: `${threshold.default} (Default)` },
              { value: threshold.max, label: threshold.max.toString() }
            ]}
          />
          {hasError && (
            <Typography variant="caption" color="error">
              {validationErrors[thresholdName]}
            </Typography>
          )}
        </Box>
      );
    }

    return null;
  };

  // Carica soglie al mount e quando cambia tenant
  useEffect(() => {
    loadThresholds();
  }, [selectedTenant, loadThresholds]);

  // Auto-clear success/error messages
  useEffect(() => {
    if (success || error) {
      const timer = setTimeout(() => {
        setSuccess(null);
        setError(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [success, error]);

  if (!selectedTenant) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Seleziona un tenant per gestire le soglie Review Queue
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center">
              <ReviewsIcon sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" component="h1">
                Soglie Review Queue
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip 
                label={selectedTenant.tenant_name} 
                color="primary" 
                variant="outlined" 
              />
              <Chip
                icon={configSource === 'custom' ? <CheckCircleIcon /> : <InfoIcon />}
                label={`Config: ${configSource === 'custom' ? 'Personalizzata' : 'Default'}`}
                color={configSource === 'custom' ? 'success' : 'info'}
                size="small"
              />
            </Box>
          </Box>

          {/* Alert informativi */}
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Controllo Qualità Intelligente:</strong> Configura quando i casi devono essere inviati 
              alla Review Queue per supervisione umana in base alla confidenza e al tipo di classificazione.
            </Typography>
          </Alert>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {success}
            </Alert>
          )}

          {/* Loading state */}
          {loading && <LinearProgress sx={{ mb: 2 }} />}

          {/* Soglie */}
          {thresholds && (
            <>
              <Box display="flex" flexWrap="wrap" gap={2}>
                {Object.entries(thresholds).map(([thresholdName, threshold]) => (
                  <Box key={thresholdName} flex="1 1 400px" minWidth="300px">
                    <Paper sx={{ p: 2, height: '100%' }}>
                      <Box display="flex" alignItems="flex-start" mb={2}>
                        <Box flexGrow={1}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="h6" component="h3">
                              {threshold.description}
                            </Typography>
                            <Tooltip title={threshold.explanation}>
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>

                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {threshold.explanation}
                          </Typography>

                          {/* Controllo soglia */}
                          {renderThresholdControl(thresholdName, threshold)}

                          {/* Impatto e raccomandazione */}
                          <Alert 
                            severity="info" 
                            sx={{ mt: 2, fontSize: '0.8rem' }}
                          >
                            <Typography variant="body2">
                              <strong>Impatto:</strong> {threshold.impact}
                              {threshold.recommendation && (
                                <>
                                  <br/>
                                  <strong>Raccomandazione:</strong> {threshold.recommendation}
                                </>
                              )}
                            </Typography>
                          </Alert>
                        </Box>
                      </Box>
                    </Paper>
                  </Box>
                ))}
              </Box>

              {/* Azioni */}
              <Box display="flex" justifyContent="space-between" alignItems="center" mt={3}>
                <Box display="flex" gap={2}>
                  <Button
                    variant="outlined"
                    startIcon={<ResetIcon />}
                    onClick={resetThresholds}
                    disabled={saving || loading}
                  >
                    Reset Default
                  </Button>
                </Box>

                <Box display="flex" gap={2}>
                  {hasChanges && (
                    <Chip 
                      label={`${Object.keys(validationErrors).length > 0 ? 'Errori presenti' : 'Modifiche non salvate'}`}
                      color={Object.keys(validationErrors).length > 0 ? 'error' : 'warning'}
                      size="small"
                    />
                  )}
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={saveThresholds}
                    disabled={!hasChanges || Object.keys(validationErrors).length > 0 || saving || loading}
                  >
                    {saving ? 'Salvataggio...' : 'Salva Modifiche'}
                  </Button>
                </Box>
              </Box>
            </>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default ReviewQueueThresholdManager;
