/**
 * File: ClusteringParametersManager.tsx
 * Autore: GitHub Copilot
 * Data: 24/08/2025
 * Descrizione: Componente React per la gestione dei parametri di clustering HDBSCAN
 * 
 * Storia aggiornamenti:
 * - 24/08/2025: Creazione componente iniziale con interfaccia user-friendly
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Alert,
  Chip,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  CardActions,
  CardHeader,
  CircularProgress,
  Snackbar,
  Tooltip,
  IconButton,
  Grid,
  Paper,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  RestartAlt as ResetIcon,
  Save as SaveIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';

interface ClusteringParameter {
  value: number | string;
  default: number | string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description: string;
  explanation: string;
  impact: {
    low?: string;
    medium?: string;
    high?: string;
    [key: string]: string | undefined;
  };
  recommendation?: string;
}

interface ClusteringParameters {
  min_cluster_size: ClusteringParameter;
  min_samples: ClusteringParameter;
  cluster_selection_epsilon: ClusteringParameter;
  metric: ClusteringParameter;
}

interface ParametersResponse {
  success: boolean;
  parameters: ClusteringParameters;
  tenant_id: string;
  tenant_info: any;
  last_updated: string;
  config_source: 'default' | 'custom' | 'error';
  config_details: {
    status: 'default' | 'custom' | 'error';
    is_using_default: boolean;
    custom_config_info: any;
    base_config_file: string;
  };
}

/**
 * Componente per la gestione dei parametri di clustering HDBSCAN
 * 
 * Funzionalità:
 * - Visualizzazione parametri attuali con spiegazioni
 * - Modifica parametri con validazione in tempo reale
 * - Anteprima impatto delle modifiche
 * - Salvataggio e reset parametri
 * 
 * Props: Nessuna
 * 
 * Ultima modifica: 24/08/2025
 */
const ClusteringParametersManager: React.FC = () => {
  const { selectedTenant } = useTenant();
  const [parameters, setParameters] = useState<ClusteringParameters | null>(null);
  const [originalParameters, setOriginalParameters] = useState<ClusteringParameters | null>(null);
  const [configStatus, setConfigStatus] = useState<'default' | 'custom' | 'error'>('default');
  const [configDetails, setConfigDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});

  /**
   * Carica i parametri di clustering attuali
   * 
   * Input: selectedTenant dal contesto
   * Output: Aggiorna state parameters
   */
  const loadParameters = async () => {
    if (!selectedTenant?.tenant_id) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.get<ParametersResponse>(
        `/api/clustering/${selectedTenant.tenant_id}/parameters`
      );

      if (response.success && response.parameters) {
        setParameters(response.parameters);
        setOriginalParameters(JSON.parse(JSON.stringify(response.parameters)));
        setConfigStatus(response.config_source || 'default');
        setConfigDetails(response.config_details || null);
        setHasChanges(false);
        
        // Log informazioni di configurazione per debug
        console.log('📊 [CLUSTERING CONFIG] Status:', response.config_source);
        console.log('📊 [CLUSTERING CONFIG] Details:', response.config_details);
      } else {
        setError('Errore caricamento parametri clustering');
      }
    } catch (err: any) {
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Salva i parametri modificati
   * 
   * Input: parameters modificati
   * Output: Aggiorna configurazione backend
   */
  const saveParameters = async () => {
    if (!selectedTenant?.tenant_id || !parameters || !hasChanges) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      // Prepara payload con solo i valori
      const payload = {
        parameters: Object.entries(parameters).reduce((acc, [key, param]) => {
          acc[key] = param.value;
          return acc;
        }, {} as Record<string, any>)
      };

      const response = await apiService.post(
        `/api/clustering/${selectedTenant.tenant_id}/parameters`,
        payload
      );

      if (response.success) {
        setSuccess('Parametri aggiornati con successo! Le modifiche saranno applicate al prossimo training.');
        setOriginalParameters(JSON.parse(JSON.stringify(parameters)));
        setHasChanges(false);
      } else {
        setError(response.error || 'Errore salvataggio parametri');
      }
    } catch (err: any) {
      setError(`Errore salvataggio: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Reset parametri ai valori di default
   * 
   * Input: Nessuno
   * Output: Ripristina parametri originali
   */
  const resetParameters = async () => {
    if (!selectedTenant?.tenant_id) return;

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await apiService.post(
        `/api/clustering/${selectedTenant.tenant_id}/parameters/reset`,
        {}
      );

      if (response.success) {
        setSuccess('Parametri ripristinati ai valori di default!');
        await loadParameters(); // Ricarica parametri
      } else {
        setError(response.error || 'Errore reset parametri');
      }
    } catch (err: any) {
      setError(`Errore reset: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  /**
   * Aggiorna un parametro specifico
   * 
   * Input: Nome parametro e nuovo valore
   * Output: Aggiorna state e valida
   */
  const updateParameter = (paramName: string, newValue: number | string) => {
    if (!parameters) return;

    const updatedParameters = {
      ...parameters,
      [paramName]: {
        ...parameters[paramName as keyof ClusteringParameters],
        value: newValue
      }
    };

    setParameters(updatedParameters);

    // Controlla se ci sono cambiamenti
    const hasChanged = originalParameters && Object.entries(updatedParameters).some(
      ([key, param]) => param.value !== originalParameters[key as keyof ClusteringParameters]?.value
    );
    setHasChanges(!!hasChanged);

    // Validazione
    validateParameter(paramName, newValue, parameters[paramName as keyof ClusteringParameters]);
  };

  /**
   * Valida un parametro singolo
   * 
   * Input: Nome parametro, valore, definizione parametro
   * Output: Aggiorna validationErrors
   */
  const validateParameter = (paramName: string, value: number | string, paramDef: ClusteringParameter) => {
    const errors = { ...validationErrors };
    delete errors[paramName];

    if (typeof value === 'number' && paramDef.min !== undefined && paramDef.max !== undefined) {
      if (value < paramDef.min || value > paramDef.max) {
        errors[paramName] = `Valore deve essere tra ${paramDef.min} e ${paramDef.max}`;
      }
    }

    if (typeof value === 'string' && paramDef.options && !paramDef.options.includes(value)) {
      errors[paramName] = `Valore non valido. Opzioni: ${paramDef.options.join(', ')}`;
    }

    setValidationErrors(errors);
  };

  /**
   * Determina l'icona di stato per un parametro
   * 
   * Input: Nome parametro, valore attuale
   * Output: Elemento JSX con icona appropriata
   */
  const getParameterStatusIcon = (paramName: string, currentValue: number | string) => {
    if (!originalParameters) return null;

    const originalValue = originalParameters[paramName as keyof ClusteringParameters]?.value;
    const isChanged = currentValue !== originalValue;
    const hasError = validationErrors[paramName];

    if (hasError) {
      return <ErrorIcon color="error" fontSize="small" />;
    } else if (isChanged) {
      return <WarningIcon color="warning" fontSize="small" />;
    } else {
      return <CheckCircleIcon color="success" fontSize="small" />;
    }
  };

  /**
   * Ottiene le informazioni di stato della configurazione
   * 
   * Output: Oggetto con colore, icona e messaggio
   */
  const getConfigStatusInfo = () => {
    switch (configStatus) {
      case 'custom':
        return {
          color: 'success' as const,
          icon: <CheckCircleIcon />,
          label: 'Personalizzata',
          message: 'Usando configurazione personalizzata per questo tenant'
        };
      case 'error':
        return {
          color: 'error' as const,
          icon: <ErrorIcon />,
          label: 'Errore',
          message: 'Errore nel caricamento configurazione personalizzata, usando default'
        };
      default:
        return {
          color: 'info' as const,
          icon: <InfoIcon />,
          label: 'Default',
          message: 'Usando configurazione default da config.yaml'
        };
    }
  };

  /**
   * Genera il componente di controllo per un parametro
   * 
   * Input: Nome parametro e definizione
   * Output: Elemento JSX per il controllo
   */
  const renderParameterControl = (paramName: string, param: ClusteringParameter) => {
    const isNumeric = typeof param.value === 'number';
    const isSelect = param.options && param.options.length > 0;
    const hasError = validationErrors[paramName];

    if (isSelect) {
      return (
        <FormControl fullWidth error={!!hasError}>
          <InputLabel>{param.description}</InputLabel>
          <Select
            value={param.value}
            label={param.description}
            onChange={(e: any) => updateParameter(paramName, e.target.value)}
          >
            {param.options!.map((option) => (
              <MenuItem key={option} value={option}>
                {option}
                {option === param.default && <Chip label="Default" size="small" sx={{ ml: 1 }} />}
              </MenuItem>
            ))}
          </Select>
          {hasError && (
            <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
              {validationErrors[paramName]}
            </Typography>
          )}
        </FormControl>
      );
    }

    if (isNumeric && param.min !== undefined && param.max !== undefined) {
      return (
        <Box>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
            <Typography variant="body2" fontWeight="medium">
              {param.description}
            </Typography>
            <TextField
              size="small"
              type="number"
              value={param.value}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateParameter(paramName, parseFloat(e.target.value))}
              inputProps={{
                min: param.min,
                max: param.max,
                step: param.step || (param.max < 1 ? 0.01 : 1)
              }}
              sx={{ width: 100 }}
              error={!!hasError}
            />
          </Box>
          <Slider
            value={param.value as number}
            min={param.min}
            max={param.max}
            step={param.step || (param.max < 1 ? 0.01 : 1)}
            onChange={(_, value) => updateParameter(paramName, value as number)}
            valueLabelDisplay="auto"
            color={hasError ? "error" : "primary"}
          />
          {hasError && (
            <Typography variant="caption" color="error">
              {validationErrors[paramName]}
            </Typography>
          )}
        </Box>
      );
    }

    return null;
  };

  // Carica parametri al mount e quando cambia tenant
  useEffect(() => {
    loadParameters();
  }, [selectedTenant]);

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
        Seleziona un tenant per gestire i parametri di clustering
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center">
              <SettingsIcon sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" component="h1">
                Parametri Clustering HDBSCAN
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip 
                label={selectedTenant.nome} 
                color="primary" 
                variant="outlined" 
              />
              <Chip
                icon={getConfigStatusInfo().icon}
                label={`Config: ${getConfigStatusInfo().label}`}
                color={getConfigStatusInfo().color}
                size="small"
              />
            </Box>
          </Box>

          {/* Alert informativi con stato configurazione */}
          <Alert 
            severity={getConfigStatusInfo().color} 
            sx={{ mb: 2 }}
            icon={getConfigStatusInfo().icon}
          >
            <Typography variant="body2">
              <strong>Stato Configurazione:</strong> {getConfigStatusInfo().message}
              {configDetails && configStatus === 'custom' && configDetails.custom_config_info && (
                <><br/>
                <strong>Ultimo aggiornamento:</strong> {new Date(configDetails.custom_config_info.last_updated || '').toLocaleString('it-IT')}
                </>
              )}
              {configDetails && configStatus === 'error' && configDetails.custom_config_info && (
                <><br/>
                <strong>Errore:</strong> {configDetails.custom_config_info.error}
                </>
              )}
            </Typography>
          </Alert>

          {/* Alert informativi */}
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Importante:</strong> Se hai troppe etichette simili dopo il training, 
              il problema è spesso nei parametri di clustering. Aumenta <strong>cluster_selection_epsilon</strong> 
              per raggruppare meglio le conversazioni semanticamente simili.
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

          {/* Parametri */}
          {parameters && (
            <>
              <Box display="flex" flexWrap="wrap" gap={2}>
                {Object.entries(parameters).map(([paramName, param]) => (
                  <Box key={paramName} flex="1 1 400px" minWidth="300px">
                    <Paper sx={{ p: 2, height: '100%' }}>
                      <Box display="flex" alignItems="flex-start" mb={2}>
                        <Box flexGrow={1}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="h6" component="h3">
                              {param.description}
                            </Typography>
                            <Box ml={1}>
                              {getParameterStatusIcon(paramName, param.value)}
                            </Box>
                            <Tooltip title={param.explanation}>
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>

                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {param.explanation}
                          </Typography>

                          {/* Controllo parametro */}
                          {renderParameterControl(paramName, param)}

                          {/* Impatto */}
                          <Accordion sx={{ mt: 2 }}>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="body2">
                                Impatto delle modifiche
                              </Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              {Object.entries(param.impact).map(([level, description]) => (
                                <Box key={level} sx={{ mb: 1 }}>
                                  <Chip 
                                    label={level.toUpperCase()} 
                                    size="small" 
                                    color={level === 'high' ? 'error' : level === 'medium' ? 'warning' : 'success'}
                                    sx={{ mr: 1 }}
                                  />
                                  <Typography variant="body2" component="span">
                                    {description as string}
                                  </Typography>
                                </Box>
                              ))}
                              {param.recommendation && (
                                <Alert severity="warning" sx={{ mt: 1 }}>
                                  <Typography variant="body2">
                                    <strong>Raccomandazione:</strong> {param.recommendation}
                                  </Typography>
                                </Alert>
                              )}
                            </AccordionDetails>
                          </Accordion>
                        </Box>
                      </Box>
                    </Paper>
                  </Box>
                ))}
              </Box>

              {/* Azioni */}
              <Box display="flex" justifyContent="space-between" alignItems="center" mt={3}>
                <Button
                  variant="outlined"
                  startIcon={<ResetIcon />}
                  onClick={resetParameters}
                  disabled={saving || loading}
                >
                  Reset Default
                </Button>

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
                    onClick={saveParameters}
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

export default ClusteringParametersManager;
