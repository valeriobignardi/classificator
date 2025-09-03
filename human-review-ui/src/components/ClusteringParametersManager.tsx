/**
 * File: ClusteringParametersManager.tsx
 * Autore: GitHub Copilot
 * Data: 24/08/2025
 * Descrizione: Componente React per la gestione dei parametri di clustering HDBSCAN
 * 
 * Storia aggiornamenti:
 * - 24/08/2025: Creazione componente iniziale con interfaccia user-friendly
 * - 25/08/2025: Aggiunta funzionalità test clustering con preview risultati
 */

import React, { useState, useEffect, useCallback } from 'react';
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
  CircularProgress,
  Tooltip,
  IconButton,
  Paper,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Switch,
  FormControlLabel,
  FormGroup,
  Collapse
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  RestartAlt as ResetIcon,
  Save as SaveIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import ClusteringTestResults from './ClusteringTestResults';
import ClusteringVersionManager from './ClusteringVersionManager';

interface ClusteringParameter {
  value: number | string | boolean;
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: (string | boolean)[];
  description: string;
  explanation: string;
  impact: {
    low?: string;
    medium?: string;
    high?: string;
    [key: string]: string | undefined;
  };
  recommendation?: string;
  gpu_supported?: boolean;  // 🆕 SUPPORTO GPU
  gpu_warning?: string;     // 🆕 AVVISO GPU
}

interface ClusteringParameters {
  // 📊 PARAMETRI HDBSCAN BASE
  min_cluster_size: ClusteringParameter;
  min_samples: ClusteringParameter;
  cluster_selection_epsilon: ClusteringParameter;
  metric: ClusteringParameter;
  
  // 🆕 PARAMETRI AVANZATI HDBSCAN
  cluster_selection_method: ClusteringParameter;
  alpha: ClusteringParameter;
  max_cluster_size: ClusteringParameter;
  allow_single_cluster: ClusteringParameter;
  
  // 🆕 PARAMETRO PREPROCESSING
  only_user: ClusteringParameter;  // 🎯 Filtra solo messaggi utente
  
  // 🗂️ PARAMETRI UMAP
  use_umap: ClusteringParameter;           // Abilita/disabilita UMAP
  umap_n_neighbors: ClusteringParameter;   // Numero di vicini UMAP
  umap_min_dist: ClusteringParameter;      // Distanza minima UMAP
  umap_metric: ClusteringParameter;        // Metrica distanza UMAP
  umap_n_components: ClusteringParameter;  // Dimensioni output UMAP
  umap_random_state: ClusteringParameter;  // Seed random UMAP
  
  // 🎯 PARAMETRI REVIEW QUEUE - SOGLIE CONFIDENZA
  outlier_confidence_threshold: ClusteringParameter;           // Soglia confidenza OUTLIER
  propagated_confidence_threshold: ClusteringParameter;        // Soglia confidenza PROPAGATO
  representative_confidence_threshold: ClusteringParameter;    // Soglia confidenza RAPPRESENTATIVO
  
  // 🎯 PARAMETRI REVIEW QUEUE - CONFIGURAZIONE
  minimum_consensus_threshold: ClusteringParameter;            // Soglia consenso minimo
  enable_smart_review: ClusteringParameter;                    // Abilita review intelligente
  max_pending_per_batch: ClusteringParameter;                 // Massimo casi pending per batch
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
 * - Test clustering rapido per validazione parametri
 * 
 * Props: Nessuna
 * 
 * Ultima modifica: 25/08/2025
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
  
  // Stati per test clustering
  const [testLoading, setTestLoading] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [testDialogOpen, setTestDialogOpen] = useState(false);

  /**
   * Carica i parametri di clustering attuali
   * 
   * Input: selectedTenant dal contesto
   * Output: Aggiorna state parameters
   */
  const loadParameters = useCallback(async () => {
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
  }, [selectedTenant?.tenant_id]);

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
  const updateParameter = (paramName: string, newValue: number | string | boolean) => {
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
   * Esegue test clustering con i parametri attuali
   * 
   * Input: Nessuno (usa parametri correnti)
   * Output: Apre dialog con risultati test
   */
  const runClusteringTest = async () => {
    if (!selectedTenant?.tenant_id || !parameters) return;

    setTestLoading(true);
    setError(null);

    try {
      // Prepara parametri per il test (usa TUTTI i valori correnti e assicura tipi corretti)
      // 🔧 [FIX] Inclusione di TUTTI i parametri disponibili, non solo i 4 base
      const testParameters: Record<string, any> = {};
      
      // Itera attraverso TUTTI i parametri disponibili
      Object.entries(parameters).forEach(([paramName, paramConfig]) => {
        const value = paramConfig.value;
        
        // Gestione tipo-specifica dei valori
        if (typeof value === 'number') {
          testParameters[paramName] = value;
        } else if (typeof value === 'string') {
          // Prova a convertire stringhe numeriche a number per parametri che dovrebbero essere numerici
          const numericParams = [
            'min_cluster_size', 'min_samples', 'cluster_selection_epsilon', 'alpha', 'max_cluster_size',
            'umap_n_neighbors', 'umap_min_dist', 'umap_n_components', 'umap_random_state'
          ];
          
          if (numericParams.includes(paramName)) {
            const numValue = parseFloat(value);
            testParameters[paramName] = isNaN(numValue) ? value : numValue;
          } else {
            testParameters[paramName] = value;
          }
        } else if (typeof value === 'boolean') {
          testParameters[paramName] = value;
        } else {
          // Fallback per altri tipi
          testParameters[paramName] = value;
        }
      });
      
      console.log('🧪 [CLUSTERING TEST] Avvio test con TUTTI i parametri disponibili:', testParameters);
      console.log('📊 [CLUSTERING TEST] Numero parametri inviati:', Object.keys(testParameters).length);
      console.log('🗂️  [CLUSTERING TEST] Parametri UMAP inclusi:', {
        use_umap: testParameters.use_umap,
        umap_n_neighbors: testParameters.umap_n_neighbors,
        umap_min_dist: testParameters.umap_min_dist,
        umap_n_components: testParameters.umap_n_components,
        umap_metric: testParameters.umap_metric,
        umap_random_state: testParameters.umap_random_state
      });

      console.log('🧪 [CLUSTERING TEST] Avvio test con parametri:', testParameters);

      // Chiamata API per test clustering
      const result = await apiService.testClustering(
        selectedTenant.tenant_id,
        testParameters,
        1000  // Campione di 1000 conversazioni per test rapido
      );

      console.log('✅ [CLUSTERING TEST] Risultati ricevuti:', result);
      console.log('🔍 [CLUSTERING TEST] Tipo detailed_clusters:', typeof result.detailed_clusters, result.detailed_clusters);
      console.log('🔍 [CLUSTERING TEST] Array.isArray(detailed_clusters):', Array.isArray(result.detailed_clusters));
      console.log('🔍 [CLUSTERING TEST] Lunghezza array detailed_clusters:', Array.isArray(result.detailed_clusters) ? result.detailed_clusters.length : 'N/A');
      console.log('🔍 [CLUSTERING TEST] Primo cluster:', Array.isArray(result.detailed_clusters) && result.detailed_clusters.length > 0 ? result.detailed_clusters[0] : 'Nessun cluster');
      console.log('🔍 [CLUSTERING TEST] result.statistics:', result.statistics);
      console.log('🔍 [CLUSTERING TEST] result.quality_metrics:', result.quality_metrics);
      console.log('🔍 [CLUSTERING TEST] result.outlier_analysis:', result.outlier_analysis);
      
      // Mappa la risposta del backend alla struttura che si aspetta il componente ClusteringTestResults
      const mappedResult = {
        success: result.success,
        error: result.error,
        execution_time: result.execution_time,
        statistics: result.statistics ? {
          total_conversations: result.statistics.total_conversations,
          n_clusters: result.statistics.n_clusters,
          n_outliers: result.statistics.n_outliers,
          clustering_ratio: result.statistics.clustering_ratio || 0
        } : undefined,
        quality_metrics: result.quality_metrics ? {
          silhouette_score: result.quality_metrics.silhouette_score,
          calinski_harabasz_score: 0, // Non disponibile nel backend attuale
          davies_bouldin_score: 0 // Non disponibile nel backend attuale
        } : undefined,
        recommendations: result.quality_metrics?.quality_assessment ? [
          `📊 Valutazione: ${result.quality_metrics.quality_assessment}`,
          `📈 Bilanciamento cluster: ${result.quality_metrics.cluster_balance}`,
          ...(result.outlier_analysis?.recommendation ? [`💡 ${result.outlier_analysis.recommendation}`] : [])
        ] : undefined,
        detailed_clusters: result.detailed_clusters && Array.isArray(result.detailed_clusters) ? {
          clusters: result.detailed_clusters.map((cluster: any) => ({
            cluster_id: cluster.cluster_id || cluster.id || 0,
            size: cluster.size || cluster.count || 0,
            conversations: (cluster.conversations && Array.isArray(cluster.conversations)) ? 
              cluster.conversations.map((conv: any) => ({
                session_id: conv.session_id || conv.id || '',
                text: conv.testo_completo || conv.text || '',
                text_length: (conv.testo_completo || conv.text || '').length
              })) : []
          }))
        } : { clusters: [] },
        outlier_analysis: result.outlier_analysis ? {
          count: result.outlier_analysis.count,
          percentage: result.outlier_analysis.ratio * 100,
          samples: result.outlier_analysis.sample_outliers?.map((outlier: any) => ({
            session_id: outlier.session_id || 'unknown',
            text: outlier.testo_completo || outlier.text || 'N/A',
            text_length: (outlier.testo_completo || outlier.text || '').length
          })) || []
        } : undefined,
        
        // 🆕 MAPPING DATI VISUALIZZAZIONE dal backend
        visualization_data: result.visualization_data ? {
          points: result.visualization_data.points || [],
          cluster_colors: result.visualization_data.cluster_colors || {},
          statistics: result.visualization_data.statistics || {
            total_points: 0,
            n_clusters: 0,
            n_outliers: 0,
            dimensions: 0
          },
          coordinates: result.visualization_data.coordinates || {
            tsne_2d: [],
            pca_2d: [],
            pca_3d: []
          }
        } : undefined
      };

      console.log('🔍 [CLUSTERING TEST] Mapped result:', mappedResult);
      console.log('🔍 [CLUSTERING TEST] mappedResult.statistics:', mappedResult.statistics);
      console.log('🔍 [CLUSTERING TEST] mappedResult.success:', mappedResult.success);

      setTestResult(mappedResult);
      setTestLoading(false);  // 🔧 Imposto loading a false PRIMA di aprire il dialog
      setTestDialogOpen(true);

      if (result.success && result.statistics) {
        setSuccess(`Test completato! ${result.statistics.n_clusters} cluster trovati in ${result.execution_time?.toFixed(2)}s`);
      }

    } catch (err: any) {
      console.error('❌ [CLUSTERING TEST] Errore:', err);
      setError(`Errore test clustering: ${err.message}`);
      
      // In caso di errore, mostra comunque il dialog con l'errore
      setTestResult({
        success: false,
        error: err.message,
        tenant_id: selectedTenant.tenant_id,
        execution_time: 0,
        sample_info: { total_conversations: 0, embedding_dimension: 0, parameters_used: {} },
        statistics: { total_conversations: 0, n_clusters: 0, n_outliers: 0, clustering_ratio: 0, parameters_used: {} },
        detailed_clusters: [],
        quality_metrics: { silhouette_score: 0, outlier_ratio: 0, cluster_balance: 'error', quality_assessment: 'error' },
        outlier_analysis: { count: 0, ratio: 0, analysis: 'error', recommendation: '', sample_outliers: [] }
      });
      setTestLoading(false);  // 🔧 Imposto loading a false PRIMA di aprire il dialog
      setTestDialogOpen(true);
      
    } finally {
      // Loading già impostato a false sopra
    }
  };

  /**
   * Valida un parametro singolo
   * 
   * Input: Nome parametro, valore, definizione parametro
   * Output: Aggiorna validationErrors
   */
  const validateParameter = (paramName: string, value: number | string | boolean, paramDef: ClusteringParameter) => {
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

    if (typeof value === 'boolean' && paramDef.options && !paramDef.options.includes(value)) {
      errors[paramName] = `Valore boolean non valido`;
    }

    setValidationErrors(errors);
  };

  /**
   * Determina l'icona di stato per un parametro
   * 
   * Input: Nome parametro, valore attuale
   * Output: Elemento JSX con icona appropriata
   */
  const getParameterStatusIcon = (paramName: string, currentValue: number | string | boolean) => {
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
    const isBoolean = typeof param.value === 'boolean';
    const isSelect = param.options && param.options.length > 0 && !isBoolean;
    const hasError = validationErrors[paramName];

    // 🆕 GESTIONE PARAMETRI BOOLEAN (Switch)
    if (isBoolean) {
      return (
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                checked={param.value as boolean}
                onChange={(e) => updateParameter(paramName, e.target.checked)}
                color={hasError ? "error" : "primary"}
              />
            }
            label={
              <Box>
                <Typography variant="body2" fontWeight="medium">
                  {param.description}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {param.value ? 'Attivato' : 'Disattivato'}
                  {param.value === param.default && ' (Default)'}
                </Typography>
              </Box>
            }
          />
          {hasError && (
            <Typography variant="caption" color="error" sx={{ mt: 0.5 }}>
              {validationErrors[paramName]}
            </Typography>
          )}
        </FormGroup>
      );
    }

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
              <MenuItem key={String(option)} value={String(option)}>
                {String(option)}
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
  }, [selectedTenant, loadParameters]);

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

  /**
   * Callback per caricare parametri da una versione specifica
   */
  const handleLoadParametersFromVersion = useCallback((versionParameters: any) => {
    if (!versionParameters) return;

    console.log('🔧 Caricamento parametri da versione:', versionParameters);

    // Crea un nuovo oggetto parametri con i valori caricati
    const updatedParameters: ClusteringParameters = { ...parameters } as ClusteringParameters;

    // Aggiorna ogni parametro se presente nei dati della versione
    Object.keys(updatedParameters).forEach(key => {
      if (versionParameters[key] !== undefined) {
        updatedParameters[key as keyof ClusteringParameters] = {
          ...updatedParameters[key as keyof ClusteringParameters],
          value: versionParameters[key]
        };
      }
    });

    setParameters(updatedParameters);
    setSuccess('✅ Parametri caricati dalla versione con successo!');
    setHasChanges(true); // Indica che ci sono modifiche da salvare
    console.log('✅ Parametri caricati dalla versione con successo');
  }, [parameters]);

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
                label={selectedTenant.tenant_name} 
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
                {Object.entries(parameters)
                  .filter(([paramName]) => !paramName.startsWith('umap') && paramName !== 'use_umap') // 🆕 Esclude parametri UMAP dalla sezione principale
                  .map(([paramName, param]) => (
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
                            
                            {/* 🆕 BADGE SUPPORTO GPU */}
                            {param.gpu_supported === false && (
                              <Tooltip title={param.gpu_warning || 'Parametro non supportato su GPU'}>
                                <Chip 
                                  label="CPU Only" 
                                  size="small" 
                                  color="warning" 
                                  variant="outlined"
                                  sx={{ ml: 1, fontSize: '0.7rem' }}
                                />
                              </Tooltip>
                            )}
                            {param.gpu_supported === true && (
                              <Tooltip title="Parametro supportato sia su CPU che GPU">
                                <Chip 
                                  label="GPU ✓" 
                                  size="small" 
                                  color="success" 
                                  variant="outlined"
                                  sx={{ ml: 1, fontSize: '0.7rem' }}
                                />
                              </Tooltip>
                            )}
                            
                            <Tooltip title={param.explanation}>
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>

                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {param.explanation}
                          </Typography>
                          
                          {/* 🆕 AVVISO GPU SPECIFICO */}
                          {param.gpu_supported === false && param.gpu_warning && (
                            <Alert severity="warning" sx={{ mb: 2, fontSize: '0.8rem' }}>
                              <Typography variant="caption">
                                {param.gpu_warning}
                              </Typography>
                            </Alert>
                          )}

                          {/* Controllo parametro */}
                          {renderParameterControl(paramName, param)}                          {/* Impatto */}
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

              {/* 🆕 SEZIONE UMAP */}
              <Paper sx={{ p: 3, mt: 3, bgcolor: 'background.default' }}>
                <Box display="flex" alignItems="center" mb={2}>
                  <Typography variant="h5" component="h2" sx={{ flexGrow: 1 }}>
                    UMAP - Dimensionality Reduction
                  </Typography>
                  <Chip 
                    label="Preprocessing" 
                    size="small" 
                    color="info" 
                    variant="outlined"
                    sx={{ ml: 1 }}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  UMAP (Uniform Manifold Approximation and Projection) riduce la dimensionalità 
                  degli embeddings preservando la struttura locale dei dati prima di applicare HDBSCAN.
                  Migliora significativamente le performance del clustering su dataset di grandi dimensioni.
                </Typography>

                {/* Toggle principale UMAP */}
                <Box sx={{ mb: 3 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={parameters?.use_umap?.value as boolean || false}
                        onChange={(e) => updateParameter('use_umap', e.target.checked)}
                        color="primary"
                      />
                    }
                    label={
                      <Box>
                        <Typography variant="subtitle1" component="span">
                          Abilita UMAP Preprocessing
                        </Typography>
                        <Typography variant="caption" display="block" color="text.secondary">
                          Applica riduzione dimensionale prima del clustering
                        </Typography>
                      </Box>
                    }
                  />
                </Box>

                {/* Parametri UMAP - visibili solo se abilitato */}
                {parameters?.use_umap?.value && (
                  <Collapse in={parameters?.use_umap?.value as boolean} timeout="auto" unmountOnExit>
                    <Box display="flex" flexWrap="wrap" gap={2}>
                      
                      {/* N Neighbors */}
                      <Box flex="1 1 300px" minWidth="250px">
                        <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                              N Neighbors
                            </Typography>
                            <Tooltip title="Numero di vicini considerati per la costruzione del grafo locale">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                            <strong>📊 Impatto delle variazioni:</strong><br/>
                            • <strong>Valori bassi (5-10):</strong> Preserva struttura locale, crea cluster più piccoli e dettagliati<br/>
                            • <strong>Valori medi (15-30):</strong> Bilanciamento ottimale globale/locale (raccomandato)<br/>
                            • <strong>Valori alti (50+):</strong> Preserva struttura globale, cluster più grandi ma meno precisi<br/>
                            💡 <strong>Consiglio:</strong> Inizia con 15-20 per dati testuali
                          </Typography>
                          <Slider
                            value={parameters?.umap_n_neighbors?.value as number || 15}
                            onChange={(_, value) => updateParameter('umap_n_neighbors', value)}
                            min={5}
                            max={100}
                            step={5}
                            marks={[
                              { value: 15, label: '15' },
                              { value: 30, label: '30' },
                              { value: 50, label: '50' }
                            ]}
                            valueLabelDisplay="auto"
                          />
                        </Paper>
                      </Box>

                      {/* Min Distance */}
                      <Box flex="1 1 300px" minWidth="250px">
                        <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                              Min Distance
                            </Typography>
                            <Tooltip title="Distanza minima tra punti nel low-dimensional embedding">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                            <strong>📊 Impatto delle variazioni:</strong><br/>
                            • <strong>0.0:</strong> Punti molto vicini, clustering denso e compatto<br/>
                            • <strong>0.1 (raccomandato):</strong> Bilanciamento ottimale densità/separazione<br/>
                            • <strong>0.3+:</strong> Maggiore dispersione, cluster meno densi ma più separati<br/>
                            💡 <strong>Consiglio:</strong> 0.1 funziona bene per la maggior parte dei casi
                          </Typography>
                          <Slider
                            value={parameters?.umap_min_dist?.value as number || 0.1}
                            onChange={(_, value) => updateParameter('umap_min_dist', value)}
                            min={0.0}
                            max={1.0}
                            step={0.05}
                            marks={[
                              { value: 0.0, label: '0.0' },
                              { value: 0.1, label: '0.1' },
                              { value: 0.3, label: '0.3' }
                            ]}
                            valueLabelDisplay="auto"
                          />
                        </Paper>
                      </Box>

                      {/* Metric */}
                      <Box flex="1 1 300px" minWidth="250px">
                        <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                              Distance Metric
                            </Typography>
                            <Tooltip title="Metrica di distanza utilizzata per il calcolo delle similarità">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                            <strong>📊 Impatto delle variazioni:</strong><br/>
                            • <strong>Cosine:</strong> Ottimale per testi, ignora lunghezza e si concentra su direzione semantica<br/>
                            • <strong>Euclidean:</strong> Distanza geometrica standard, può essere influenzato dalla magnitudo<br/>
                            • <strong>Manhattan:</strong> Distanza L1, meno sensibile agli outlier<br/>
                            💡 <strong>Raccomandazione:</strong> Usa sempre <strong>cosine</strong> per embeddings testuali
                          </Typography>
                          <FormControl fullWidth size="small">
                            <Select
                              value={parameters?.umap_metric?.value as string || 'cosine'}
                              onChange={(e) => updateParameter('umap_metric', e.target.value)}
                            >
                              <MenuItem value="cosine">Cosine</MenuItem>
                              <MenuItem value="euclidean">Euclidean</MenuItem>
                              <MenuItem value="manhattan">Manhattan</MenuItem>
                              <MenuItem value="correlation">Correlation</MenuItem>
                            </Select>
                          </FormControl>
                        </Paper>
                      </Box>

                      {/* N Components */}
                      <Box flex="1 1 300px" minWidth="250px">
                        <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                              Target Dimensions
                            </Typography>
                            <Tooltip title="Numero di dimensioni dell'embedding ridotto">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                            <strong>📊 Impatto delle variazioni:</strong><br/>
                            • <strong>2-5 dim:</strong> Massima riduzione, solo per visualizzazione 2D/3D<br/>
                            • <strong>10-20 dim:</strong> Buon bilanciamento riduzione/informazione (raccomandato)<br/>
                            • <strong>50+ dim:</strong> Preserva più informazione ma aumenta tempo computazionale<br/>
                            💡 <strong>Consiglio:</strong> 10-15 dimensioni per clustering efficace (da 768→12 nell'ottimale)
                          </Typography>
                          <Slider
                            value={parameters?.umap_n_components?.value as number || 50}
                            onChange={(_, value) => updateParameter('umap_n_components', value)}
                            min={2}
                            max={100}
                            step={5}
                            marks={[
                              { value: 2, label: '2' },
                              { value: 50, label: '50' },
                              { value: 100, label: '100' }
                            ]}
                            valueLabelDisplay="auto"
                          />
                        </Paper>
                      </Box>

                      {/* Random State */}
                      <Box flex="1 1 300px" minWidth="250px">
                        <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                          <Box display="flex" alignItems="center" mb={1}>
                            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                              Random Seed
                            </Typography>
                            <Tooltip title="Seed per la riproducibilità dei risultati">
                              <IconButton size="small">
                                <InfoIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                            <strong>📊 Scopo:</strong> Garantisce risultati identici tra esecuzioni multiple<br/>
                            • <strong>Valore fisso (es. 42):</strong> Risultati riproducibili per confronti<br/>
                            • <strong>Cambiare valore:</strong> Esplora variazioni casuali nell'algoritmo<br/>
                            💡 <strong>Consiglio:</strong> Mantieni 42 per consistenza, cambia solo per testare robustezza
                          </Typography>
                          <TextField
                            type="number"
                            value={parameters?.umap_random_state?.value as number || 42}
                            onChange={(e) => updateParameter('umap_random_state', parseInt(e.target.value) || 42)}
                            size="small"
                            fullWidth
                            inputProps={{ min: 1, max: 999999 }}
                          />
                        </Paper>
                      </Box>

                    </Box>

                    {/* Alert informativo sulla performance */}
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        <strong>Performance:</strong> UMAP può richiedere tempo aggiuntivo per la riduzione dimensionale, 
                        ma generalmente migliora la qualità del clustering e riduce i tempi di HDBSCAN.
                        Ideale per dataset con embeddings ad alta dimensionalità (768D → 50D tipico).
                      </Typography>
                    </Alert>
                  </Collapse>
                )}
              </Paper>

              {/* 🎯 SEZIONE REVIEW QUEUE - SOGLIE CONFIDENZA */}
              <Paper sx={{ p: 3, mt: 3, bgcolor: 'background.default' }}>
                <Box display="flex" alignItems="center" mb={2}>
                  <Typography variant="h5" component="h2" sx={{ flexGrow: 1 }}>
                    Review Queue - Soglie Confidenza
                  </Typography>
                  <Chip 
                    label="Post-processing" 
                    size="small" 
                    color="secondary" 
                    variant="outlined"
                    sx={{ ml: 1 }}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Configura le soglie di confidenza per determinare quali casi richiedono revisione umana.
                  I casi sotto le soglie definite vengono inseriti nella Review Queue per validazione manuale.
                </Typography>

                <Box display="flex" flexWrap="wrap" gap={2}>
                  
                  {/* Soglia Outlier */}
                  {parameters?.outlier_confidence_threshold && (
                    <Box flex="1 1 300px" minWidth="250px">
                      <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            Soglia Outlier
                          </Typography>
                          <Tooltip title="Soglia di confidenza per casi OUTLIER">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                          Casi OUTLIER con confidenza sotto questa soglia vanno in review
                        </Typography>
                        <Slider
                          value={parameters.outlier_confidence_threshold.value as number}
                          onChange={(_, newValue) => updateParameter('outlier_confidence_threshold', newValue)}
                          min={parameters.outlier_confidence_threshold.min || 0.1}
                          max={parameters.outlier_confidence_threshold.max || 1.0}
                          step={parameters.outlier_confidence_threshold.step || 0.01}
                          marks={[
                            { value: 0.1, label: '0.1' },
                            { value: 0.5, label: '0.5' },
                            { value: 1.0, label: '1.0' }
                          ]}
                          valueLabelDisplay="on"
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Box>
                  )}

                  {/* Soglia Propagato */}
                  {parameters?.propagated_confidence_threshold && (
                    <Box flex="1 1 300px" minWidth="250px">
                      <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            Soglia Propagato
                          </Typography>
                          <Tooltip title="Soglia di confidenza per casi PROPAGATO">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                          Casi PROPAGATO con confidenza sotto questa soglia vanno in review
                        </Typography>
                        <Slider
                          value={parameters.propagated_confidence_threshold.value as number}
                          onChange={(_, newValue) => updateParameter('propagated_confidence_threshold', newValue)}
                          min={parameters.propagated_confidence_threshold.min || 0.1}
                          max={parameters.propagated_confidence_threshold.max || 1.0}
                          step={parameters.propagated_confidence_threshold.step || 0.01}
                          marks={[
                            { value: 0.1, label: '0.1' },
                            { value: 0.5, label: '0.5' },
                            { value: 1.0, label: '1.0' }
                          ]}
                          valueLabelDisplay="on"
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Box>
                  )}

                  {/* Soglia Rappresentativo */}
                  {parameters?.representative_confidence_threshold && (
                    <Box flex="1 1 300px" minWidth="250px">
                      <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            Soglia Rappresentativo
                          </Typography>
                          <Tooltip title="Soglia di confidenza per casi RAPPRESENTATIVO">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                          Casi RAPPRESENTATIVO con confidenza sotto questa soglia vanno in review
                        </Typography>
                        <Slider
                          value={parameters.representative_confidence_threshold.value as number}
                          onChange={(_, newValue) => updateParameter('representative_confidence_threshold', newValue)}
                          min={parameters.representative_confidence_threshold.min || 0.1}
                          max={parameters.representative_confidence_threshold.max || 1.0}
                          step={parameters.representative_confidence_threshold.step || 0.01}
                          marks={[
                            { value: 0.1, label: '0.1' },
                            { value: 0.5, label: '0.5' },
                            { value: 1.0, label: '1.0' }
                          ]}
                          valueLabelDisplay="on"
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Box>
                  )}

                  {/* Soglia Consenso Minimo */}
                  {parameters?.minimum_consensus_threshold && (
                    <Box flex="1 1 300px" minWidth="250px">
                      <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            Consenso Minimo
                          </Typography>
                          <Tooltip title="Numero minimo di algoritmi che devono concordare">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                          Numero minimo di algoritmi concordi per auto-classificazione
                        </Typography>
                        <Slider
                          value={parameters.minimum_consensus_threshold.value as number}
                          onChange={(_, newValue) => updateParameter('minimum_consensus_threshold', newValue)}
                          min={parameters.minimum_consensus_threshold.min || 1}
                          max={parameters.minimum_consensus_threshold.max || 5}
                          step={parameters.minimum_consensus_threshold.step || 1}
                          marks={[
                            { value: 1, label: '1' },
                            { value: 2, label: '2' },
                            { value: 3, label: '3' },
                            { value: 4, label: '4' },
                            { value: 5, label: '5' }
                          ]}
                          valueLabelDisplay="on"
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Box>
                  )}

                  {/* Max Pending per Batch */}
                  {parameters?.max_pending_per_batch && (
                    <Box flex="1 1 300px" minWidth="250px">
                      <Paper sx={{ p: 2, height: '100%', bgcolor: 'background.paper' }}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
                            Max Pending/Batch
                          </Typography>
                          <Tooltip title="Massimo numero di casi pending per ogni batch">
                            <IconButton size="small">
                              <InfoIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                          Limite massimo casi da inviare in review per batch
                        </Typography>
                        <Slider
                          value={parameters.max_pending_per_batch.value as number}
                          onChange={(_, newValue) => updateParameter('max_pending_per_batch', newValue)}
                          min={parameters.max_pending_per_batch.min || 10}
                          max={parameters.max_pending_per_batch.max || 500}
                          step={parameters.max_pending_per_batch.step || 10}
                          marks={[
                            { value: 50, label: '50' },
                            { value: 150, label: '150' },
                            { value: 300, label: '300' },
                            { value: 500, label: '500' }
                          ]}
                          valueLabelDisplay="on"
                          sx={{ mt: 1 }}
                        />
                      </Paper>
                    </Box>
                  )}

                </Box>

                {/* Toggle Smart Review */}
                {parameters?.enable_smart_review && (
                  <Box sx={{ mt: 3 }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={parameters.enable_smart_review.value as boolean || false}
                          onChange={(e) => updateParameter('enable_smart_review', e.target.checked)}
                          color="primary"
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="subtitle1" component="span">
                            Abilita Review Intelligente
                          </Typography>
                          <Typography variant="caption" display="block" color="text.secondary">
                            Utilizza logiche avanzate per prioritizzare i casi da rivedere
                          </Typography>
                        </Box>
                      }
                    />
                  </Box>
                )}

                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    <strong>Come funziona:</strong> I casi con confidenza sotto le soglie definite 
                    vengono automaticamente inseriti nella Review Queue per validazione umana. 
                    Soglie più basse = più casi in review (maggior precisione), 
                    soglie più alte = meno casi in review (maggior automazione).
                  </Typography>
                </Alert>
              </Paper>

              {/* Azioni */}
              <Box display="flex" justifyContent="space-between" alignItems="center" mt={3}>
                <Box display="flex" gap={2}>
                  <Button
                    variant="outlined"
                    startIcon={<ResetIcon />}
                    onClick={resetParameters}
                    disabled={saving || loading}
                  >
                    Reset Default
                  </Button>
                  
                  {/* 🆕 TASTO PROVA CLUSTERING */}
                  <Button
                    variant="outlined"
                    color="primary"
                    startIcon={testLoading ? <CircularProgress size={16} /> : <AssessmentIcon />}
                    onClick={runClusteringTest}
                    disabled={saving || loading || testLoading || Object.keys(validationErrors).length > 0}
                    sx={{
                      fontWeight: 'bold',
                      borderWidth: 2,
                      '&:hover': {
                        borderWidth: 2,
                        backgroundColor: 'primary.50'
                      }
                    }}
                  >
                    {testLoading ? 'Testing...' : 'PROVA CLUSTERING'}
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

      {/* 🆕 SEZIONE CRONOLOGIA E VERSIONING CLUSTERING */}
      <Box sx={{ mt: 3 }}>
        <ClusteringVersionManager onLoadParameters={handleLoadParametersFromVersion} />
      </Box>
      
      {/* 🆕 DIALOG RISULTATI TEST CLUSTERING */}
      <ClusteringTestResults
        open={testDialogOpen}
        onClose={() => setTestDialogOpen(false)}
        result={testResult}
        isLoading={testLoading}
      />
    </Box>
  );
};

export default ClusteringParametersManager;
