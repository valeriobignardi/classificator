/**
 * File: AIConfigurationManager.tsx
 * Autore: GitHub Copilot
 * Data: 25/08/2025
 * Descrizione: Componente React per gestione configurazione AI (Embedding + LLM)
 * 
 * Storia aggiornamenti:
 * - 25/08/2025: Creazione componente con interfaccia completa per configurazione AI
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  IconButton,
  Tabs,
  Tab
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Memory as MemoryIcon,
  Psychology as PsychologyIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  BugReport as BugReportIcon,
  Computer as ComputerIcon,
  Storage as StorageIcon,
  Star as StarIcon,
  Tune as TuneIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import LLMConfigurationPage from './LLMConfigurationPage';

interface EmbeddingEngine {
  name: string;
  description: string;
  provider: string;
  model: string;
  embedding_dim: number;
  supports_gpu: boolean;
  supports_cpu: boolean;
  available: boolean;
  requirements: string[];
  pros: string[];
  cons: string[];
}

interface LLMModel {
  name: string;
  display_name: string;
  provider: string;  // 'ollama' | 'openai'
  max_input_tokens: number;
  max_output_tokens: number;
  context_limit: number;
  requires_raw_mode: boolean;
  parallel_calls_max?: number;
  rate_limit_per_minute?: number;
  rate_limit_per_day?: number;
  // Campi legacy per compatibilità
  description?: string;
  size?: string;
  category?: string;
  installed?: boolean;
  recommended?: boolean;
  modified_at?: string;
}

interface AIConfiguration {
  tenant_id: string;
  embedding_engine: {
    current: string;
    config: any;
    available_engines: string[];
  };
  llm_model: {
    current: string;
    available_models: any;
  };
  last_updated: string;
  status: {
    embedding_engine_ok: boolean;
    llm_model_ok: boolean;
    overall_status: string;
  };
}

interface DebugInfo {
  tenant_id: string;
  timestamp: string;
  embedding_engine: {
    type: string;
    status: string;
    test_details?: any;
    error?: string;
  };
  llm_model: {
    name: string;
    status: string;
    test_details?: any;
    error?: string;
  };
  system_status: {
    config_loaded: boolean;
    tenant_configs_loaded: boolean;
    ollama_connected: boolean;
    ollama_details?: any;
    ollama_error?: string;
  };
}

/**
 * Componente principale per gestione configurazione AI
 * 
 * Scopo: Permettere all'utente di cambiare embedding engines e modelli LLM
 * dall'interfaccia grafica con test e debug in tempo reale
 * 
 * Args:
 *   open: Se il componente è visibile
 * 
 * Returns:
 *   Interfaccia completa per configurazione AI
 * 
 * Data ultima modifica: 2025-08-25
 */
const AIConfigurationManager: React.FC<{ open: boolean }> = ({ open }) => {
  const { selectedTenant } = useTenant();
  const tenantId = selectedTenant?.tenant_id;

  // Stati per tab management
  const [currentTab, setCurrentTab] = useState(0);

  // Stati per embedding engines
  const [embeddingEngines, setEmbeddingEngines] = useState<Record<string, EmbeddingEngine>>({});
  const [selectedEmbeddingEngine, setSelectedEmbeddingEngine] = useState<string>('');
  const [embeddingEngineLoading, setEmbeddingEngineLoading] = useState(false);

  // Stati per modelli LLM
  const [llmModels, setLlmModels] = useState<any>(null);
  const [selectedLlmModel, setSelectedLlmModel] = useState<string>('');
  const [llmModelLoading, setLlmModelLoading] = useState(false);

  // Stati generali
  const [configuration, setConfiguration] = useState<AIConfiguration | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Stati debug
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [debugDialogOpen, setDebugDialogOpen] = useState(false);
  const [debugLoading, setDebugLoading] = useState(false);

  /**
   * Carica configurazione AI completa del tenant
   */
  const loadConfiguration = useCallback(async () => {
    if (!tenantId) return;

    setLoading(true);
    setError(null);

    try {
      // Carica configurazione corrente
      const configResponse = await apiService.getAIConfiguration(tenantId);
      if (configResponse.success) {
        setConfiguration(configResponse.configuration);
        setSelectedEmbeddingEngine(configResponse.configuration.embedding_engine?.current || '');
        setSelectedLlmModel(configResponse.configuration.llm_model?.current || '');
      }

      // Carica embedding engines disponibili
      const embeddingResponse = await apiService.getEmbeddingEngines(tenantId);
      if (embeddingResponse.success && embeddingResponse.engines) {
        setEmbeddingEngines(embeddingResponse.engines);
      }

      // Carica modelli LLM disponibili (nuova API unificata)
      try {
        const llmResponse = await apiService.getLLMModels(tenantId);
        if (llmResponse.success) {
          // Nuova API restituisce array piatto invece di struttura annidata
          console.log('✅ Modelli LLM caricati (nuova API):', llmResponse.models);
          console.log('🔍 Struttura primo modello:', llmResponse.models[0]);
          setLlmModels({ 
            models: llmResponse.models || [],
            success: true 
          });
        }
      } catch (llmError) {
        console.warn('Errore caricamento modelli LLM:', llmError);
        // Non bloccare il caricamento se i modelli LLM non sono disponibili
      }

    } catch (err) {
      setError(`Errore caricamento configurazione: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [tenantId]);

  // Carica configurazione iniziale
  useEffect(() => {
    if (tenantId && open) {
      loadConfiguration();
    }
  }, [tenantId, open, loadConfiguration]);

  /**
   * Imposta nuovo embedding engine
   */
  const handleSetEmbeddingEngine = async (engineType: string) => {
    if (!tenantId) return;

    setEmbeddingEngineLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await apiService.setEmbeddingEngine(tenantId, engineType, {});

      if (response.success) {
        setSuccess(`Embedding engine ${engineType} impostato con successo`);
        setSelectedEmbeddingEngine(engineType);
        // Ricarica configurazione
        await loadConfiguration();
      } else {
        setError(response.error || 'Errore impostazione embedding engine');
      }

    } catch (err) {
      setError(`Errore: ${err}`);
    } finally {
      setEmbeddingEngineLoading(false);
    }
  };

  /**
   * Imposta nuovo modello LLM
   */
  const handleSetLlmModel = async (modelName: string) => {
    if (!tenantId) return;

    setLlmModelLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // STEP 1: Imposta nuovo modello nel database
      const response = await apiService.setLLMModel(tenantId, modelName);

      if (response.success) {
        console.log('✅ [UI] Modello LLM salvato nel database:', modelName);
        
        // STEP 2: CORREZIONE CRITICA - Ricarica configurazione LLM nel server
        try {
          console.log('🔄 [UI] Ricaricamento configurazione LLM nel server...');
          const reloadResponse = await apiService.reloadLLMConfiguration(tenantId);
          
          if (reloadResponse.success) {
            console.log('✅ [UI] Configurazione LLM ricaricata nel server');
            console.log('🔄 [UI] Cambio modello:', reloadResponse.old_model, '->', reloadResponse.new_model);
            
            setSuccess(`Modello LLM ${modelName} impostato e attivato con successo`);
            setSelectedLlmModel(modelName);
            
            // STEP 3: Ricarica configurazione UI per riflettere i cambiamenti
            await loadConfiguration();
            
          } else {
            console.error('❌ [UI] Errore reload configurazione LLM:', reloadResponse.error);
            setError(`Modello salvato ma errore attivazione: ${reloadResponse.error}`);
          }
          
        } catch (reloadError) {
          console.error('❌ [UI] Errore durante reload LLM:', reloadError);
          setError(`Modello salvato ma errore attivazione: ${reloadError}`);
        }
        
      } else {
        setError(response.error || 'Errore impostazione modello LLM');
      }

    } catch (err) {
      setError(`Errore: ${err}`);
    } finally {
      setLlmModelLoading(false);
    }
  };

  /**
   * Carica informazioni debug
   */
  const loadDebugInfo = async () => {
    if (!tenantId) return;

    setDebugLoading(true);

    try {
      const response = await apiService.getAIDebugInfo(tenantId);
      if (response.success) {
        setDebugInfo(response.debug_info);
      }
    } catch (err) {
      console.error('Errore caricamento debug info:', err);
    } finally {
      setDebugLoading(false);
    }
  };

  /**
   * Apre dialog debug
   */
  const handleOpenDebug = async () => {
    setDebugDialogOpen(true);
    await loadDebugInfo();
  };

  /**
   * Renderizza stato disponibilità
   */
  const renderAvailabilityStatus = (available: boolean, requirements: string[]) => {
    if (available) {
      return (
        <Chip
          icon={<CheckCircleIcon />}
          label="Disponibile"
          color="success"
          size="small"
        />
      );
    } else {
      return (
        <Tooltip title={`Requisiti: ${requirements.join(', ')}`}>
          <span>
            <Chip
              icon={<ErrorIcon />}
              label="Non Disponibile"
              color="error"
              size="small"
            />
          </span>
        </Tooltip>
      );
    }
  };

  /**
   * Helper per ottenere configurazione status
   */
  const getStatusConfig = (status: string) => {
    const statusConfig = {
      ok: { color: 'success' as const, icon: <CheckCircleIcon />, label: 'OK' },
      partial: { color: 'warning' as const, icon: <WarningIcon />, label: 'Parziale' },
      error: { color: 'error' as const, icon: <ErrorIcon />, label: 'Errore' }
    };

    return statusConfig[status as keyof typeof statusConfig] || statusConfig.error;
  };

  /**
   * Renderizza lo status del sistema come chip
   */
  const renderSystemStatus = (status: string) => {
    const config = getStatusConfig(status);

    return (
      <Chip
        icon={config.icon}
        label={config.label}
        color={config.color}
        size="small"
      />
    );
  };

  if (!open || !tenantId) {
    return null;
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
          <SettingsIcon color="primary" />
          Configurazione AI
        </Typography>
        <Box>
          <Tooltip title="Debug Info">
            <IconButton onClick={handleOpenDebug} color="primary">
              <BugReportIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Aggiorna">
            <span>
              <IconButton onClick={loadConfiguration} disabled={loading} color="primary">
                <RefreshIcon />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </Box>

      {/* Tab Navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={currentTab} 
          onChange={(e, newValue) => setCurrentTab(newValue)}
          aria-label="AI configuration tabs"
        >
          <Tab 
            icon={<SettingsIcon />}
            label="Configurazione Base" 
            iconPosition="start"
          />
          <Tab 
            icon={<TuneIcon />}
            label="Configurazione Avanzata LLM" 
            iconPosition="start"
          />
        </Tabs>
      </Box>

      {/* Tab Panel 0: Configurazione Base (Existing) */}
      {currentTab === 0 && (
        <Box>
          {/* Status generale */}
          {configuration && !loading && !embeddingEngineLoading && !llmModelLoading && (
            <Alert
              severity={getStatusConfig(configuration.status?.overall_status || 'ok').color}
              icon={getStatusConfig(configuration.status?.overall_status || 'ok').icon}
              sx={{ mb: 3 }}
            >
              <strong>Stato Configurazione:</strong> {getStatusConfig(configuration.status?.overall_status || 'ok').label}
              <br />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Embedding Engine: {configuration.status?.embedding_engine_ok ? '✅' : '❌'} | 
                Modello LLM: {configuration.status?.llm_model_ok ? '✅' : '❌'}
              </Typography>
            </Alert>
          )}

          {/* Indicatore caricamento configurazione */}
          {(loading || embeddingEngineLoading || llmModelLoading) && (
            <Alert severity="info" sx={{ mb: 3 }}>
              <Box display="flex" alignItems="center" gap={1}>
                <CircularProgress size={16} />
                <Typography>
                  Caricamento configurazione AI in corso...
                </Typography>
              </Box>
            </Alert>
          )}

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

          {loading && (
            <Box display="flex" justifyContent="center" my={3}>
              <CircularProgress />
            </Box>
          )}

          <Stack spacing={3} direction={{ xs: 'column', md: 'row' }}>
            {/* Sezione Embedding Engines */}
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <MemoryIcon color="primary" />
                    Motore di Embedding
                  </Typography>

                  {configuration && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <strong>Attualmente in uso:</strong> {configuration.embedding_engine?.current || 'N/A'}
                    </Alert>
                  )}

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Seleziona Embedding Engine</InputLabel>
                    <Select
                      value={selectedEmbeddingEngine && embeddingEngines[selectedEmbeddingEngine] ? selectedEmbeddingEngine : ''}
                      onChange={(e) => setSelectedEmbeddingEngine(e.target.value)}
                      disabled={embeddingEngineLoading}
                    >
                      {Object.entries(embeddingEngines || {}).map(([key, engine]) => (
                        <MenuItem key={key} value={key} disabled={!engine.available}>
                          {engine.name} {!engine.available && ' (Non disponibile)'}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  {selectedEmbeddingEngine && embeddingEngines && embeddingEngines[selectedEmbeddingEngine] && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        {embeddingEngines[selectedEmbeddingEngine].name}
                      </Typography>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        {embeddingEngines[selectedEmbeddingEngine].description}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', my: 1 }}>
                        {renderAvailabilityStatus(
                          embeddingEngines[selectedEmbeddingEngine].available,
                          embeddingEngines[selectedEmbeddingEngine].requirements
                        )}
                        <Chip label={`${embeddingEngines[selectedEmbeddingEngine].embedding_dim}D`} size="small" />
                        <Chip 
                          icon={embeddingEngines[selectedEmbeddingEngine].supports_gpu ? <ComputerIcon /> : <StorageIcon />}
                          label={embeddingEngines[selectedEmbeddingEngine].provider} 
                          size="small" 
                        />
                      </Box>

                      <Accordion sx={{ mt: 2 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="body2">Dettagli Tecnici</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Stack direction="row" spacing={2}>
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="subtitle2" color="success.main">Pro:</Typography>
                              <List dense>
                                {(embeddingEngines[selectedEmbeddingEngine].pros || []).map((pro, idx) => (
                                  <ListItem key={idx} sx={{ py: 0 }}>
                                    <ListItemIcon sx={{ minWidth: 20 }}>
                                      <CheckCircleIcon color="success" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText primary={pro} />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="subtitle2" color="warning.main">Contro:</Typography>
                              <List dense>
                                {(embeddingEngines[selectedEmbeddingEngine].cons || []).map((con, idx) => (
                                  <ListItem key={idx} sx={{ py: 0 }}>
                                    <ListItemIcon sx={{ minWidth: 20 }}>
                                      <WarningIcon color="warning" fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText primary={con} />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          </Stack>
                        </AccordionDetails>
                      </Accordion>
                    </Box>
                  )}

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={() => handleSetEmbeddingEngine(selectedEmbeddingEngine)}
                    disabled={
                      !selectedEmbeddingEngine || 
                      embeddingEngineLoading || 
                      !embeddingEngines || 
                      !embeddingEngines[selectedEmbeddingEngine]?.available ||
                      (configuration ? selectedEmbeddingEngine === configuration.embedding_engine?.current : false)
                    }
                    startIcon={embeddingEngineLoading ? <CircularProgress size={20} /> : <SettingsIcon />}
                  >
                    {embeddingEngineLoading ? 'Configurando...' : 'Applica Embedding Engine'}
                  </Button>
                </CardContent>
              </Card>
            </Box>

            {/* Sezione Modelli LLM */}
            <Box sx={{ flex: 1 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <PsychologyIcon color="primary" />
                    Modello LLM
                  </Typography>

                  {configuration && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      <strong>Attualmente in uso:</strong> {configuration.llm_model?.current || 'N/A'}
                    </Alert>
                  )}

                  {/* Status Multi-Provider */}
                  {llmModels && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" component="div" sx={{ mb: 1 }}>
                        <strong>Modelli Disponibili:</strong>
                      </Typography>
                      
                      {/* Conta modelli OpenAI */}
                      {llmModels.models?.filter((m: any) => m.provider === 'openai').length > 0 && (
                        <Chip 
                          icon={<StarIcon />} 
                          label={`🤖 ${llmModels.models.filter((m: any) => m.provider === 'openai').length} OpenAI`} 
                          color="primary" 
                          size="small" 
                          sx={{ mr: 1, mb: 0.5 }}
                        />
                      )}
                      
                      {/* Conta modelli Ollama */}
                      {llmModels.models?.filter((m: any) => m.provider === 'ollama').length > 0 && (
                        <Chip 
                          icon={<CheckCircleIcon />} 
                          label={`🦙 ${llmModels.models.filter((m: any) => m.provider === 'ollama').length} Ollama`} 
                          color="success" 
                          size="small" 
                          sx={{ mr: 1, mb: 0.5 }}
                        />
                      )}
                      
                      <Typography variant="body2" color="textSecondary" component="div" sx={{ mt: 0.5 }}>
                        Totale: {llmModels.models?.length || 0} modelli configurati
                      </Typography>
                    </Box>
                  )}

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Seleziona Modello LLM</InputLabel>
                    <Select
                      value={selectedLlmModel || ''}
                      onChange={(e) => {
                        console.log('🔄 Modello selezionato:', e.target.value);
                        setSelectedLlmModel(e.target.value);
                      }}
                      disabled={llmModelLoading || !llmModels?.success}
                    >
                      {/* Lista semplificata senza separatori complessi */}
                      {llmModels?.models?.map((model: any) => (
                        <MenuItem key={model.name} value={model.name}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {model.provider === 'openai' ? (
                              <>
                                <StarIcon color="primary" fontSize="small" />
                                <strong>{model.display_name}</strong>
                                <Chip label="🤖 OpenAI" size="small" color="primary" />
                                <Chip label={`⚡ ${model.parallel_calls_max}`} size="small" />
                                <Chip label={`${Math.round(model.context_limit/1000)}K`} size="small" />
                              </>
                            ) : (
                              <>
                                <CheckCircleIcon color="success" fontSize="small" />
                                <span>{model.display_name || model.name}</span>
                                <Chip label="🦙 Ollama" size="small" color="success" />
                                <Chip label={`${Math.round(model.context_limit/1000)}K`} size="small" />
                              </>
                            )}
                          </Box>
                        </MenuItem>
                      )) || []}
                    </Select>
                  </FormControl>

                  {selectedLlmModel && llmModels?.models && (
                    <Alert severity="success" sx={{ mb: 2 }}>
                      <strong>Modello selezionato:</strong> {selectedLlmModel}
                      {(() => {
                        const model = llmModels.models.find((m: any) => m.name === selectedLlmModel);
                        if (model?.provider === 'openai' && model?.parallel_calls_max) {
                          return (
                            <Box sx={{ mt: 1 }}>
                              🤖 OpenAI • ⚡ {model.parallel_calls_max} chiamate parallele • 📊 {(model.context_limit/1000).toFixed(0)}K token
                            </Box>
                          );
                        } else if (model?.provider === 'ollama') {
                          return <Box sx={{ mt: 1 }}>🦙 Modello locale Ollama</Box>;
                        }
                        return null;
                      })()}
                    </Alert>
                  )}

                  <Button
                    variant="contained"
                    fullWidth
                    onClick={() => handleSetLlmModel(selectedLlmModel)}
                    disabled={
                      !selectedLlmModel || 
                      llmModelLoading || 
                      !llmModels?.success ||
                      (configuration ? selectedLlmModel === configuration.llm_model?.current : false)
                    }
                    startIcon={llmModelLoading ? <CircularProgress size={20} /> : <SettingsIcon />}
                  >
                    {llmModelLoading ? 'Configurando...' : 'Applica Modello LLM'}
                  </Button>
                </CardContent>
              </Card>
            </Box>
          </Stack>
        </Box>
      )}

      {/* Tab Panel 1: Configurazione Avanzata LLM */}
      {currentTab === 1 && tenantId && (
        <Box>
          <LLMConfigurationPage 
            tenantId={tenantId}
            onConfigurationChange={(config) => {
              console.log('Configurazione LLM cambiata:', config);
              // Ricarica la configurazione base se necessario
              loadConfiguration();
            }}
          />
        </Box>
      )}

      {/* Dialog Debug */}
      <Dialog 
        open={debugDialogOpen} 
        onClose={() => setDebugDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <BugReportIcon />
            Debug Informazioni - Modelli in Uso
          </Box>
        </DialogTitle>
        <DialogContent>
          {debugLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : debugInfo ? (
            <Box>
              {/* Timestamp */}
              <Alert severity="info" sx={{ mb: 2 }}>
                <strong>Ultimo aggiornamento:</strong> {new Date(debugInfo.timestamp).toLocaleString()}
              </Alert>

              <Stack spacing={2} direction={{ xs: 'column', md: 'row' }}>
                {/* Embedding Engine Status */}
                <Box sx={{ flex: 1 }}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Embedding Engine
                      </Typography>
                      <Box sx={{ mb: 1 }}>
                        <strong>Tipo:</strong> {debugInfo.embedding_engine.type}
                      </Box>
                      <Box sx={{ mb: 1 }}>
                        <strong>Status:</strong> {renderSystemStatus(debugInfo.embedding_engine.status)}
                      </Box>
                      {debugInfo.embedding_engine.test_details && (
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="body2">Dettagli Test</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                              {JSON.stringify(debugInfo.embedding_engine.test_details, null, 2)}
                            </pre>
                          </AccordionDetails>
                        </Accordion>
                      )}
                      {debugInfo.embedding_engine.error && (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          {debugInfo.embedding_engine.error}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Box>

                {/* LLM Model Status */}
                <Box sx={{ flex: 1 }}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Modello LLM
                      </Typography>
                      <Box sx={{ mb: 1 }}>
                        <strong>Nome:</strong> {debugInfo.llm_model.name}
                      </Box>
                      <Box sx={{ mb: 1 }}>
                        <strong>Status:</strong> {renderSystemStatus(debugInfo.llm_model.status)}
                      </Box>
                      {debugInfo.llm_model.test_details && (
                        <Accordion>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="body2">Dettagli Test</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                              {JSON.stringify(debugInfo.llm_model.test_details, null, 2)}
                            </pre>
                          </AccordionDetails>
                        </Accordion>
                      )}
                      {debugInfo.llm_model.error && (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          {debugInfo.llm_model.error}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Box>
              </Stack>

              {/* System Status */}
              <Box sx={{ mt: 2 }}>
                <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Stato Sistema
                      </Typography>
                      <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap' }}>
                        <Box>
                          <strong>Config Caricata:</strong> {
                            debugInfo.system_status.config_loaded ? '✅' : '❌'
                          }
                        </Box>
                        <Box>
                          <strong>Tenant Config:</strong> {
                            debugInfo.system_status.tenant_configs_loaded ? '✅' : '❌'
                          }
                        </Box>
                        <Box>
                          <strong>Ollama:</strong> {
                            debugInfo.system_status.ollama_connected ? '✅' : '❌'
                          }
                        </Box>
                      </Stack>
                      
                      {debugInfo.system_status.ollama_details && (
                        <Accordion sx={{ mt: 2 }}>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography variant="body2">Dettagli Ollama</Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <pre style={{ fontSize: '12px', overflow: 'auto' }}>
                              {JSON.stringify(debugInfo.system_status.ollama_details, null, 2)}
                            </pre>
                          </AccordionDetails>
                        </Accordion>
                      )}
                      
                      {debugInfo.system_status.ollama_error && (
                        <Alert severity="error" sx={{ mt: 1 }}>
                          <strong>Errore Ollama:</strong> {debugInfo.system_status.ollama_error}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Box>
            </Box>
          ) : (
            <Alert severity="warning">
              Nessuna informazione debug disponibile
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDebugDialogOpen(false)}>
            Chiudi
          </Button>
          <Button onClick={loadDebugInfo} disabled={debugLoading}>
            Aggiorna
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AIConfigurationManager;
