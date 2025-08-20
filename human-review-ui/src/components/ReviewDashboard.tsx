import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
  Alert,
  LinearProgress,
  IconButton,
  Tooltip,
  TextField,
  FormControlLabel,
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  PlayArrow as PlayIcon,
  Science as ScienceIcon,
  Assignment as AssignmentIcon,
  School as SchoolIcon,
  Settings as SettingsIcon,
  ModelTraining as ModelTrainingIcon,
  List as ListIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';
import { ReviewCase } from '../types/ReviewCase';
import AllSessionsView from './AllSessionsView';

interface ReviewDashboardProps {
  tenant: string;
  onCaseSelect: (caseItem: ReviewCase) => void;
  onCreateMockCases: () => void;
  refreshTrigger: number;
  loading: boolean;
}

const ReviewDashboard: React.FC<ReviewDashboardProps> = ({
  tenant,
  onCaseSelect,
  onCreateMockCases,
  refreshTrigger,
  loading
}) => {
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [uiConfig, setUiConfig] = useState<any>(null);
  const [customLimit, setCustomLimit] = useState<number | string>('');
  const [currentLimit, setCurrentLimit] = useState<number>(2000);  // AUMENTATO LIMITE INIZIALE: ora parte con 2000 per vedere tutte le sessioni

  // Stato per force_review toggle
  const [forceReview, setForceReview] = useState(false);
  
  // Stato per dialogo training supervisionato SEMPLIFICATO - 4 PARAMETRI
  const [trainingDialogOpen, setTrainingDialogOpen] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    max_sessions: 500,                // Numero massimo sessioni per review umana
    confidence_threshold: 0.7,        // Soglia di confidenza
    force_review: false,              // Forza revisione casi già revisionati
    disagreement_threshold: 0.3       // Soglia disagreement ensemble
  });

  // Stato per gestire le tab
  const [activeTab, setActiveTab] = useState(0);

  // Ref per debouncing delle chiamate API
  const lastLoadTime = useRef<number>(0);

  // Carica configurazione UI
  useEffect(() => {
    const loadUIConfig = async () => {
      try {
        const configResponse = await apiService.getUIConfig();
        setUiConfig(configResponse.config);
        // Inizializza il limit corrente dalla config
        const defaultLimit = configResponse.config?.review?.default_limit || 20;
        setCurrentLimit(defaultLimit);
      } catch (err) {
        console.error('Error loading UI config:', err);
        // Usa valori predefiniti se non riesce a caricare la config
        const fallbackConfig = {
          classification: {
            confidence_threshold: 0.7,
            force_retrain: true,
            max_sessions: null,
            debug_mode: false
          },
          review: {
            default_limit: 20
          },
          mock_cases: {
            default_count: 3
          }
        };
        setUiConfig(fallbackConfig);
        // Usa il default_limit anche nel fallback, non un valore hardcoded
        setCurrentLimit(fallbackConfig.review.default_limit);
      }
    };

    loadUIConfig();
  }, []);

  const loadCases = useCallback(async (limit?: number) => {
    // Debounce: evita troppe chiamate consecutive
    const now = Date.now();
    if (now - lastLoadTime.current < 1000) { // Massimo 1 chiamata al secondo
      return;
    }
    lastLoadTime.current = now;

    setDashboardLoading(true);
    setError(null);

    try {
      const effectiveLimit = limit || currentLimit;
      const response = await apiService.getReviewCases(tenant, effectiveLimit);
      setCases(response.cases);
    } catch (err) {
      setError('Errore nel caricamento dei casi');
      console.error('Error loading cases:', err);
    } finally {
      setDashboardLoading(false);
    }
  }, [tenant, currentLimit]);

  const handleRefreshCases = () => {
    loadCases();
  };

  const handleLimitChange = (newLimit: number) => {
    setCurrentLimit(newLimit);
    setCustomLimit('');
    loadCases(newLimit);
  };

  const handleCustomLimitSubmit = () => {
    const limit = typeof customLimit === 'string' ? parseInt(customLimit, 10) : customLimit;
    if (!isNaN(limit) && limit > 0) {  // RIMOSSO LIMITE MASSIMO: ora accetta qualsiasi numero positivo
      handleLimitChange(limit);
    }
  };

  const handleStartSupervisedTraining = async () => {
    setTrainingLoading(true);
    setError(null);
    setSuccessMessage(null);
    setTrainingDialogOpen(false);

    try {
      const response = await apiService.startSupervisedTraining(tenant, trainingConfig);
      
      // Messaggio di successo basato sulla nuova struttura della risposta
      const isNewClient = response.client_type === 'new';
      
      let strategyMessage = '';
      if (isNewClient) {
        strategyMessage = '🆕 Nuovo cliente: Classificazione con LLM standalone';
      } else {
        strategyMessage = '🔄 Cliente esistente: Ensemble ML + LLM';
      }
      
      let analysisMessage = `📊 DATASET COMPLETO: Tutte le discussioni analizzate per clustering`;
      
      setSuccessMessage(
        `🎯 Training supervisionato completato con successo!\n\n` +
        `${strategyMessage}\n` +
        `${analysisMessage}\n\n` +
        `✅ Auto-classificate: ${response.auto_classified || 0} sessioni (alta confidenza)\n` +
        `📝 In review: ${response.needs_review || 0} sessioni (necessitano revisione umana)\n` +
        `📋 Coda totale: ${response.current_queue_size || 0} casi pending\n\n` +
        `${response.no_review_limit ? '🚀 Nessun limite applicato sulla coda di review' : ''}\n` +
        `${response.pending_save ? '⚠️ Le auto-classificazioni non sono ancora salvate nel DB' : ''}\n\n` +
        `🔄 Prossimo passo: Rivedi i casi selezionati e poi usa "Riaddestra Modello"`
      );
      
      // Ricarica i casi dopo il training
      setTimeout(() => {
        loadCases();
      }, 1000);

    } catch (err: any) {
      setError(`Errore durante il training supervisionato: ${err.message}`);
      console.error('Supervised training error:', err);
    } finally {
      setTrainingLoading(false);
    }
  };

  const [retrainingLoading, setRetrainingLoading] = useState(false);

  const handleManualRetraining = async () => {
    setRetrainingLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await apiService.triggerManualRetraining(tenant);
      
      if (response.success) {
        setSuccessMessage(
          `Riaddestramento completato! Utilizzate ${response.decision_count} decisioni umane per aggiornare il modello. ` +
          `Timestamp: ${new Date(response.timestamp).toLocaleString()}`
        );
      } else {
        setError(response.message);
      }

    } catch (err: any) {
      setError(`Errore durante il riaddestramento: ${err.message}`);
      console.error('Retraining error:', err);
    } finally {
      setRetrainingLoading(false);
    }
  };

  const handleStartClassification = async () => {
    setClassificationLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const config = uiConfig?.classification || {};
      const response = await apiService.startFullClassification(tenant, {
        confidence_threshold: config.confidence_threshold || 0.7,
        force_retrain: config.force_retrain !== false,
        max_sessions: config.max_sessions || null,
        debug_mode: config.debug_mode || false,
        force_review: forceReview
      });

      setSuccessMessage(
        `Classificazione completata: ${response.sessions_processed || 0} sessioni processate` +
        (response.forced_review_count > 0 ? `, ${response.forced_review_count} casi forzati in coda per revisione` : '')
      );
      
      // Ricarica i casi dopo la classificazione
      setTimeout(() => {
        loadCases();
      }, 1000);

    } catch (err: any) {
      setError(`Errore durante la classificazione: ${err.message}`);
      console.error('Classification error:', err);
    } finally {
      setClassificationLoading(false);
    }
  };

  useEffect(() => {
    if (uiConfig) {
      loadCases();
    }
  }, [loadCases, refreshTrigger, uiConfig]);

  // Ricarica casi quando necessario
  useEffect(() => {
    if (refreshTrigger > 0) {
      handleRefreshCases();
    }
  }, [refreshTrigger, handleRefreshCases]);

  // Callback quando viene aggiunta una sessione alla queue dal componente AllSessionsView
  const handleSessionAddedToQueue = useCallback((sessionId: string) => {
    setSuccessMessage(`Sessione ${sessionId} aggiunta alla review queue`);
    // Ricarica i casi per aggiornare la lista
    handleRefreshCases();
  }, [handleRefreshCases]);

  const getReasonColor = (reason: string) => {
    if (reason.toLowerCase().includes('disagreement')) return 'warning';
    if (reason.toLowerCase().includes('confidence')) return 'info';
    if (reason.toLowerCase().includes('uncertainty')) return 'secondary';
    return 'default';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Box>
      {/* Header with Actions */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Dashboard Revisione
        </Typography>
        <Box display="flex" alignItems="center" gap={1}>
          <Tooltip title="Aggiorna casi">
            <span>
              <IconButton onClick={handleRefreshCases} disabled={dashboardLoading}>
                <RefreshIcon />
              </IconButton>
            </span>
          </Tooltip>
          
          <Button
            variant="outlined"
            startIcon={<ScienceIcon />}
            onClick={() => {
              onCreateMockCases();
            }}
            disabled={loading || classificationLoading || trainingLoading || retrainingLoading || !uiConfig}
            sx={{ mr: 1 }}
          >
            Crea Casi Mock
          </Button>

          <Button
            variant="outlined"
            startIcon={trainingLoading ? undefined : <SchoolIcon />}
            onClick={() => setTrainingDialogOpen(true)}
            disabled={!uiConfig || trainingLoading || classificationLoading || retrainingLoading}
            color="secondary"
            sx={{ mr: 1 }}
          >
            {trainingLoading ? 'Training in corso...' : 'Training Supervisionato'}
          </Button>

          <Button
            variant="outlined"
            startIcon={<ModelTrainingIcon />}
            onClick={handleManualRetraining}
            disabled={!uiConfig || trainingLoading || classificationLoading || retrainingLoading}
            color="info"
            sx={{ mr: 1 }}
          >
            {retrainingLoading ? 'Riaddestramento...' : 'Riaddestra Modello'}
          </Button>
          
          <Box display="flex" flexDirection="column" alignItems="center">
            <Button
              variant="contained"
              startIcon={classificationLoading ? undefined : <PlayIcon />}
              onClick={handleStartClassification}
              disabled={!uiConfig || classificationLoading || trainingLoading || retrainingLoading}
              sx={{ mb: 0.5 }}
            >
              {classificationLoading ? 'Classificazione in corso...' : 'Avvia Classificazione Completa'}
            </Button>
            <FormControlLabel
              control={
                <Switch
                  checked={forceReview}
                  onChange={(e) => setForceReview(e.target.checked)}
                  disabled={classificationLoading || trainingLoading || retrainingLoading}
                  size="small"
                />
              }
              label="Force Review"
              sx={{ fontSize: '0.75rem', color: 'text.secondary' }}
            />
          </Box>
        </Box>
      </Box>

      {/* Loading */}
      {(dashboardLoading || loading || classificationLoading || trainingLoading || retrainingLoading) && <LinearProgress sx={{ mb: 2 }} />}

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Success */}
      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
          {successMessage}
        </Alert>
      )}

      {/* Tabs Navigation */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab 
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <AssignmentIcon />
                Review Queue ({cases.length})
              </Box>
            } 
          />
          <Tab 
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <ListIcon />
                Tutte le Sessioni
              </Box>
            } 
          />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {activeTab === 0 && (
        // Review Queue Content (existing content)
        <div>

          {/* Rimossa sezione "Configurazione Coda Revisione" ridondante */}
          {/* I parametri di analisi sono configurabili nel dialogo Training Supervisionato */}
          
          {/* Info Card per guidare l'utente - mostra solo se ci sono casi */}
          {cases.length > 0 && (
            <Card sx={{ mb: 3, backgroundColor: '#e8f5e8' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <CheckCircleIcon color="success" />
                  <Box>
                    <Typography variant="body1" fontWeight="bold">
                      🎯 Casi Pronti per Revisione ({cases.length})
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Il training supervisionato ha identificato questi casi per la revisione umana. 
                      <strong>Prossimo passo:</strong> Clicca sui casi per rivederli, poi usa "Riaddestra Modello" per aggiornare l'AI.
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Info Card per quando non ci sono casi */}
          {cases.length === 0 && !dashboardLoading && (
            <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <SchoolIcon color="secondary" />
                  <Typography variant="body1">
                    <strong>Inizia con il Training Supervisionato:</strong> Clicca "Training Supervisionato" per analizzare le conversazioni e identificare i casi che richiedono revisione umana.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Cases Grid */}
          {cases.length === 0 && !dashboardLoading ? (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <AssignmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Nessun caso da rivedere
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  Tutti i casi sono stati processati correttamente o non ci sono nuovi casi che richiedono supervisione umana.
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<ScienceIcon />}
                  onClick={() => {
                    onCreateMockCases();
                  }}
                  disabled={loading || classificationLoading || !uiConfig}
                >
                  Crea Casi Mock per Test
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Box display="flex" flexWrap="wrap" gap={3}>
              {cases.map((caseItem) => (
                <Box key={caseItem.case_id} flex="1 1 calc(50% - 12px)" minWidth="300px">
                  <Card 
                    sx={{ 
                      height: '100%',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 3
                      }
                    }}
                    onClick={() => onCaseSelect(caseItem)}
                  >
                    <CardContent>
                      {/* Header */}
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Typography variant="h6" component="div">
                          Sessione: {caseItem.session_id.substring(0, 12)}...
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {caseItem.created_at}
                        </Typography>
                      </Box>

                      {/* Reason */}
                      <Box mb={2}>
                        {caseItem.reason.split(';').map((reason, index) => (
                          <Chip
                            key={index}
                            label={reason.trim()}
                            color={getReasonColor(reason)}
                            size="small"
                            sx={{ mr: 0.5, mb: 0.5 }}
                          />
                        ))}
                      </Box>

                      {/* Predictions Comparison - ENHANCED */}
                      <Box 
                        sx={{ 
                          border: '2px solid',
                          borderColor: caseItem.ml_prediction === caseItem.llm_prediction ? 'success.main' : 'warning.main',
                          borderRadius: 2,
                          p: 2,
                          mb: 2,
                          background: caseItem.ml_prediction === caseItem.llm_prediction 
                            ? 'linear-gradient(45deg, #e8f5e8 50%, #e8f5e8 50%)'
                            : 'linear-gradient(45deg, #ffe8e8 50%, #fff3cd 50%)'
                        }}
                      >
                        {/* Agreement/Disagreement Indicator */}
                        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                          <Typography variant="subtitle2" fontWeight="bold" color="text.primary">
                            🤖 PREDIZIONI MODELLI
                          </Typography>
                          <Chip
                            icon={caseItem.ml_prediction === caseItem.llm_prediction ? 
                              <CheckCircleIcon /> : <WarningIcon />}
                            label={caseItem.ml_prediction === caseItem.llm_prediction ? 
                              "ACCORDO" : "DISACCORDO"}
                            color={caseItem.ml_prediction === caseItem.llm_prediction ? 
                              "success" : "warning"}
                            size="small"
                            sx={{ fontWeight: 'bold' }}
                          />
                        </Box>

                        <Box display="flex" gap={2}>
                          <Box 
                            flex={1}
                            sx={{
                              border: '1px solid #1976d2',
                              borderRadius: 1,
                              p: 1.5,
                              backgroundColor: '#e3f2fd'
                            }}
                          >
                            <Typography variant="subtitle2" fontWeight="bold" color="primary">
                              🧠 ML Prediction
                            </Typography>
                            <Typography 
                              variant="h6" 
                              color="primary" 
                              sx={{ fontWeight: 'bold', my: 0.5 }}
                            >
                              {caseItem.ml_prediction || 'N/A'}
                            </Typography>
                            <Chip
                              label={`Confidenza: ${(caseItem.ml_confidence * 100).toFixed(1)}%`}
                              color={getConfidenceColor(caseItem.ml_confidence)}
                              size="small"
                              sx={{ fontWeight: 'bold' }}
                            />
                          </Box>
                          
                          <Box 
                            flex={1}
                            sx={{
                              border: '1px solid #ff9800',
                              borderRadius: 1,
                              p: 1.5,
                              backgroundColor: '#fff3e0'
                            }}
                          >
                            <Typography variant="subtitle2" fontWeight="bold" color="warning.main">
                              🤖 LLM Prediction
                            </Typography>
                            <Typography 
                              variant="h6" 
                              color="warning.main" 
                              sx={{ fontWeight: 'bold', my: 0.5 }}
                            >
                              {caseItem.llm_prediction || 'N/A'}
                            </Typography>
                            <Chip
                              label={`Confidenza: ${(caseItem.llm_confidence * 100).toFixed(1)}%`}
                              color={getConfidenceColor(caseItem.llm_confidence)}
                              size="small"
                              sx={{ fontWeight: 'bold' }}
                            />
                          </Box>
                        </Box>

                        {/* Best Prediction Indicator */}
                        {caseItem.ml_confidence !== caseItem.llm_confidence && (
                          <Box mt={1} textAlign="center">
                            <Typography variant="caption" color="text.secondary">
                              💡 Miglior confidenza: {
                                caseItem.ml_confidence > caseItem.llm_confidence 
                                  ? `ML (${(caseItem.ml_confidence * 100).toFixed(1)}%)`
                                  : `LLM (${(caseItem.llm_confidence * 100).toFixed(1)}%)`
                              }
                            </Typography>
                          </Box>
                        )}
                      </Box>

                      {/* Conversation Preview */}
                      <Box 
                        sx={{ 
                          backgroundColor: 'grey.100',
                          borderRadius: 1,
                          p: 2,
                          mb: 2,
                          maxHeight: 100,
                          overflow: 'hidden'
                        }}
                      >
                        <Typography variant="body2">
                          {caseItem.conversation_text.length > 200
                            ? `${caseItem.conversation_text.substring(0, 200)}...`
                            : caseItem.conversation_text}
                        </Typography>
                      </Box>

                      {/* Metrics */}
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Uncertainty: {caseItem.uncertainty_score.toFixed(3)}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Novelty: {caseItem.novelty_score.toFixed(3)}
                          </Typography>
                        </Box>
                        <Button size="small" variant="outlined">
                          Rivedi Caso
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              ))}
            </Box>
          )}
        </div>
      )}

      {activeTab === 1 && (
        // All Sessions Content
        <AllSessionsView 
          clientName={tenant} 
          onSessionAdd={handleSessionAddedToQueue}
        />
      )}

      {/* Dialogo Training Supervisionato SEMPLIFICATO - 4 PARAMETRI */}
      <Dialog open={trainingDialogOpen} onClose={() => setTrainingDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <SchoolIcon sx={{ mr: 1, color: 'secondary.main' }} />
            Training Supervisionato
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Il sistema estrarrà <strong>TUTTE le discussioni</strong> dal database per il clustering, 
            ma limiterà la revisione umana ai cluster più rappresentativi.
          </Typography>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>🚀 NUOVA LOGICA:</strong><br/>
              • 📊 <strong>Estrazione:</strong> Tutte le discussioni (no limiti)<br/>
              • 🧩 <strong>Clustering:</strong> Su dataset completo<br/>
              • 👤 <strong>Review Umana:</strong> Solo {trainingConfig.max_sessions} sessioni rappresentative<br/>
            </Typography>
          </Alert>
          
          <Box display="flex" flexDirection="column" gap={3}>
            {/* Max Sessions per Review Umana */}
            <TextField
              fullWidth
              type="number"
              label="📊 Max Sessioni per Review Umana"
              value={trainingConfig.max_sessions}
              onChange={(e) => setTrainingConfig({
                ...trainingConfig,
                max_sessions: parseInt(e.target.value) || 500
              })}
              helperText="Numero massimo di sessioni rappresentative da sottoporre all'umano"
              inputProps={{ min: 10, max: 2000 }}
            />
            
            {/* Confidence Threshold */}
            <TextField
              fullWidth
              type="number"
              label="🎯 Soglia Confidenza"
              value={trainingConfig.confidence_threshold}
              onChange={(e) => setTrainingConfig({
                ...trainingConfig,
                confidence_threshold: parseFloat(e.target.value) || 0.7
              })}
              helperText="Soglia di confidenza per auto-classificazione (0.0-1.0)"
              inputProps={{ min: 0, max: 1, step: 0.1 }}
            />
            
            {/* Disagreement Threshold */}
            <TextField
              fullWidth
              type="number" 
              label="⚖️ Soglia Disagreement"
              value={trainingConfig.disagreement_threshold}
              onChange={(e) => setTrainingConfig({
                ...trainingConfig,
                disagreement_threshold: parseFloat(e.target.value) || 0.3
              })}
              helperText="Soglia per ensemble disagreement - priorità review (0.0-1.0)"
              inputProps={{ min: 0, max: 1, step: 0.1 }}
            />
            
            {/* Force Review Switch */}
            <FormControlLabel
              control={
                <Switch
                  checked={trainingConfig.force_review}
                  onChange={(e) => setTrainingConfig({
                    ...trainingConfig,
                    force_review: e.target.checked
                  })}
                />
              }
              label="🔄 Forza Review (rivaluta anche casi già revisionati)"
            />
            
            {/* Quick preset buttons */}
            <Box>
              <Typography variant="body2" fontWeight="bold" mb={1}>
                ⚡ Preset Rapidi:
              </Typography>
              <Box display="flex" gap={1} flexWrap="wrap">
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setTrainingConfig({
                    max_sessions: 200,
                    confidence_threshold: 0.8,
                    force_review: false,
                    disagreement_threshold: 0.3
                  })}
                >
                  🚀 Veloce
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setTrainingConfig({
                    max_sessions: 500,
                    confidence_threshold: 0.7,
                    force_review: false,
                    disagreement_threshold: 0.3
                  })}
                >
                  ⚖️ Standard
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => setTrainingConfig({
                    max_sessions: 1000,
                    confidence_threshold: 0.6,
                    force_review: true,
                    disagreement_threshold: 0.2
                  })}
                >
                  🔍 Approfondito
                </Button>
              </Box>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainingDialogOpen(false)}>
            Annulla
          </Button>
          <Button 
            onClick={handleStartSupervisedTraining}
            variant="contained"
            color="secondary"
            startIcon={<SchoolIcon />}
          >
            Avvia Training
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ReviewDashboard;
