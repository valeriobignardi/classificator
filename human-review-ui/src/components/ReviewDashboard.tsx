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
  ModelTraining as ModelTrainingIcon,
  List as ListIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Group as GroupIcon,
  Link as LinkIcon,
  Star as StarIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';
import ClusterGroupAccordion from './ClusterGroupAccordion';
import { ReviewCase, ClusterCase } from '../types/ReviewCase';
import { Tenant } from '../types/Tenant';
import AllSessionsView from './AllSessionsView';
import { useTenant } from '../contexts/TenantContext';

interface ReviewDashboardProps {
  tenant: Tenant;
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
  // 🆕 Nuovo stato per cluster view
  const [clusters, setClusters] = useState<ClusterCase[]>([]);
  const [extraRepsByCluster, setExtraRepsByCluster] = useState<Record<string, ReviewCase[]>>({});
  // 🆕 NUOVA LOGICA FILTRI: Di base vedi tutto, flag per nascondere categorie specifiche
  const [hideOutliers, setHideOutliers] = useState(true);        // Flag per nascondere outliers (default ON)
  const [hideRepresentatives, setHideRepresentatives] = useState(false); // Flag per nascondere rappresentanti
  
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [uiConfig, setUiConfig] = useState<any>(null);
  const [currentLimit, setCurrentLimit] = useState<number>(2000);  // AUMENTATO LIMITE INIZIALE: ora parte con 2000 per vedere tutte le sessioni

  // Stato per force_review toggle
  const [forceReview, setForceReview] = useState(false);
  
  // 🔍 Context per gestire tenant e prompt status
  const { promptStatus } = useTenant();
  
  // Stato per dialogo training supervisionato SEMPLIFICATO - SOLO Force Review
  const [trainingDialogOpen, setTrainingDialogOpen] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    force_review: false              // Forza revisione casi già revisionati
  });

  // Stato per gestire le tab
  const [activeTab, setActiveTab] = useState(0);
  const CLUSTER_TAB = 0;
  const REVIEW_TAB = 1;
  const SESSIONS_TAB = 2;
  const isClusterTab = activeTab === CLUSTER_TAB;
  const isReviewTab = activeTab === REVIEW_TAB;

  // 🚨 Logica per controllare se i prompt esistono
  const hasPrompts = promptStatus?.canOperate === true;
  const promptsLoading = promptStatus === null; // Ancora caricando

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

  const loadClusterCases = useCallback(async () => {
    const now = Date.now();
    if (now - lastLoadTime.current < 1000) {
      return;
    }
    lastLoadTime.current = now;

    setDashboardLoading(true);
    setError(null);

    try {
      const includeOutliers = !hideOutliers;
      const includePropagated = true;
      const includeRepresentatives = !hideRepresentatives;

      const response = await apiService.getReviewCases(
        tenant.tenant_id,
        currentLimit,
        includePropagated,
        includeOutliers,
        includeRepresentatives
      );

      const grouped: Record<string, ClusterCase & { representatives_list: ReviewCase[] }> = {} as any;
      (response.cases || []).forEach((c: ReviewCase) => {
        const cid = (c.cluster_id || 'unknown').toString();
        if (!grouped[cid]) {
          grouped[cid] = {
            cluster_id: cid,
            representative: undefined as any,
            propagated_sessions: [],
            total_sessions: 0,
            cluster_size: 0,
            representatives_list: []
          } as any;
        }
        if (c.is_representative === false) {
          grouped[cid].propagated_sessions.push(c);
        } else {
          grouped[cid].representatives_list.push(c);
          if (!grouped[cid].representative) grouped[cid].representative = c;
        }
      });

      const clusterArr: ClusterCase[] = Object.values(grouped)
        .filter((cluster) => cluster.cluster_id !== '-1')
        .map((g) => ({
          cluster_id: g.cluster_id,
          representative: g.representative || g.representatives_list[0] || g.propagated_sessions[0],
          propagated_sessions: g.propagated_sessions,
          total_sessions: (g.representatives_list?.length || 0) + (g.propagated_sessions?.length || 0),
          cluster_size: (g.representatives_list?.length || 0) + (g.propagated_sessions?.length || 0)
        }));

      setClusters(clusterArr);
      setCases([]);
    } catch (err) {
      setError('Errore nel caricamento dei cluster');
      console.error('Error loading cluster cases:', err);
    } finally {
      setDashboardLoading(false);
    }
  }, [tenant.tenant_id, currentLimit, hideOutliers, hideRepresentatives]);

  const loadReviewCases = useCallback(async (limit?: number) => {
    const now = Date.now();
    if (now - lastLoadTime.current < 1000) {
      return;
    }
    lastLoadTime.current = now;

    setDashboardLoading(true);
    setError(null);

    try {
      const effectiveLimit = limit || currentLimit;
      const includeOutliers = !hideOutliers;
      const includePropagated = true;
      const includeRepresentatives = !hideRepresentatives;

      const response = await apiService.getReviewCases(
        tenant.tenant_id,
        effectiveLimit,
        includePropagated,
        includeOutliers,
        includeRepresentatives
      );
      setCases(response.cases);
      setClusters([]);
    } catch (err) {
      setError('Errore nel caricamento dei casi');
      console.error('Error loading review cases:', err);
    } finally {
      setDashboardLoading(false);
    }
  }, [tenant.tenant_id, currentLimit, hideOutliers, hideRepresentatives]);

  // 🆕 Gestori per i nuovi flag di esclusione
  const handleHideOutliersToggle = () => {
    setHideOutliers(!hideOutliers);
  };

  const handleHideRepresentativesToggle = () => {
    setHideRepresentatives(!hideRepresentatives);
  };

  const handleRefreshCases = useCallback(() => {
    if (isClusterTab) {
      loadClusterCases();
    } else if (isReviewTab) {
      loadReviewCases();
    }
  }, [isClusterTab, isReviewTab, loadClusterCases, loadReviewCases]);

  // Prefetch di altri rappresentanti per cluster dalla vista "Tutte le sessioni"
  useEffect(() => {
    const prefetchExtraRepresentatives = async () => {
      if (!isClusterTab) return;
      try {
        const allSess = await apiService.getAllSessions(tenant.tenant_id, true);
        const map: Record<string, ReviewCase[]> = {};
        (allSess.sessions || []).forEach((s: any) => {
          const cid = (s.cluster_id !== undefined && s.cluster_id !== null) ? s.cluster_id.toString() : undefined;
          if (!cid) return;
          if (s.is_representative !== true) return;
          const rc: ReviewCase = {
            case_id: s.session_id,
            session_id: s.session_id,
            conversation_text: s.conversation_text,
            classification: (s.classifications && s.classifications[0]?.tag_name) || 'N/A',
            ml_prediction: '',
            ml_confidence: 0,
            llm_prediction: '',
            llm_confidence: 0,
            uncertainty_score: 0,
            novelty_score: 0,
            reason: '',
            created_at: s.created_at,
            tenant: tenant.tenant_slug || tenant.tenant_name || tenant.tenant_id,
            cluster_id: cid,
            is_representative: true
          };
          if (!map[cid]) map[cid] = [];
          map[cid].push(rc);
        });
        setExtraRepsByCluster(map);
      } catch (e) {
        console.error('Errore prefetch rappresentanti', e);
      }
    };
    prefetchExtraRepresentatives();
  }, [isClusterTab, tenant]);

  const handleStartSupervisedTraining = async () => {
    setTrainingLoading(true);
    setError(null);
    setSuccessMessage(null);
    setTrainingDialogOpen(false);

    try {
      // 🔧 NUOVO: Solo force_review, tutti gli altri parametri vengono dal database TAG.soglie
      const apiConfig = {
        force_review: trainingConfig.force_review
      };
      
      console.log('🔍 [DEBUG] Training config (parametri centralizzati):', {
        'force_review': apiConfig.force_review,
        'note': 'Soglie e parametri clustering caricati dal database TAG.soglie'
      });
      
      const response = await apiService.startSupervisedTraining(tenant.tenant_id, apiConfig);
      
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
        handleRefreshCases();
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
      const response = await apiService.triggerManualRetraining(tenant.tenant_id);
      
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
      const tenantIdentifier = tenant.tenant_slug || tenant.tenant_id || tenant.tenant_name;
      const response = await apiService.startFullClassification(tenantIdentifier, {
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
        handleRefreshCases();
      }, 1000);

    } catch (err: any) {
      setError(`Errore durante la classificazione: ${err.message}`);
      console.error('Classification error:', err);
    } finally {
      setClassificationLoading(false);
    }
  };

  useEffect(() => {
    if (!uiConfig) return;
    if (isClusterTab) {
      loadClusterCases();
    } else if (isReviewTab) {
      loadReviewCases();
    }
  }, [uiConfig, isClusterTab, isReviewTab, loadClusterCases, loadReviewCases]);

  // Ricarica casi quando necessario
  useEffect(() => {
    if (refreshTrigger > 0 && uiConfig) {
      handleRefreshCases();
    }
  }, [refreshTrigger, handleRefreshCases, uiConfig]);

  // Callback quando viene aggiunta una sessione alla queue dal componente AllSessionsView
  const handleSessionAddedToQueue = useCallback(() => {
    setSuccessMessage(`Sessione aggiunta alla review queue`);
    // Ricarica i casi per aggiornare la lista
    handleRefreshCases();
  }, [handleRefreshCases]);

  const getReasonColor = (reason: string) => {
    if (!reason) return 'default';
    const lowerReason = reason.toLowerCase();
    if (lowerReason.includes('disagreement')) return 'warning';
    if (lowerReason.includes('confidence')) return 'info';
    if (lowerReason.includes('uncertainty')) return 'secondary';
    return 'default';
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

          <Tooltip 
            title={!hasPrompts ? "Devi configurare i prompt nella sezione Configurazione prima di poter usare il training supervisionato" : ""}
            arrow
          >
            <span>
              <Button
                variant="outlined"
                startIcon={trainingLoading ? undefined : <SchoolIcon />}
                onClick={() => setTrainingDialogOpen(true)}
                disabled={!uiConfig || trainingLoading || classificationLoading || retrainingLoading || !hasPrompts}
                color="secondary"
                sx={{ mr: 1 }}
              >
                {trainingLoading ? 'Training in corso...' : 'Training Supervisionato'}
              </Button>
            </span>
          </Tooltip>

          <Tooltip 
            title={!hasPrompts ? "Devi configurare i prompt nella sezione Configurazione prima di poter riaddestrare il modello" : ""}
            arrow
          >
            <span>
              <Button
                variant="outlined"
                startIcon={<ModelTrainingIcon />}
                onClick={handleManualRetraining}
                disabled={!uiConfig || trainingLoading || classificationLoading || retrainingLoading || !hasPrompts}
                color="info"
                sx={{ mr: 1 }}
              >
                {retrainingLoading ? 'Riaddestramento...' : 'Riaddestra Modello'}
              </Button>
            </span>
          </Tooltip>
          
          <Box display="flex" flexDirection="column" alignItems="center">
            <Tooltip 
              title={!hasPrompts ? "Devi configurare i prompt nella sezione Configurazione prima di avviare la classificazione" : ""}
              arrow
            >
              <span>
                <Button
                  variant="contained"
                  startIcon={classificationLoading ? undefined : <PlayIcon />}
                  onClick={handleStartClassification}
                  disabled={!uiConfig || classificationLoading || trainingLoading || retrainingLoading || !hasPrompts}
                  sx={{ mb: 0.5 }}
                >
                  {classificationLoading ? 'Classificazione in corso...' : 'Avvia Classificazione Completa'}
                </Button>
              </span>
            </Tooltip>
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
                <GroupIcon />
                {`Cluster Focus (${clusters.length})`}
              </Box>
            }
          />
          <Tab
            label={
              <Box display="flex" alignItems="center" gap={1}>
                <AssignmentIcon />
                {`Review Queue (${cases.length})`}
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
      {activeTab === CLUSTER_TAB && (
        // Review Queue Content (existing content)
        <div>

          {/* 🆕 Controlli Vista Cluster e Filtri Review Queue */}
          <Card sx={{ mb: 3, backgroundColor: '#f0f8ff', borderLeft: '4px solid #1976d2' }}>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <GroupIcon color="primary" />
                <Typography variant="h6" color="primary">
                  👑 Vista Cluster Focus
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Analizza rappresentanti e sessioni propagate dello stesso cluster per verificare la coerenza delle etichette.
              </Typography>

              <Box sx={{ mt: 2, p: 2, bgcolor: 'rgba(25, 118, 210, 0.05)', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom color="primary">
                  🎛️ Filtri cluster
                </Typography>
                <Box display="flex" alignItems="center" gap={3} flexWrap="wrap">
                  <FormControlLabel
                    control={
                      <Switch
                        checked={hideOutliers}
                        onChange={handleHideOutliersToggle}
                        disabled={dashboardLoading}
                        size="small"
                        color="warning"
                      />
                    }
                    label={<Typography variant="body2">🚫 Escludi outlier (-1)</Typography>}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={hideRepresentatives}
                        onChange={handleHideRepresentativesToggle}
                        disabled={dashboardLoading}
                        size="small"
                        color="success"
                      />
                    }
                    label={<Typography variant="body2">🙈 Nascondi rappresentanti</Typography>}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* Rimossa sezione "Configurazione Coda Revisione" ridondante */}
          {/* I parametri di analisi sono configurabili nel dialogo Training Supervisionado */}
          
          {/* Info Card per guidare l'utente - cluster */}
          {clusters.length > 0 && (
            <Card sx={{ mb: 3, backgroundColor: '#e8f5e8' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <CheckCircleIcon color="success" />
                  <Box>
                    <Typography variant="body1" fontWeight="bold">
                      🎯 Cluster pronti per revisione ({clusters.length})
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Ogni cluster include il rappresentante principale e le sessioni propagate. Confronta le decisioni per assicurarti che la propagazione sia corretta e usa "Conferma maggioranza" quando le etichette coincidono.
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Alert per prompt mancanti */}
          {!hasPrompts && !promptsLoading && (
            <Card sx={{ mb: 3, backgroundColor: '#fff3e0', border: '1px solid #ff9800' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <WarningIcon color="warning" />
                  <Box>
                    <Typography variant="h6" color="warning.dark" gutterBottom>
                      ⚠️ Configurazione Incompleta
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      <strong>Prima di poter utilizzare il sistema di training e classificazione, devi configurare i prompt.</strong>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      📋 <strong>VAI NELLA SEZIONE CONFIGURAZIONE E CREA IL PRIMO PROMPT</strong>
                    </Typography>
                    {promptStatus && promptStatus.missingCount > 0 && (
                      <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                        Mancano {promptStatus.missingCount} prompt richiesti per questo tenant.
                      </Typography>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Info Card per quando non ci sono cluster (ma ci sono i prompt) */}
          {hasPrompts && clusters.length === 0 && !dashboardLoading && (
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

          {/* Cluster Focus */}
          {clusters.length === 0 && !dashboardLoading ? (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <GroupIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Nessun cluster da rivedere
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  Non ci sono cluster organizzati per la revisione. Esegui il training supervisionato o la classificazione completa per popolare la coda.
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<ScienceIcon />}
                  onClick={onCreateMockCases}
                  disabled={loading || classificationLoading || !uiConfig}
                >
                  Crea Casi Mock per Test
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Box display="flex" flexDirection="column" gap={2}>
              {clusters.map((cluster) => (
                <ClusterGroupAccordion
                  key={cluster.cluster_id}
                  clusterId={cluster.cluster_id}
                  representatives={
                    extraRepsByCluster[cluster.cluster_id] && extraRepsByCluster[cluster.cluster_id].length > 0
                      ? extraRepsByCluster[cluster.cluster_id]
                      : (cluster.representative ? [cluster.representative] : [])
                  }
                  propagated={cluster.propagated_sessions || []}
                  onConfirmMajority={async (cid, chosenLabel) => {
                    try {
                      const trimmedLabel = (chosenLabel || '').trim();
                      if (!trimmedLabel) {
                        setError('Seleziona o inserisci un\'etichetta valida prima di confermare.');
                        return;
                      }
                      const res = await apiService.resolveClusterMajority(tenant.tenant_id, cid, {
                        selected_label: trimmedLabel
                      });
                      const appliedLabel = res.applied_label || trimmedLabel.toUpperCase();
                      setSuccessMessage(`✅ Cluster ${cid}: applicata etichetta '${appliedLabel}'. Risolti ${res.resolved_count}/${res.total_candidates}.`);
                      setTimeout(() => loadClusterCases(), 500);
                    } catch (e: any) {
                      setError(e?.message || 'Errore Conferma maggioranza');
                    }
                  }}
                />
              ))}
            </Box>
          )}
        </div>
      )}

      {activeTab === REVIEW_TAB && (
        <>
          <Card sx={{ mb: 3, backgroundColor: '#f4f7ff', borderLeft: '4px solid #5e35b1' }}>
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <AssignmentIcon color="primary" />
                <Typography variant="h6" color="primary">
                  📋 Review Queue - Lista Cronologica
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Esamina tutti i casi in coda con il dettaglio completo della classificazione automatica.
              </Typography>

              <Box sx={{ mt: 2, p: 2, bgcolor: 'rgba(94, 53, 177, 0.08)', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom color="primary">
                  🎛️ Filtri lista
                </Typography>
                <Box display="flex" alignItems="center" gap={3} flexWrap="wrap">
                  <FormControlLabel
                    control={
                      <Switch
                        checked={hideOutliers}
                        onChange={handleHideOutliersToggle}
                        disabled={dashboardLoading}
                        size="small"
                        color="warning"
                      />
                    }
                    label={<Typography variant="body2">🚫 Escludi outlier (-1)</Typography>}
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={hideRepresentatives}
                        onChange={handleHideRepresentativesToggle}
                        disabled={dashboardLoading}
                        size="small"
                        color="success"
                      />
                    }
                    label={<Typography variant="body2">🙈 Nascondi rappresentanti</Typography>}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>

          {!hasPrompts && !promptsLoading && (
            <Card sx={{ mb: 3, backgroundColor: '#fff3e0', border: '1px solid #ff9800' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <WarningIcon color="warning" />
                  <Box>
                    <Typography variant="h6" color="warning.dark" gutterBottom>
                      ⚠️ Configurazione Incompleta
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      <strong>Prima di poter utilizzare il sistema di training e classificazione, devi configurare i prompt.</strong>
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      📋 <strong>VAI NELLA SEZIONE CONFIGURAZIONE E CREA IL PRIMO PROMPT</strong>
                    </Typography>
                    {promptStatus && promptStatus.missingCount > 0 && (
                      <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                        Mancano {promptStatus.missingCount} prompt richiesti per questo tenant.
                      </Typography>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}

          {hasPrompts && cases.length === 0 && !dashboardLoading && (
            <Card sx={{ mb: 3, backgroundColor: '#f5f5f5' }}>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <SchoolIcon color="secondary" />
                  <Typography variant="body1">
                    <strong>La coda è vuota:</strong> esegui il training supervisionato o avvia una classificazione per generare nuovi casi.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}

          {cases.length === 0 && !dashboardLoading ? (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 6 }}>
                <AssignmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Nessun caso da rivedere
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  Tutti i casi sono stati processati correttamente o non ci sono nuove discussioni con bassa confidenza.
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<ScienceIcon />}
                  onClick={onCreateMockCases}
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
                      border: caseItem.is_representative === false
                        ? '2px dashed #ff9800'
                        : '2px solid #1976d2',
                      backgroundColor: caseItem.is_representative === false
                        ? '#fff8e1'
                        : 'white',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 3
                      }
                    }}
                    onClick={() => onCaseSelect(caseItem)}
                  >
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Box display="flex" alignItems="center" gap={1}>
                          {caseItem.is_representative === false ? <LinkIcon color="secondary" /> : <StarIcon color="primary" />}
                          <Box>
                            <Typography variant="h6" component="div">
                              Sessione: {(caseItem.session_id || '').substring(0, 12)}...
                            </Typography>
                            {(caseItem.cluster_id !== undefined && caseItem.cluster_id !== null) && (
                              <Typography variant="body2" color="primary" fontWeight="bold">
                                📊 CLUSTER: {caseItem.cluster_id}
                              </Typography>
                            )}
                          </Box>
                        </Box>
                        <Box display="flex" flexDirection="column" alignItems="end">
                          <Typography variant="body2" color="text.secondary">
                            {caseItem.created_at}
                          </Typography>
                          <Chip
                            icon={caseItem.is_representative === false ? <LinkIcon /> : <StarIcon />}
                            label={caseItem.is_representative === false ? "🔗 Ereditata" : "👑 Rappresentante"}
                            color={caseItem.is_representative === false ? "secondary" : "primary"}
                            size="small"
                            variant={caseItem.is_representative === false ? "outlined" : "filled"}
                          />
                        </Box>
                      </Box>

                      <Box mb={2}>
                        {(caseItem.reason || '').split(';').map((reason: string, index: number) => (
                          <Chip
                            key={index}
                            label={reason.trim()}
                            color={getReasonColor(reason)}
                            size="small"
                            sx={{ mr: 0.5, mb: 0.5 }}
                          />
                        ))}
                      </Box>

                      <Box
                        sx={{
                          border: '1px solid',
                          borderColor: 'info.main',
                          borderRadius: 1,
                          p: 1,
                          mb: 2,
                          backgroundColor: 'info.light',
                          color: 'info.contrastText'
                        }}
                      >
                        <Typography variant="subtitle2" fontWeight="bold">
                          🏷️  Tipo: <Chip
                            label={caseItem.classification_type || 'NORMALE'}
                            color={
                              caseItem.classification_type === 'RAPPRESENTANTE' ? 'primary' :
                              caseItem.classification_type === 'PROPAGATO' ? 'success' :
                              caseItem.classification_type === 'OUTLIER' ? 'warning' : 'default'
                            }
                            size="small"
                            sx={{ ml: 1, fontWeight: 'bold' }}
                          />
                        </Typography>
                      </Box>

                      <Box
                        sx={{
                          border: '3px solid #2e7d32',
                          borderRadius: 2,
                          p: 2,
                          mb: 2,
                          backgroundColor: '#e8f5e8',
                          textAlign: 'center'
                        }}
                      >
                        <Typography variant="subtitle1" fontWeight="bold" color="success.dark" gutterBottom>
                          🏷️ CLASSIFICAZIONE FINALE
                        </Typography>
                        <Typography
                          variant="h4"
                          color="success.dark"
                          sx={{ fontWeight: 'bold', mb: 1 }}
                        >
                          {caseItem.classification || 'N/A'}
                        </Typography>
                      </Box>

                      {(caseItem.ml_prediction || caseItem.llm_prediction) && (
                        <Box
                          sx={{
                            border: '1px solid #ccc',
                            borderRadius: 2,
                            p: 1.5,
                            mb: 2,
                            backgroundColor: '#f9f9f9'
                          }}
                        >
                          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                            <Typography variant="body2" fontWeight="bold" color="text.secondary">
                              🤖 Dettagli Predizioni Modelli
                            </Typography>
                            <Chip
                              icon={caseItem.ml_prediction === caseItem.llm_prediction ?
                                <CheckCircleIcon /> : <WarningIcon />}
                              label={caseItem.ml_prediction === caseItem.llm_prediction ?
                                "ACCORDO" : "DISACCORDO"}
                              color={caseItem.ml_prediction === caseItem.llm_prediction ?
                                "success" : "warning"}
                              size="small"
                            />
                          </Box>

                          <Box display="flex" gap={1}>
                            {caseItem.ml_prediction && (
                              <Box flex={1} p={1} sx={{ border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                <Typography variant="caption" color="text.secondary">
                                  🤖 ML Model
                                </Typography>
                                <Typography variant="subtitle1" fontWeight="bold">
                                  {caseItem.ml_prediction}
                                </Typography>
                                <Chip
                                  label={`Conf ${(caseItem.ml_confidence * 100).toFixed(1)}%`}
                                  color={
                                    caseItem.ml_confidence > 0.8 ? 'success' :
                                    caseItem.ml_confidence > 0.6 ? 'warning' : 'error'
                                  }
                                  size="small"
                                />
                              </Box>
                            )}

                            {caseItem.llm_prediction && (
                              <Box
                                flex={1}
                                sx={{
                                  border: '1px solid #ff9800',
                                  borderRadius: 1,
                                  p: 1,
                                  backgroundColor: '#fff3e0'
                                }}
                              >
                                <Typography variant="caption" fontWeight="bold" color="warning.main">
                                  🤖 LLM
                                </Typography>
                                <Typography
                                  variant="body2"
                                  color="warning.main"
                                  sx={{ fontWeight: 'bold' }}
                                >
                                  {caseItem.llm_prediction}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {(caseItem.llm_confidence * 100).toFixed(1)}%
                                </Typography>
                              </Box>
                            )}

                            {!caseItem.ml_prediction && !caseItem.llm_prediction && (
                              <Box
                                sx={{
                                  border: '1px solid #9e9e9e',
                                  borderRadius: 1,
                                  p: 1,
                                  backgroundColor: '#f5f5f5',
                                  width: '100%',
                                  textAlign: 'center'
                                }}
                              >
                                <Typography variant="caption" color="text.secondary">
                                  🤖 Classificazione: {caseItem.classification_method || 'LLM'}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        </Box>
                      )}

                      {!caseItem.ml_prediction && !caseItem.llm_prediction && (
                        <Box
                          sx={{
                            border: '1px solid #e0e0e0',
                            borderRadius: 2,
                            p: 1,
                            mb: 2,
                            backgroundColor: '#fafafa',
                            textAlign: 'center'
                          }}
                        >
                          <Typography variant="body2" color="text.secondary">
                            📋 Metodo: {caseItem.classification_method || 'Non specificato'}
                          </Typography>
                        </Box>
                      )}

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
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => onCaseSelect(caseItem)}
                        >
                          {caseItem.is_representative === false ? "🔗 Rivedi Ereditata" : "👑 Rivedi Rappresentante"}
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              ))}
            </Box>
          )}
        </>
      )}

      {activeTab === SESSIONS_TAB && (
        // All Sessions Content
        <AllSessionsView 
          tenantIdentifier={tenant.tenant_slug || tenant.tenant_id || tenant.tenant_name}
          tenantDisplayName={tenant.tenant_name}
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
            Il sistema utilizzerà i parametri configurati in <strong>PARAMETRI CLUSTERING</strong> per
            il processo di training supervisionato.
          </Typography>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>🚀 CONFIGURAZIONE CENTRALIZZATA:</strong><br/>
              • 📊 <strong>Soglie Review:</strong> Configurate in "Parametri Clustering"<br/>
              • 🧩 <strong>Parametri HDBSCAN/UMAP:</strong> Dal database locale<br/>
              • 👤 <strong>Force Review:</strong> Rivaluta casi già revisionati<br/>
            </Typography>
          </Alert>
          
          <Box display="flex" flexDirection="column" gap={3}>
            {/* Solo Force Review Switch */}
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
            
            <Typography variant="body2" color="text.secondary">
              <strong>📋 Nota:</strong> Per modificare soglie di confidenza, disagreement e parametri di clustering,
              utilizzare la sezione <strong>"Parametri Clustering"</strong> nel menu principale.
            </Typography>
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
