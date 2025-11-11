import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  TextField,
  Slider,
  Chip,
  Alert,
  IconButton,
  Grid
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  ThumbUp as ThumbUpIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';
import { ReviewCase, ClusterContextCase, ClusterContextSummary } from '../types/ReviewCase';
import { Tenant } from '../types/Tenant';
import TagSuggestions from './TagSuggestions';
import ClusterCaseStrip from './ClusterCaseStrip';
import ClusterContextPanel from './ClusterContextPanel';
import { formatLabelForDisplay, normalizeLabel } from '../utils/labelUtils';

interface CaseDetailProps {
  case: ReviewCase;
  tenant: Tenant;
  onCaseResolved: () => void;
  onBack: () => void;
  onNavigateCase: (caseItem: ReviewCase) => void;
}

const CaseDetail: React.FC<CaseDetailProps> = ({
  case: caseItem,
  tenant,
  onCaseResolved,
  onBack,
  onNavigateCase
}) => {
  const [humanDecision, setHumanDecision] = useState('');
  const [confidence, setConfidence] = useState(0.8);
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  
  // Stati per tag suggestions
  const [availableTags, setAvailableTags] = useState<Array<{tag: string, count: number, source: string, avg_confidence: number}>>([]);
  const [tagsLoading, setTagsLoading] = useState(true);

  // Cluster context state
  const [clusterSessions, setClusterSessions] = useState<ClusterContextCase[]>([]);
  const [clusterSummary, setClusterSummary] = useState<ClusterContextSummary | null>(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [clusterError, setClusterError] = useState<string | null>(null);
  const [selectedClusterSessionId, setSelectedClusterSessionId] = useState<string>(caseItem.session_id);
  const [navigatingSessionId, setNavigatingSessionId] = useState<string | null>(null);

  // Carica tag disponibili al mount del componente
  useEffect(() => {
    const loadAvailableTags = async () => {
      setTagsLoading(true);
      try {
        console.log('🔍 [CaseDetail] Caricamento tag per tenant:', tenant);
        console.log('🔍 [CaseDetail] Tenant ID utilizzato:', tenant.tenant_id);
        
        const response = await apiService.getAvailableTags(tenant.tenant_id);
        console.log('✅ [CaseDetail] Tag ricevuti:', response.tags.length);
        setAvailableTags(response.tags);
      } catch (err) {
        console.error('❌ [CaseDetail] Error loading available tags:', err);
        // Non mostrare errore per i tag, continua senza di essi
      } finally {
        setTagsLoading(false);
      }
    };

    loadAvailableTags();
  }, [tenant]);

  const clusterId = useMemo(() => {
    if (caseItem.cluster_id === undefined || caseItem.cluster_id === null) {
      return null;
    }
    return caseItem.cluster_id.toString();
  }, [caseItem.cluster_id]);

  useEffect(() => {
    setSelectedClusterSessionId(caseItem.session_id);
  }, [caseItem.session_id]);

  useEffect(() => {
    if (!clusterId || clusterId === '-1') {
      setClusterSessions([]);
      setClusterSummary(null);
      setClusterError(null);
      return;
    }

    let isMounted = true;

    const loadClusterContext = async () => {
      setClusterLoading(true);
      setClusterError(null);

      try {
        const [sessionsResp, reviewCasesResp] = await Promise.all([
          apiService.getAllSessions(tenant.tenant_id, true),
          apiService.getReviewCases(tenant.tenant_id, 500, true, false, true)
        ]);

        const sessionList = sessionsResp.sessions || [];
        const reviewCases = (reviewCasesResp.cases || []) as ReviewCase[];

        const reviewCaseBySession = new Map<string, ReviewCase>();
        reviewCases.forEach((rc) => {
          if (rc.session_id) {
            reviewCaseBySession.set(rc.session_id, rc);
          }
        });

        const normalized: ClusterContextCase[] = [];

        sessionList.forEach((session: any) => {
          const sessionClusterId =
            session.cluster_id !== undefined && session.cluster_id !== null
              ? session.cluster_id.toString()
              : session.classifications?.find(
                  (cls: any) => cls.cluster_id !== undefined && cls.cluster_id !== null
                )?.cluster_id?.toString();

          if (sessionClusterId !== clusterId) {
            return;
          }

          if (sessionClusterId === '-1') {
            return;
          }

        const reviewCase = reviewCaseBySession.get(session.session_id);
        const classificationEntry = session.classifications?.[0];

        const rawLabelInput =
          (reviewCase?.classification ||
            classificationEntry?.tag_name ||
            session.tag_name ||
            '')?.toString() || '';
        const normalizedLabel = normalizeLabel(rawLabelInput);
        const displayLabel = formatLabelForDisplay(normalizedLabel);
        const confidence = classificationEntry?.confidence;
        const method = classificationEntry?.method;
        const isRepresentative =
          reviewCase?.is_representative === true ||
          session.is_representative === true ||
          reviewCase?.classification_type === 'RAPPRESENTANTE';
        const classificationType =
          reviewCase?.classification_type ||
          (isRepresentative
            ? 'RAPPRESENTANTE'
            : session.is_representative === false
            ? 'PROPAGATO'
            : undefined);

        const conversationText =
          reviewCase?.conversation_text ||
          session.conversation_text ||
          session.full_conversation ||
          (Array.isArray(session.messages)
            ? session.messages
                .map((message: any) => {
                  const speaker = message.role || message.speaker;
                  const content = message.content || message.text || '';
                  if (!content) {
                    return '';
                  }
                  return `${speaker ? `[${speaker.toString().toUpperCase()}] ` : ''}${content}`;
                })
                .filter(Boolean)
                .join('\n')
            : '');

        normalized.push({
          case_id: reviewCase?.case_id,
          session_id: session.session_id,
          label: normalizedLabel,
          raw_label: rawLabelInput || normalizedLabel,
          display_label: displayLabel,
          status: session.status,
          is_representative: isRepresentative,
          classification_type: classificationType,
          propagated_from: reviewCase?.propagated_from,
          confidence,
          method,
          human_decision: undefined,
          created_at: session.created_at,
          conversation_text: conversationText
        });
      });

      if (!normalized.some((item) => item.session_id === caseItem.session_id)) {
        const fallbackLabelNormalized = normalizeLabel(caseItem.classification);
        normalized.push({
          case_id: caseItem.case_id,
          session_id: caseItem.session_id,
          label: fallbackLabelNormalized,
          raw_label: caseItem.classification,
          display_label: formatLabelForDisplay(fallbackLabelNormalized),
          status: 'in_review_queue',
          is_representative: caseItem.is_representative,
          classification_type: caseItem.classification_type,
          propagated_from: caseItem.propagated_from,
          confidence: caseItem.ml_confidence,
          method: caseItem.classification_method,
          conversation_text: caseItem.conversation_text
        });
      }

      const counts = normalized.reduce<Record<string, number>>((acc, item) => {
        const key = item.label || 'N/A';
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {});

      const majorityLabelRaw = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
      const majorityLabelDisplay = majorityLabelRaw
        ? formatLabelForDisplay(majorityLabelRaw)
        : undefined;
      const reviewedCount = normalized.filter((item) => item.status === 'reviewed').length;

      const summary: ClusterContextSummary = {
        cluster_id: clusterId,
        majority_label: majorityLabelDisplay || majorityLabelRaw,
        majority_label_raw: majorityLabelRaw,
        majority_label_display: majorityLabelDisplay,
        total_cases: normalized.length,
        reviewed_cases: reviewedCount,
        pending_cases: normalized.length - reviewedCount,
        representatives: normalized.filter((item) => item.is_representative).length,
        propagated: normalized.filter((item) => !item.is_representative).length
        };

        const sorted = normalized.sort((a, b) => {
          if (a.is_representative && !b.is_representative) return -1;
          if (!a.is_representative && b.is_representative) return 1;
          if (a.status === 'in_review_queue' && b.status !== 'in_review_queue') return -1;
          if (a.status !== 'in_review_queue' && b.status === 'in_review_queue') return 1;
          return (b.confidence || 0) - (a.confidence || 0);
        });

        if (isMounted) {
          setClusterSessions(sorted);
          setClusterSummary(summary);
        }
      } catch (err) {
        console.error('Errore caricamento contesto cluster:', err);
        if (isMounted) {
          setClusterError('Impossibile caricare il contesto del cluster');
        }
      } finally {
        if (isMounted) {
          setClusterLoading(false);
        }
      }
    };

    loadClusterContext();

    return () => {
      isMounted = false;
    };
  }, [
    tenant.tenant_id,
    clusterId,
    caseItem.case_id,
    caseItem.session_id,
    caseItem.classification,
    caseItem.is_representative,
    caseItem.classification_type,
    caseItem.ml_confidence,
    caseItem.propagated_from
  ]);

  const handleClusterCaseSelect = useCallback(
    async (sessionId: string) => {
      setSelectedClusterSessionId(sessionId);

      if (sessionId === caseItem.session_id) {
        return;
      }

      const target = clusterSessions.find((item) => item.session_id === sessionId);
      if (!target) {
        return;
      }

      if (!target.case_id) {
        return;
      }

      try {
        setClusterError(null);
        setNavigatingSessionId(sessionId);
        const response = await apiService.getCaseDetail(tenant.tenant_id, target.case_id);
        onNavigateCase(response.case);
      } catch (err) {
        console.error('Errore apertura caso cluster:', err);
        setClusterError('Impossibile aprire il caso selezionato. Verifica che sia ancora in coda.');
      } finally {
        setNavigatingSessionId(null);
      }
    },
    [clusterSessions, caseItem.session_id, tenant.tenant_id, onNavigateCase]
  );

  const handleQuickDecision = (decision: string) => {
    setHumanDecision(decision);
  };

  const handleTagSelect = (tag: string) => {
    setHumanDecision(tag);
  };

  const handleSubmit = async () => {
    if (!humanDecision.trim()) {
      setError('Per favore inserisci una classificazione corretta.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await apiService.resolveCase(
        tenant.tenant_id,
        caseItem.case_id,
        humanDecision.trim(),
        confidence,
        notes
      );
      setSuccess(true);
      setTimeout(() => {
        onCaseResolved();
      }, 1500);
    } catch (err) {
      setError('Errore nella risoluzione del caso');
      console.error('Error resolving case:', err);
    } finally {
      setLoading(false);
    }
  };

  const getReasonColor = (reason: string) => {
    if (reason.toLowerCase().includes('disagreement')) return 'warning';
    if (reason.toLowerCase().includes('confidence')) return 'info';
    if (reason.toLowerCase().includes('uncertainty')) return 'secondary';
    return 'default';
  };

  // 🔧 Gestione speciale per casi propagati da clustering
  const isClusterPropagated = caseItem.ml_prediction === "unknown" || caseItem.ml_prediction === "N/A";
  const isAgreement = !isClusterPropagated && (caseItem.ml_prediction === caseItem.llm_prediction);

  if (success) {
    return (
      <Card>
        <CardContent sx={{ textAlign: 'center', py: 6 }}>
          <CheckCircleIcon sx={{ fontSize: 64, color: 'success.main', mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Caso Risolto con Successo!
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Decisione: <strong>{humanDecision}</strong>
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Confidenza: {confidence}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box display="flex" alignItems="center" mb={3}>
        <IconButton onClick={onBack} sx={{ mr: 2 }}>
          <ArrowBackIcon />
        </IconButton>
        <Typography variant="h4">
          Revisione Caso: {caseItem.case_id.substring(0, 8)}...
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box
        display="flex"
        gap={3}
        sx={{
          overflow: 'hidden',
          flexWrap: { xs: 'wrap', lg: 'nowrap' },
          alignItems: 'flex-start'
        }}
      >
        {/* Case Information + Cluster Strip */}
        <Box flex={{ xs: '1 1 100%', lg: 3 }} sx={{ minWidth: 0, overflow: 'hidden' }}>
          <ClusterCaseStrip
            cases={clusterSessions}
            selectedSessionId={selectedClusterSessionId}
            openedSessionId={caseItem.session_id}
            onSelectCase={handleClusterCaseSelect}
            summary={clusterSummary}
          />

          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Informazioni Caso
              </Typography>
              
              <Box display="flex" gap={3}>
                <Box flex={1} sx={{ minWidth: 0 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Case ID:</strong> {caseItem.case_id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Session ID:</strong> {caseItem.session_id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Creato il:</strong> {caseItem.created_at}
                  </Typography>
                </Box>
                <Box flex={1} sx={{ minWidth: 0 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Tenant:</strong> {caseItem.tenant}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Tenant:</strong> {caseItem.tenant}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Uncertainty:</strong> {caseItem.uncertainty_score.toFixed(3)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-word' }}>
                    <strong>Novelty:</strong> {caseItem.novelty_score.toFixed(3)}
                  </Typography>
                </Box>
              </Box>

              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Motivo della Revisione:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {caseItem.reason.split(';').map((reason, index) => (
                    <Chip
                      key={index}
                      label={reason.trim()}
                      color={getReasonColor(reason)}
                      size="small"
                    />
                  ))}
                </Box>
              </Box>
            </CardContent>
          </Card>

          {/* ML/LLM Predictions */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🤖 Predizioni Automatiche
              </Typography>
              
              <Box display="flex" gap={3}>
                <Box flex={1}>
                  <Typography variant="subtitle2" fontWeight="bold" color="primary.main">
                    🎯 Classificazione Finale:
                  </Typography>
                  <Typography variant="h6" sx={{ mt: 0.5, mb: 1 }}>
                    {caseItem.classification || caseItem.ml_prediction || 'N/A'}
                  </Typography>
                  <Chip
                    label={isClusterPropagated 
                      ? 'Propagato da cluster' 
                      : `Confidenza: ${(caseItem.ml_confidence * 100).toFixed(1)}%`
                    }
                    color={isClusterPropagated 
                      ? 'info' 
                      : (caseItem.ml_confidence > 0.8 ? 'success' : caseItem.ml_confidence > 0.6 ? 'warning' : 'error')
                    }
                    size="small"
                  />
                </Box>
                <Box flex={1}>
                  <Typography variant="subtitle2" fontWeight="bold" color="warning.main">
                    🤖 LLM Prediction:
                  </Typography>
                  <Typography variant="h6" sx={{ mt: 0.5, mb: 1 }}>
                    {caseItem.llm_prediction || 'N/A'}
                  </Typography>
                  <Chip
                    label={`Confidenza: ${(caseItem.llm_confidence * 100).toFixed(1)}%`}
                    color={caseItem.llm_confidence > 0.8 ? 'success' : caseItem.llm_confidence > 0.6 ? 'warning' : 'error'}
                    size="small"
                  />
                </Box>
              </Box>

              {/* Agreement/Disagreement Indicator */}
              <Box mt={2} p={2} sx={{ 
                backgroundColor: isClusterPropagated ? 'info.light' : (isAgreement ? 'success.light' : 'warning.light'),
                borderRadius: 1,
                border: `2px solid ${isClusterPropagated ? '#2196f3' : (isAgreement ? '#4caf50' : '#ff9800')}`
              }}>
                <Typography variant="subtitle2" fontWeight="bold">
                  {isClusterPropagated 
                    ? '🔄 Classificazione da Clustering' 
                    : (isAgreement ? '✅ Accordo' : '⚠️ Disaccordo')} tra ML e LLM
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {isClusterPropagated 
                    ? 'Questo caso è stato classificato tramite propagazione da cluster - solo il modello LLM ha una predizione valida'
                    : (isAgreement 
                      ? 'I modelli sono d\'accordo sulla classificazione'
                      : 'I modelli hanno predizioni diverse - necessaria supervisione umana'
                    )
                  }
                </Typography>
              </Box>
            </CardContent>
          </Card>

          {/* Conversation */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Testo Conversazione
              </Typography>
              <Box 
                sx={{ 
                  backgroundColor: 'grey.100',
                  borderRadius: 1,
                  p: 2,
                  maxHeight: 400,
                  overflow: 'auto',
                  wordBreak: 'break-word'
                }}
              >
                <Typography variant="body2" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {caseItem.conversation_text}
                </Typography>
              </Box>
            </CardContent>
          </Card>

          {/* Human Decision Form */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Decisione Umana
              </Typography>

              <Grid container spacing={3} alignItems="flex-start">
                <Grid item xs={12} md={7}>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Scelte Rapide
                      </Typography>
                      <Box display="flex" flexWrap="wrap" gap={1}>
                        {!isClusterPropagated && (
                          <Button
                            variant={humanDecision === caseItem.ml_prediction ? 'contained' : 'outlined'}
                            color="primary"
                            size="small"
                            startIcon={<ThumbUpIcon />}
                            onClick={() => handleQuickDecision(caseItem.ml_prediction)}
                          >
                            Conferma ML
                          </Button>
                        )}
                        <Button
                          variant={humanDecision === caseItem.llm_prediction ? 'contained' : 'outlined'}
                          color="warning"
                          size="small"
                          startIcon={<ThumbUpIcon />}
                          onClick={() => handleQuickDecision(caseItem.llm_prediction)}
                        >
                          Conferma LLM
                        </Button>
                      </Box>
                    </Box>

                    <TextField
                      fullWidth
                      label="Classificazione Corretta"
                      value={humanDecision}
                      onChange={(e) => setHumanDecision(e.target.value)}
                      placeholder="Inserisci classificazione corretta o seleziona dai suggerimenti"
                      required
                    />

                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Confidenza della Decisione: {confidence.toFixed(1)}
                      </Typography>
                      <Slider
                        value={confidence}
                        onChange={(_, value) => setConfidence(value as number)}
                        min={0.1}
                        max={1.0}
                        step={0.1}
                        marks
                        valueLabelDisplay="auto"
                      />
                    </Box>

                    <TextField
                      fullWidth
                      multiline
                      rows={3}
                      label="Note (opzionale)"
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Aggiungi note sulla decisione..."
                    />

                    <Button
                      fullWidth
                      variant="contained"
                      color="success"
                      size="large"
                      onClick={handleSubmit}
                      disabled={loading || !humanDecision.trim()}
                      startIcon={<CheckCircleIcon />}
                    >
                      {loading ? 'Conferma in corso...' : 'Conferma Decisione'}
                    </Button>
                  </Box>
                </Grid>
                <Grid item xs={12} md={5}>
                  <TagSuggestions
                    tags={availableTags as any}
                    onTagSelect={handleTagSelect}
                    currentValue={humanDecision}
                    loading={tagsLoading}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Box>

        {/* Model + Cluster Column */}
        <Box
          flex={{ xs: '1 1 100%', lg: 1 }}
          sx={{
            minWidth: 280,
            maxWidth: { lg: 360 },
            display: 'flex',
            flexDirection: 'column',
            gap: 3,
            position: { lg: 'sticky' },
            top: { lg: 20 },
            alignSelf: 'flex-start'
          }}
        >
          {/* Model Predictions */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Predizioni Modelli
              </Typography>

              {/* ML Model */}
              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  ML Model:
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body1" color="primary" fontWeight="bold">
                    {caseItem.ml_prediction}
                  </Typography>
                  <Chip 
                    label={caseItem.ml_confidence.toFixed(3)} 
                    color="primary" 
                    size="small" 
                  />
                </Box>
              </Box>

              {/* LLM Model */}
              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  LLM Model:
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body1" color="warning.main" fontWeight="bold">
                    {caseItem.llm_prediction}
                  </Typography>
                  <Chip 
                    label={caseItem.llm_confidence.toFixed(3)} 
                    color="warning" 
                    size="small" 
                  />
                </Box>
              </Box>

              {/* Agreement Status */}
              <Box 
                sx={{ 
                  p: 2, 
                  borderRadius: 1,
                  backgroundColor: isAgreement ? 'success.light' : 'error.light',
                  color: isAgreement ? 'success.contrastText' : 'error.contrastText'
                }}
              >
                {isAgreement ? (
                  <><CheckCircleIcon sx={{ mr: 1 }} />Modelli in accordo</>
                ) : (
                  <><WarningIcon sx={{ mr: 1 }} />Modelli in disaccordo</>
                )}
              </Box>
            </CardContent>
          </Card>

          <ClusterContextPanel
            summary={clusterSummary}
            cases={clusterSessions}
            loading={clusterLoading}
            error={clusterError}
            selectedSessionId={selectedClusterSessionId}
            openedSessionId={caseItem.session_id}
            onSelectCase={handleClusterCaseSelect}
            navigatingSessionId={navigatingSessionId}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default CaseDetail;
