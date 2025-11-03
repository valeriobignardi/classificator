import React, { useState, useEffect } from 'react';
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
  Divider,
  IconButton
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  ThumbUp as ThumbUpIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';
import { ReviewCase } from '../types/ReviewCase';
import { Tenant } from '../types/Tenant';
import TagSuggestions from './TagSuggestions';

interface CaseDetailProps {
  case: ReviewCase;
  tenant: Tenant;
  onCaseResolved: () => void;
  onBack: () => void;
}

const CaseDetail: React.FC<CaseDetailProps> = ({
  case: caseItem,
  tenant,
  onCaseResolved,
  onBack
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

      <Box display="flex" gap={3} sx={{ overflow: 'hidden' }}>
        {/* Case Information */}
        <Box flex={2} sx={{ minWidth: 0, overflow: 'hidden' }}>
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
          <Card>
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
        </Box>

        {/* Decision Panel */}
        <Box flex={1} sx={{ minWidth: 0, maxWidth: '400px', position: 'sticky', top: 20, alignSelf: 'flex-start' }}>
          {/* Model Predictions */}
          <Card sx={{ mb: 3 }}>
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

          {/* Human Decision Form */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Decisione Umana
              </Typography>

              {/* Quick Buttons */}
              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Scelte Rapide:
                </Typography>
                {!isClusterPropagated && (
                  <Button
                    variant={humanDecision === caseItem.ml_prediction ? 'contained' : 'outlined'}
                    color="primary"
                    size="small"
                    startIcon={<ThumbUpIcon />}
                    onClick={() => handleQuickDecision(caseItem.ml_prediction)}
                    sx={{ mr: 1, mb: 1 }}
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
                  sx={{ mb: 1 }}
                >
                  Conferma LLM
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Manual Input and Tag Suggestions */}
              <Box display="flex" gap={2} flexDirection={{ xs: 'column', md: 'row' }}>
                <Box flex={1}>
                  <TextField
                    fullWidth
                    label="Classificazione Corretta"
                    value={humanDecision}
                    onChange={(e) => setHumanDecision(e.target.value)}
                    placeholder="Inserisci classificazione corretta o seleziona dai suggerimenti"
                    required
                    sx={{ mb: 2 }}
                  />

                  {/* Confidence Slider */}
                  <Box mb={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      Confidenza della Decisione: {confidence}
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
                </Box>
                
                <Box flex={1}>
                  {/* Tag Suggestions */}
                  <TagSuggestions
                    tags={availableTags as any}
                    onTagSelect={handleTagSelect}
                    currentValue={humanDecision}
                    loading={tagsLoading}
                  />
                </Box>
              </Box>

              {/* Notes */}
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Note (opzionale)"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Aggiungi note sulla decisione..."
                sx={{ mb: 2 }}
              />

              {/* Submit Button */}
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
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Box>
  );
};

export default CaseDetail;
