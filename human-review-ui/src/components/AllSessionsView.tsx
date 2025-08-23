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
  IconButton,
  Tooltip,
  TextField,
  FormControlLabel,
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  InputLabel,
  Select,
  MenuItem,
  FormControl
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Add as AddIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Assignment as AssignmentIcon,
  AutoFixHigh as AutoFixHighIcon,
  PlayArrow as PlayArrowIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { apiService } from '../services/apiService';
import FineTuningPanel from './FineTuningPanel';

interface Classification {
  tag_name: string;
  confidence: number;
  method: string;
  created_at: string;
  source?: 'database' | 'cache_pending';  // Nuovo campo per distinguere la fonte
}

interface Session {
  session_id: string;
  conversation_text: string;
  full_text: string;
  num_messages: number;
  num_user_messages: number;
  status: 'available' | 'in_review_queue' | 'reviewed';
  created_at: string;
  last_activity: string;
  classifications: Classification[];
}

interface AllSessionsViewProps {
  clientName: string;
  onSessionAdd: (sessionId: string) => void;
}

// Nuove interfacce per integrazione MongoDB
interface MongoTenant {
  tenant_name: string;
  client: string;
  session_count: number;
  classification_count: number;
}

interface MongoLabel {
  label: string;
  count: number;
  session_count: number;
  avg_confidence: number;
}

const AllSessionsView: React.FC<AllSessionsViewProps> = ({ clientName, onSessionAdd }) => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'available' | 'in_review_queue' | 'reviewed'>('all');
  const [includeReviewed, setIncludeReviewed] = useState(false);
  const [addingToQueue, setAddingToQueue] = useState<string | null>(null);
  const [currentLimit, setCurrentLimit] = useState<number>(50);
  const [selectedLabel, setSelectedLabel] = useState<string>('all'); // Nuovo stato per filtro etichetta
  const [recentlyAddedSessions, setRecentlyAddedSessions] = useState<Set<string>>(new Set()); // Per mostrare temporaneamente sessioni appena aggiunte
  
  // Nuovi stati per integrazione MongoDB  
  const [availableTenants, setAvailableTenants] = useState<MongoTenant[]>([]);
  const [selectedTenant, setSelectedTenant] = useState<string>('');
  const [availableLabels, setAvailableLabels] = useState<MongoLabel[]>([]);
  const [tenantLoading, setTenantLoading] = useState(false);
  const [labelsLoading, setLabelsLoading] = useState(false);
  const [mongoMode, setMongoMode] = useState<boolean>(true); // Usa MongoDB per default
  
  // Stati per classificazione completa
  const [classificationLoading, setClassificationLoading] = useState(false);
  const [classificationDialogOpen, setClassificationDialogOpen] = useState(false);
  const [classificationOptions, setClassificationOptions] = useState({
    confidence_threshold: 0.7,
    force_retrain: true,
    max_sessions: null as number | null,
    force_review: false,
    force_reprocess_all: false  // NUOVO: Riclassifica tutto dall'inizio
  });
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const loadSessions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getAllSessions(clientName, includeReviewed);
      setSessions(response.sessions);
    } catch (err) {
      setError('Errore di connessione');
      console.error('Errore getAllSessions:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessions();
  }, [clientName, includeReviewed]);

  const handleAddToQueue = async (sessionId: string) => {
    setAddingToQueue(sessionId);
    
    try {
      await apiService.addSessionToQueue(clientName, sessionId, 'manual_addition');
      
      // Aggiorna immediatamente lo stato locale della sessione
      setSessions(prevSessions => 
        prevSessions.map(session => 
          session.session_id === sessionId 
            ? { ...session, status: 'in_review_queue' as const }
            : session
        )
      );
      
      // Traccia la sessione come recentemente aggiunta per mostrarla temporaneamente
      setRecentlyAddedSessions(prev => {
        const newSet = new Set(prev);
        newSet.add(sessionId);
        return newSet;
      });
      
      // Rimuovi dalla lista dopo 3 secondi per evitare confusione
      setTimeout(() => {
        setRecentlyAddedSessions(prev => {
          const newSet = new Set(prev);
          newSet.delete(sessionId);
          return newSet;
        });
      }, 3000);
      
      onSessionAdd(sessionId);
      setSuccessMessage(`✅ Sessione ${sessionId.substring(0, 12)}... aggiunta alla Review Queue e rimossa dalla lista`);
      
      // Ricarica le sessioni per sincronizzare con il server (in background)
      setTimeout(() => {
        loadSessions();
      }, 500);
    } catch (err) {
      setError('Errore nell\'aggiunta alla queue');
      console.error('Errore addSessionToReviewQueue:', err);
    } finally {
      setAddingToQueue(null);
    }
  };

  const handleStartFullClassification = async () => {
    setClassificationLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await apiService.startFullClassification(clientName, {
        confidence_threshold: classificationOptions.confidence_threshold,
        force_retrain: classificationOptions.force_retrain,
        max_sessions: classificationOptions.max_sessions || undefined,
        force_review: classificationOptions.force_review,
        force_reprocess_all: classificationOptions.force_reprocess_all  // NUOVO parametro
      });

      if (response.success) {
        setSuccessMessage(
          `✅ Classificazione completata: ${response.sessions_processed || 0} sessioni processate` +
          (response.forced_review_count > 0 ? `, ${response.forced_review_count} casi forzati in coda per revisione` : '')
        );
        
        // Ricarica le sessioni dopo la classificazione
        setTimeout(() => {
          loadSessions();
        }, 1000);
      } else {
        setError(response.error || 'Errore durante la classificazione');
      }
    } catch (err: any) {
      setError(`Errore durante la classificazione: ${err.message}`);
      console.error('Classification error:', err);
    } finally {
      setClassificationLoading(false);
      setClassificationDialogOpen(false);
    }
  };

  // Funzione per estrarre tutte le etichette uniche dalle sessioni
  const getUniqueLabels = (): string[] => {
    const labels = new Set<string>();
    sessions.forEach(session => {
      session.classifications?.forEach(classification => {
        if (classification.tag_name) {
          labels.add(classification.tag_name);
        }
      });
    });
    return Array.from(labels).sort();
  };

  const filteredSessions = sessions.filter(session => {
    // Mostra sempre le sessioni recentemente aggiunte per 3 secondi
    if (recentlyAddedSessions.has(session.session_id)) {
      return true;
    }
    
    // Filtro per status
    if (filter !== 'all' && session.status !== filter) {
      return false;
    }
    
    // Filtro per etichetta
    if (selectedLabel !== 'all') {
      const hasSelectedLabel = session.classifications?.some(
        classification => classification.tag_name === selectedLabel
      );
      if (!hasSelectedLabel) {
        return false;
      }
    }
    
    return true;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'available': return 'success';
      case 'in_review_queue': return 'warning';
      case 'reviewed': return 'info';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'available': return <CheckCircleIcon />;
      case 'in_review_queue': return <ScheduleIcon />;
      case 'reviewed': return <AssignmentIcon />;
      default: return <CheckCircleIcon />; // Default icon instead of null
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString('it-IT', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'N/A';
    }
  };

  return (
    <Box>
      {/* Header with Controls */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" gutterBottom>
          Tutte le Sessioni ({filteredSessions.length}{selectedLabel !== 'all' ? ` - Filtrate per: ${selectedLabel}` : ''})
        </Typography>
        <Box display="flex" alignItems="center" gap={1}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AutoFixHighIcon />}
            onClick={() => setClassificationDialogOpen(true)}
            disabled={loading || classificationLoading}
            sx={{ mr: 1 }}
          >
            Classifica Tutte
          </Button>
          
          <Tooltip title="Aggiorna sessioni">
            <span>
              <IconButton onClick={loadSessions} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      </Box>

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Success Message */}
      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
          {successMessage}
        </Alert>
      )}

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Fine-Tuning Panel */}
      <Box mb={3}>
        <FineTuningPanel clientName={clientName} />
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filtri e Configurazione
          </Typography>
          <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
            {/* Status filter buttons */}
            <Box display="flex" gap={1}>
              {[
                { key: 'all', label: 'Tutte' },
                { key: 'available', label: 'Disponibili' },
                { key: 'in_review_queue', label: 'In Coda' },
                { key: 'reviewed', label: 'Reviewate' }
              ].map((filterOption) => (
                <Button
                  key={filterOption.key}
                  size="small"
                  variant={filter === filterOption.key ? "contained" : "outlined"}
                  onClick={() => setFilter(filterOption.key as any)}
                >
                  {filterOption.label}
                </Button>
              ))}
            </Box>

            {/* Filtro Etichette */}
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Filtra per Etichetta</InputLabel>
              <Select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
                label="Filtra per Etichetta"
              >
                <MenuItem value="all">
                  <em>Tutte le Etichette</em>
                </MenuItem>
                {getUniqueLabels().map((label) => (
                  <MenuItem key={label} value={label}>
                    {label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Switch
                  checked={includeReviewed}
                  onChange={(e) => setIncludeReviewed(e.target.checked)}
                  size="small"
                />
              }
              label="Includi sessioni reviewate"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Sessions Grid - SAME STYLE AS REVIEW DASHBOARD */}
      {filteredSessions.length === 0 && !loading ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <AssignmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Nessuna sessione trovata
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Non ci sono sessioni che corrispondono ai filtri selezionati.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Box display="flex" flexWrap="wrap" gap={3}>
          {filteredSessions.slice(0, currentLimit).map((session) => (
            <Box key={session.session_id} flex="1 1 calc(50% - 12px)" minWidth="300px">
              <Card 
                sx={{ 
                  height: '100%',
                  transition: 'all 0.2s',
                  // Evidenzia le sessioni recentemente aggiunte alla review queue
                  ...(recentlyAddedSessions.has(session.session_id) && {
                    border: '2px solid #ff9800',
                    backgroundColor: '#fff3e0',
                    boxShadow: '0 4px 20px rgba(255, 152, 0, 0.3)',
                    animation: 'pulse 2s infinite'
                  }),
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: recentlyAddedSessions.has(session.session_id) 
                      ? '0 6px 25px rgba(255, 152, 0, 0.4)' 
                      : 3
                  }
                }}
              >
                <CardContent>
                  {/* Header */}
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6" component="div">
                      Sessione: {session.session_id.substring(0, 12)}...
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      {/* Badge speciale per sessioni recentemente aggiunte */}
                      {recentlyAddedSessions.has(session.session_id) && (
                        <Chip
                          label="✅ APPENA AGGIUNTA"
                          color="warning"
                          size="small"
                          sx={{ 
                            fontWeight: 'bold',
                            fontSize: '0.7rem',
                            animation: 'pulse 1.5s infinite'
                          }}
                        />
                      )}
                      <Chip
                        icon={getStatusIcon(session.status)}
                        label={session.status.replace('_', ' ').toUpperCase()}
                        color={getStatusColor(session.status)}
                        size="small"
                      />
                      <Typography variant="body2" color="text.secondary">
                        {formatDate(session.created_at)}
                      </Typography>
                    </Box>
                  </Box>

                  {/* Classifications Display - SAME STYLE AS REVIEW DASHBOARD */}
                  {session.classifications && session.classifications.length > 0 ? (
                    <Box 
                      sx={{ 
                        border: '2px solid',
                        borderColor: 'success.main',
                        borderRadius: 2,
                        p: 2,
                        mb: 2,
                        background: 'linear-gradient(45deg, #e8f5e8 50%, #e8f5e8 50%)'
                      }}
                    >
                      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                        <Typography variant="subtitle2" fontWeight="bold" color="text.primary">
                          🏷️ CLASSIFICAZIONI
                        </Typography>
                        <Box display="flex" gap={1}>
                          {session.classifications.some(c => c.source === 'database') && (
                            <Chip
                              icon={<CheckCircleIcon />}
                              label="SALVATE"
                              color="success"
                              size="small"
                              sx={{ fontWeight: 'bold', fontSize: '0.7rem' }}
                            />
                          )}
                          {session.classifications.some(c => c.source === 'cache_pending') && (
                            <Chip
                              icon={<ScheduleIcon />}
                              label="PENDING"
                              color="warning"
                              size="small"
                              sx={{ fontWeight: 'bold', fontSize: '0.7rem' }}
                            />
                          )}
                        </Box>
                      </Box>

                      {session.classifications.map((classification, index) => {
                        const isPending = classification.source === 'cache_pending';
                        const borderColor = isPending ? '#ff9800' : '#4caf50';
                        const backgroundColor = isPending ? '#fff3e0' : '#f1f8e9';
                        const textColor = isPending ? 'warning.main' : 'success.main';
                        
                        return (
                          <Box 
                            key={index}
                            sx={{
                              border: `1px solid ${borderColor}`,
                              borderRadius: 1,
                              p: 1.5,
                              mb: index < session.classifications.length - 1 ? 1 : 0,
                              backgroundColor: backgroundColor,
                              position: 'relative'
                            }}
                          >
                            {/* Indicatore stato classificazione */}
                            <Box display="flex" alignItems="center" justifyContent="space-between" mb={0.5}>
                              <Typography variant="subtitle2" fontWeight="bold" color={textColor}>
                                📋 {classification.method?.toUpperCase() || 'AUTOMATIC'}
                              </Typography>
                              <Chip
                                icon={isPending ? <ScheduleIcon /> : <CheckCircleIcon />}
                                label={isPending ? 'PENDING' : 'SALVATA'}
                                color={isPending ? 'warning' : 'success'}
                                size="small"
                                sx={{ 
                                  fontWeight: 'bold',
                                  fontSize: '0.7rem'
                                }}
                              />
                            </Box>
                            
                            <Typography 
                              variant="h6" 
                              color={textColor}
                              sx={{ fontWeight: 'bold', my: 0.5 }}
                            >
                              {classification.tag_name}
                            </Typography>
                            
                            <Box display="flex" gap={1} flexWrap="wrap">
                              <Chip
                                label={`Confidenza: ${(classification.confidence * 100).toFixed(1)}%`}
                                color={getConfidenceColor(classification.confidence)}
                                size="small"
                                sx={{ fontWeight: 'bold' }}
                              />
                              <Chip
                                label={formatDate(classification.created_at)}
                                size="small"
                                color="default"
                              />
                              {isPending && (
                                <Chip
                                  label="⏳ Non ancora salvata"
                                  color="warning"
                                  size="small"
                                  sx={{ fontWeight: 'bold' }}
                                />
                              )}
                            </Box>
                          </Box>
                        );
                      })}
                    </Box>
                  ) : (
                    <Box 
                      sx={{ 
                        border: '2px dashed',
                        borderColor: 'grey.400',
                        borderRadius: 2,
                        p: 2,
                        mb: 2,
                        background: 'linear-gradient(45deg, #f5f5f5 50%, #f5f5f5 50%)',
                        textAlign: 'center'
                      }}
                    >
                      <Typography variant="subtitle2" color="text.secondary">
                        ❓ NON ANCORA CLASSIFICATA
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Sessione non ancora processata dal sistema di classificazione
                      </Typography>
                    </Box>
                  )}

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
                      {session.conversation_text.length > 200
                        ? `${session.conversation_text.substring(0, 200)}...`
                        : session.conversation_text}
                    </Typography>
                  </Box>

                  {/* Metrics and Actions */}
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        📝 {session.num_messages} messaggi totali
                      </Typography>
                      <br />
                      <Typography variant="caption" color="text.secondary">
                        👤 {session.num_user_messages} utente
                      </Typography>
                    </Box>
                    
                    <Box>
                      {session.status === 'available' && (
                        <Button 
                          size="small" 
                          variant="contained"
                          startIcon={<AddIcon />}
                          onClick={() => handleAddToQueue(session.session_id)}
                          disabled={addingToQueue === session.session_id}
                          color="primary"
                        >
                          {addingToQueue === session.session_id ? 'Aggiungendo...' : 'Aggiungi a Review'}
                        </Button>
                      )}
                      {session.status === 'in_review_queue' && (
                        <Chip
                          icon={<ScheduleIcon />}
                          label="Già in Review Queue"
                          color="warning"
                          size="small"
                          sx={{ fontWeight: 'bold' }}
                        />
                      )}
                      {session.status === 'reviewed' && (
                        <Chip
                          label="Completata"
                          color="info"
                          size="small"
                        />
                      )}
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          ))}
        </Box>
      )}

      {/* Load More if needed */}
      {filteredSessions.length > currentLimit && (
        <Box textAlign="center" mt={3}>
          <Button 
            variant="outlined" 
            onClick={() => setCurrentLimit(prev => prev + 50)}
          >
            Carica altre sessioni ({filteredSessions.length - currentLimit} rimanenti)
          </Button>
        </Box>
      )}

      {/* Classificazione Completa Dialog */}
      <Dialog 
        open={classificationDialogOpen} 
        onClose={() => setClassificationDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={1}>
            <AutoFixHighIcon color="primary" />
            Classificazione Completa - {clientName}
          </Box>
        </DialogTitle>
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="body2">
              <strong>🎯 Classificazione Automatica Completa</strong><br />
              Questo processo classificherà <strong>TUTTE le sessioni</strong> del database usando l'intelligenza artificiale.
              Puoi personalizzare i parametri di classificazione e scegliere se forzare la revisione umana.
            </Typography>
          </Alert>

          <Box display="flex" flexDirection="column" gap={3} mt={2}>
            {/* Soglia di Confidenza */}
            <FormControl fullWidth>
              <TextField
                label="Soglia di Confidenza"
                type="number"
                value={classificationOptions.confidence_threshold}
                onChange={(e) => setClassificationOptions({
                  ...classificationOptions,
                  confidence_threshold: parseFloat(e.target.value) || 0.7
                })}
                inputProps={{ min: 0.1, max: 1.0, step: 0.1 }}
                helperText="Soglia minima di confidenza per accettare una classificazione automatica (0.1-1.0)"
              />
            </FormControl>

            {/* Limite Sessioni */}
            <FormControl fullWidth>
              <InputLabel>Numero Massimo Sessioni</InputLabel>
              <Select
                value={classificationOptions.max_sessions || 'all'}
                onChange={(e) => setClassificationOptions({
                  ...classificationOptions,
                  max_sessions: e.target.value === 'all' ? null : parseInt(String(e.target.value))
                })}
              >
                <MenuItem value="all">🌟 TUTTE LE SESSIONI</MenuItem>
                <MenuItem value={100}>📊 100 sessioni</MenuItem>
                <MenuItem value={500}>📈 500 sessioni</MenuItem>
                <MenuItem value={1000}>🔥 1000 sessioni</MenuItem>
                <MenuItem value={2000}>⚡ 2000 sessioni</MenuItem>
                <MenuItem value={5000}>🚀 5000 sessioni</MenuItem>  {/* AGGIUNTA OPZIONE: per scenari con molte sessioni */}
              </Select>
            </FormControl>

            {/* Opzioni Avanzate */}
            <Box>
              <Typography variant="h6" gutterBottom>
                Opzioni Avanzate
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={classificationOptions.force_retrain}
                    onChange={(e) => setClassificationOptions({
                      ...classificationOptions,
                      force_retrain: e.target.checked
                    })}
                  />
                }
                label="Riaddestra modello ML prima della classificazione"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={classificationOptions.force_review}
                    onChange={(e) => setClassificationOptions({
                      ...classificationOptions,
                      force_review: e.target.checked
                    })}
                  />
                }
                label="Forza revisione umana per tutti i casi classificati"
              />
              
              <FormControlLabel
                control={
                  <Switch
                    checked={classificationOptions.force_reprocess_all}
                    onChange={(e) => setClassificationOptions({
                      ...classificationOptions,
                      force_reprocess_all: e.target.checked
                    })}
                    color="error"
                  />
                }
                label="🔄 RICLASSIFICA TUTTO DALL'INIZIO (cancella classificazioni esistenti)"
                sx={{ 
                  '& .MuiFormControlLabel-label': { 
                    color: classificationOptions.force_reprocess_all ? 'error.main' : 'inherit',
                    fontWeight: classificationOptions.force_reprocess_all ? 'bold' : 'normal'
                  }
                }}
              />
            </Box>

            {/* Alert dinamico */}
            {classificationOptions.max_sessions === null && (
              <Alert severity="warning">
                <Typography variant="body2">
                  <strong>⚠️ ATTENZIONE:</strong> Hai scelto di processare <strong>TUTTE LE SESSIONI</strong> del database.
                  Questa operazione potrebbe richiedere molto tempo e risorse computazionali.
                </Typography>
              </Alert>
            )}

            {classificationOptions.force_review && (
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>👁️ REVISIONE FORZATA:</strong> Tutte le sessioni classificate saranno automaticamente 
                  aggiunte alla coda di revisione umana per controllo qualità.
                </Typography>
              </Alert>
            )}

            {classificationOptions.force_reprocess_all && (
              <Alert severity="error">
                <Typography variant="body2">
                  <strong>⚠️ ATTENZIONE - RICLASSIFICAZIONE COMPLETA:</strong> Questa opzione cancellerà <strong>TUTTE LE CLASSIFICAZIONI ESISTENTI</strong> 
                  dal database e riprocesserà ogni sessione da zero. Questa operazione è <strong>IRREVERSIBILE</strong> e potrebbe richiedere molto tempo.
                  <br /><br />
                  <strong>🔄 Conseguenze:</strong>
                  <br />• Tutte le classificazioni automatiche esistenti verranno eliminate
                  <br />• Tutte le sessioni verranno riprocessate dall'inizio
                  <br />• I dati di training precedenti verranno mantenuti
                </Typography>
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClassificationDialogOpen(false)}>
            Annulla
          </Button>
          <Button
            variant="contained"
            onClick={handleStartFullClassification}
            disabled={classificationLoading}
            startIcon={classificationLoading ? <LinearProgress /> : <PlayArrowIcon />}
          >
            {classificationLoading ? 'Classificando...' : 'Avvia Classificazione'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AllSessionsView;
