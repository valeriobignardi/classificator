/**
 * =====================================================================
 * EXAMPLE MANAGER COMPONENT - GESTIONE ESEMPI MULTI-TENANT
 * =====================================================================
 * Autore: Sistema di Classificazione AI
 * Data: 2025-08-25
 * Descrizione: Componente React per gestione esempi formattati
 *              UTENTE:/ASSISTENTE: per placeholder {examples_text}
 * =====================================================================
 */

import React, { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Button,
  Chip,
  Alert,
  AlertTitle,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tooltip,
  Card,
  CardContent,
  Paper,
  Stack
} from '@mui/material';
import {
  ExpandMore,
  Save,
  Add,
  Delete,
  Visibility,
  Chat,
  Group,
  AutoFixHigh
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';

interface Example {
  id: number;
  esempio_name: string;
  esempio_type: string;
  categoria: string;
  livello_difficolta: string;
  description: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface ExampleContent {
  examples_text: string;
  num_conversations: number;
  length: number;
}

interface ExampleManagerProps {
  open: boolean;
}

/**
 * Componente per gestione esempi conversazioni multi-tenant
 * @param open - Se il componente deve essere mostrato
 * @returns ExampleManager component
 * Ultima modifica: 2025-08-25
 */
const ExampleManager: React.FC<ExampleManagerProps> = ({ open }) => {
  const { selectedTenant } = useTenant();
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Stati per nuovo esempio
  const [showAddForm, setShowAddForm] = useState(false);
  const [newExample, setNewExample] = useState({
    esempio_name: '',
    esempio_content: '',
    description: '',
    categoria: '',
    livello_difficolta: 'MEDIO'
  });
  
  // Stati per preview placeholder
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewContent, setPreviewContent] = useState<ExampleContent | null>(null);

  /**
   * Carica esempi dal backend per il tenant corrente
   * Input: Usa selectedTenant dal context
   * Output: Aggiorna state examples[]
   * Ultima modifica: 2025-08-25
   */
    const loadExamples = useCallback(async () => {
    if (!selectedTenant) {
      setExamples([]);
      return;
    }

    setLoading(true);
    try {
      console.log('🔍 [DEBUG] Caricamento esempi per tenant:', selectedTenant.tenant_id);
      
      const examples = await apiService.getExamples(selectedTenant.tenant_id);
      console.log('✅ [DEBUG] Esempi caricati:', examples);
      
      setExamples(examples);
    } catch (error) {
      console.error('❌ [DEBUG] Errore caricamento esempi:', error);
      setError(`Errore durante il caricamento degli esempi: ${error}`);
      setExamples([]);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant]);

  /**
   * Crea nuovo esempio nel database
   * Input: newExample state object
   * Output: Nuovo esempio nel database, ricarica lista
   * Ultima modifica: 2025-08-25
   */
  const createExample = async () => {
    if (!selectedTenant?.tenant_id) return;

    try {
      setLoading(true);
      setError(null);

      const exampleData = {
        tenant_id: selectedTenant.tenant_id,
        engine: 'LLM',  // Esplicito
        esempio_type: 'CONVERSATION',  // Esplicito
        ...newExample
      };

      console.log('🔍 [DEBUG] React createExample - Dati inviati:', exampleData);

      await apiService.createExample(exampleData);

      setSuccess(`Esempio "${newExample.esempio_name}" creato con successo!`);
      setShowAddForm(false);
      setNewExample({
        esempio_name: '',
        esempio_content: '',
        description: '',
        categoria: '',
        livello_difficolta: 'MEDIO'
      });
      loadExamples();

    } catch (err) {
      console.error('Errore creazione esempio:', err);
      setError(err instanceof Error ? err.message : 'Errore creazione esempio');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Elimina esempio (soft delete)
   * Input: exampleId, exampleName per conferma
   * Output: Esempio eliminato dal database
   * Ultima modifica: 2025-08-25
   */
  const deleteExample = async (exampleId: number, exampleName: string) => {
    if (!selectedTenant?.tenant_id) return;

    const confirmed = window.confirm(`Vuoi eliminare l'esempio "${exampleName}"?`);
    if (!confirmed) return;

    try {
      setLoading(true);
      setError(null);

      await apiService.deleteExample(exampleId, selectedTenant.tenant_id);

      setSuccess(`Esempio "${exampleName}" eliminato con successo!`);
      loadExamples();

    } catch (err) {
      console.error('Errore eliminazione esempio:', err);
      setError(err instanceof Error ? err.message : 'Errore eliminazione esempio');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Preview esempi formattati per placeholder
   * Input: selectedTenant per filtro
   * Output: Contenuto formattato del placeholder
   * Ultima modifica: 2025-08-25
   */
  const previewPlaceholder = async () => {
    if (!selectedTenant?.tenant_id) return;

    try {
      setLoading(true);
      setError(null);

      const exampleData = await apiService.getExamplesForPlaceholder(selectedTenant.tenant_id, 3);
      
      // Usa direttamente i dati dall'API
      const exampleContent: ExampleContent = {
        examples_text: exampleData.examples_text,
        num_conversations: exampleData.num_conversations,
        length: exampleData.length
      };
      
      setPreviewContent(exampleContent);
      setPreviewOpen(true);

    } catch (err) {
      console.error('Errore preview placeholder:', err);
      setError(err instanceof Error ? err.message : 'Errore preview placeholder');
    } finally {
      setLoading(false);
    }
  };

  // Carica esempi quando cambia tenant
  useEffect(() => {
    loadExamples();
  }, [loadExamples]);

  // Auto-clear messaggi dopo 5 secondi
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 8000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  if (!open) return null;

  return (
    <Box>
      {/* Header con statistiche */}
      <Box sx={{ mb: 3 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 500, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chat color="primary" />
              Gestione Esempi Conversazioni
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              Esempi per placeholder examples_text nei prompt
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              size="small"
              onClick={previewPlaceholder}
              startIcon={<Visibility />}
              disabled={loading || examples.length === 0}
            >
              Preview examples_text
            </Button>
            <Button
              variant="contained"
              size="small"
              onClick={() => setShowAddForm(true)}
              startIcon={<Add />}
              disabled={loading}
            >
              Nuovo Esempio
            </Button>
          </Stack>
        </Stack>

        {/* Statistiche rapide */}
        <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip 
            icon={<Group />} 
            label={`${examples.length} esempi totali`} 
            size="small" 
            variant="outlined" 
          />
          <Chip 
            icon={<Chat />} 
            label={`${examples.filter(e => e.is_active).length} attivi`} 
            size="small" 
            color="primary"
            variant="outlined" 
          />
        </Box>
      </Box>

      {/* Alert messaggi */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          <AlertTitle>Errore</AlertTitle>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          <AlertTitle>Successo</AlertTitle>
          {success}
        </Alert>
      )}

      {/* Loading indicator */}
      {loading && (
        <Box display="flex" justifyContent="center" sx={{ mb: 2 }}>
          <CircularProgress size={24} />
        </Box>
      )}

      {/* Messaggio tenant */}
      {!selectedTenant && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <AlertTitle>Seleziona Tenant</AlertTitle>
          Seleziona un tenant per gestire gli esempi
        </Alert>
      )}

      {/* Lista esempi */}
      {selectedTenant && (
        <Box sx={{ mb: 3 }}>
          {examples.length === 0 && !loading ? (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 4 }}>
                <Chat sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Nessun esempio trovato
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Crea il tuo primo esempio per iniziare
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<Add />}
                  onClick={() => setShowAddForm(true)}
                >
                  Crea Primo Esempio
                </Button>
              </CardContent>
            </Card>
          ) : (
            examples.map((example) => (
              <Box key={example.id} sx={{ position: 'relative', mb: 1, '&:hover .delete-button': { opacity: 1 } }}>
                <Accordion sx={{ mb: 0 }}>
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{ pr: 8 }} // Space for external button
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 2 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                          {example.esempio_name}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, mt: 0.5, flexWrap: 'wrap' }}>
                          <Chip 
                            label={example.categoria || 'Nessuna categoria'} 
                            size="small" 
                            variant="outlined"
                            color="primary"
                          />
                          <Chip 
                            label={example.livello_difficolta} 
                            size="small" 
                            variant="outlined"
                            color={
                              example.livello_difficolta === 'FACILE' ? 'success' :
                              example.livello_difficolta === 'MEDIO' ? 'warning' : 'error'
                            }
                          />
                          <Chip 
                            label={example.is_active ? 'Attivo' : 'Inattivo'} 
                            size="small" 
                            color={example.is_active ? 'success' : 'default'}
                          />
                        </Box>
                      </Box>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      {example.description && (
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          {example.description}
                        </Typography>
                      )}
                      <Typography variant="caption" color="text.secondary">
                        Creato: {new Date(example.created_at).toLocaleString('it-IT')}
                        {example.updated_at !== example.created_at && (
                          <span> • Aggiornato: {new Date(example.updated_at).toLocaleString('it-IT')}</span>
                        )}
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
                
                {/* Pulsante elimina posizionato FUORI dall'Accordion - NO HTML NESTING! */}
                <Box
                  className="delete-button"
                  sx={{ 
                    position: 'absolute',
                    right: 48,
                    top: 12,
                    opacity: 0,
                    transition: 'opacity 0.2s',
                    zIndex: 10
                  }}
                >
                  <Tooltip title="Elimina esempio">
                    <IconButton
                      size="small"
                      color="error"
                      onClick={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        deleteExample(example.id, example.esempio_name);
                      }}
                      sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
                    >
                      <Delete />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
            ))
          )}
        </Box>
      )}

      {/* Dialog per nuovo esempio */}
      <Dialog open={showAddForm} onClose={() => setShowAddForm(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Add color="primary" />
            Nuovo Esempio Conversazione
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
              <TextField
                fullWidth
                label="Nome Esempio"
                value={newExample.esempio_name}
                onChange={(e) => setNewExample(prev => ({ ...prev, esempio_name: e.target.value }))}
                placeholder="es. richiesta_preventivo_auto"
                required
              />
              <TextField
                fullWidth
                label="Categoria"
                value={newExample.categoria}
                onChange={(e) => setNewExample(prev => ({ ...prev, categoria: e.target.value }))}
                placeholder="es. preventivi_auto"
              />
            </Stack>
            
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
              <FormControl fullWidth>
                <InputLabel>Livello Difficoltà</InputLabel>
                <Select
                  value={newExample.livello_difficolta}
                  onChange={(e) => setNewExample(prev => ({ ...prev, livello_difficolta: e.target.value }))}
                  label="Livello Difficoltà"
                >
                  <MenuItem value="FACILE">Facile</MenuItem>
                  <MenuItem value="MEDIO">Medio</MenuItem>
                  <MenuItem value="DIFFICILE">Difficile</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                label="Descrizione"
                value={newExample.description}
                onChange={(e) => setNewExample(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Breve descrizione dell'esempio"
              />
            </Stack>
            
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 500 }}>
                Contenuto Conversazione
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
                Formato: UTENTE: testo domanda (a capo) ASSISTENTE: testo risposta (a capo) e così via...
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={8}
                value={newExample.esempio_content}
                onChange={(e) => setNewExample(prev => ({ ...prev, esempio_content: e.target.value }))}
                placeholder="UTENTE: La mia domanda qui\n\nASSISTENTE: La risposta dell'assistente qui\n\nUTENTE: Un'altra domanda\n\nASSISTENTE: Un'altra risposta"
                required
                sx={{ 
                  '& textarea': { 
                    fontFamily: 'monospace', 
                    fontSize: '0.875rem' 
                  } 
                }}
              />
            </Box>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAddForm(false)} disabled={loading}>
            Annulla
          </Button>
          <Button
            onClick={createExample}
            variant="contained"
            disabled={loading || !newExample.esempio_name || !newExample.esempio_content}
            startIcon={loading ? <CircularProgress size={16} /> : <Save />}
          >
            Crea Esempio
          </Button>
        </DialogActions>
      </Dialog>

      {/* Dialog per preview placeholder */}
      <Dialog open={previewOpen} onClose={() => setPreviewOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoFixHigh color="primary" />
            Preview Placeholder examples_text
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {previewContent && (
            <Box>
              <Box sx={{ mb: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Chip 
                  label={`${previewContent.num_conversations} conversazioni`} 
                  color="primary" 
                  size="small"
                />
                <Chip 
                  label={`${previewContent.length} caratteri totali`} 
                  variant="outlined" 
                  size="small"
                />
              </Box>
              <Paper sx={{ p: 2, bgcolor: '#f8f9fa' }}>
                <Typography 
                  variant="body2" 
                  component="pre" 
                  sx={{ 
                    whiteSpace: 'pre-wrap', 
                    fontFamily: 'monospace',
                    fontSize: '0.875rem',
                    lineHeight: 1.6,
                    margin: 0
                  }}
                >
                  {previewContent.examples_text}
                </Typography>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>
            Chiudi
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ExampleManager;
