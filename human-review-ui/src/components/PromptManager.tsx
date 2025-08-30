/**
 * =====================================================================
 * PROMPT MANAGER COMPONENT - GESTIONE PROMPT MULTI-TENANT
 * =====================================================================
 * Autore: Sistema di Classificazione AI
 * Data: 2025-08-21
 * Descrizione: Componente React per gestione prompt database-driven
 *              con supporto multi-tenant e editing avanzato
 * =====================================================================
 */

import React, { useState, useEffect, useCallback } from 'react';
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
  SelectChangeEvent
} from '@mui/material';
import {
  ExpandMore,
  Edit,
  Save,
  Cancel,
  Add,
  Delete,
  Visibility,
  Code
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';

interface Prompt {
  id: number;
  tenant_id: string;
  engine: string;
  prompt_type: string;
  prompt_name: string;
  content: string;
  variables: Record<string, any>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface PromptManagerProps {
  open: boolean;
}

const PromptManager: React.FC<PromptManagerProps> = ({ open }) => {
  const { selectedTenant } = useTenant();
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editingPrompt, setEditingPrompt] = useState<Prompt | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewContent, setPreviewContent] = useState('');

  // Form states per nuovo prompt
  const [newPrompt, setNewPrompt] = useState({
    engine: 'LLM',
    prompt_type: 'SYSTEM',
    prompt_name: '',
    content: ''
  });
  const [showAddForm, setShowAddForm] = useState(false);

  /**
   * Carica prompt dal backend per il tenant corrente
   */
    const loadPrompts = useCallback(async () => {
        if (!selectedTenant?.tenant_id) {
            setPrompts([]);
            return;
        }

        try {
            setLoading(true);
            setError(null);

            const response = await fetch(`/api/prompts/tenant/${selectedTenant.tenant_id}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`Errore HTTP: ${response.status}`);
            }

            const data = await response.json();
            setPrompts(Array.isArray(data) ? data : []);
            
        } catch (err) {
            console.error('Errore caricamento prompt:', err);
            setError(err instanceof Error ? err.message : 'Errore caricamento prompt');
            setPrompts([]);
        } finally {
            setLoading(false);
        }
    }, [selectedTenant]);

  /**
   * Salva modifiche prompt esistente
   */
  const savePrompt = async (prompt: Prompt) => {
    try {
      const response = await fetch(`/api/prompts/${prompt.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: prompt.content,
          variables: prompt.variables,
          is_active: prompt.is_active
        }),
      });

      if (!response.ok) {
        throw new Error(`Errore salvataggio: ${response.statusText}`);
      }

      await loadPrompts();
      setEditingPrompt(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore salvataggio');
    }
  };

  /**
   * Copia prompt da Humanitas per il tenant selezionato
   * 
   * Autore: Sistema
   * Data: 2025-08-24
   * Descrizione: Sostituisce creazione manuale con copia template da Humanitas
   */
  const copyPromptsFromHumanitas = async () => {
    if (!selectedTenant) {
      setError('Nessun tenant selezionato');
      return;
    }

    try {
      setLoading(true);
      console.log('🔄 [DEBUG] Inizio copia prompt da Humanitas per tenant:', selectedTenant.tenant_name);
      
      const result = await apiService.copyPromptsFromHumanitas(selectedTenant.tenant_id);
      
      console.log('✅ [DEBUG] Copia completata:', result);
      
      // Ricarica i prompt per mostrare quelli nuovi
      await loadPrompts();
      
      // Chiudi il form e mostra messaggio di successo
      setShowAddForm(false);
      setError(null);
      
      // Mostra messaggio di successo
      alert(`✅ Copiati ${result.copied_prompts} prompt dal tenant Humanitas!\n\nOra puoi modificarli secondo le tue esigenze.`);
      
    } catch (error) {
      console.error('❌ [DEBUG] Errore copia prompt:', error);
      setError(`Errore durante la copia: ${error instanceof Error ? error.message : 'Errore sconosciuto'}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Crea nuovo prompt (metodo legacy, ora usato per prompt personalizzati)
   */
  const createPrompt = async () => {
    if (!selectedTenant || !newPrompt.prompt_name || !newPrompt.content) {
      setError('Compilare tutti i campi obbligatori');
      return;
    }

    try {
      const response = await fetch('/api/prompts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tenant_id: selectedTenant.tenant_id,
          tenant_name: selectedTenant.tenant_name,  // ✅ AGGIUNTO campo mancante
          prompt_name: newPrompt.prompt_name,       // ✅ AGGIUNTO campo mancante
          engine: newPrompt.engine,                 // ✅ AGGIUNTO campo mancante
          prompt_type: newPrompt.prompt_type,
          content: newPrompt.content,
          variables: {},
          is_active: true
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        console.log('❌ Errore response:', errorData);
        throw new Error(`Errore creazione: ${response.status} - ${errorData}`);
      }

      await loadPrompts();
      setShowAddForm(false);
      setNewPrompt({ engine: 'LLM', prompt_type: 'SYSTEM', prompt_name: '', content: '' });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore creazione');
    }
  };

  /**
   * Elimina prompt
   */
  const deletePrompt = async (promptId: number) => {
    console.log('🗑️ deletePrompt called with ID:', promptId);
    
    if (!window.confirm('Sei sicuro di voler eliminare questo prompt?')) {
      console.log('❌ Eliminazione annullata dall\'utente');
      return;
    }

    try {
      console.log('🌐 Making DELETE request to:', `/api/prompts/${promptId}`);
      
      const response = await fetch(`/api/prompts/${promptId}`, {
        method: 'DELETE',
      });

      console.log('📡 Response status:', response.status);
      console.log('📡 Response ok:', response.ok);

      if (!response.ok) {
        const errorText = await response.text();
        console.log('❌ Response error:', errorText);
        throw new Error(`Errore eliminazione: ${response.status} - ${errorText}`);
      }

      console.log('✅ Prompt eliminato con successo, ricaricando lista...');
      await loadPrompts();
      
    } catch (err) {
      console.error('❌ Errore eliminazione prompt:', err);
      setError(err instanceof Error ? err.message : 'Errore eliminazione');
    }
  };

  /**
   * Anteprima prompt con variabili risolte
   */
  const previewPrompt = async (prompt: Prompt) => {
    try {
      const response = await fetch(`/api/prompts/${prompt.id}/preview`);
      if (!response.ok) {
        throw new Error(`Errore anteprima: ${response.statusText}`);
      }
      const data = await response.json();
      setPreviewContent(data.content);
      setPreviewOpen(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore anteprima');
    }
  };

  useEffect(() => {
    if (selectedTenant && open) {
      loadPrompts();
    }
  }, [selectedTenant, open, loadPrompts]);

  if (!selectedTenant) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">
          Seleziona un tenant per gestire i prompt
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      {/* Header sezione */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
          <Code sx={{ mr: 1 }} />
          Gestione Prompt
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setShowAddForm(true)}
          size="small"
        >
          Nuovo Prompt
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Form creazione nuovo prompt */}
      {showAddForm && (
        <Card sx={{ mb: 3, border: '2px solid', borderColor: 'primary.main' }}>
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Crea Nuovo Prompt
            </Typography>
            
            {/* Alert informativo per guidare l'utente */}
            <Alert severity="info" sx={{ mb: 3 }}>
              <AlertTitle>💡 Suggerimento</AlertTitle>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>🔄 Copia da Template:</strong> Copia automaticamente tutti i prompt pre-configurati dal tenant template (consigliato per iniziare velocemente)
              </Typography>
              <Typography variant="body2">
                <strong>➕ Crea Vuoto:</strong> Crea un singolo prompt vuoto da configurare manualmente
              </Typography>
            </Alert>
            
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Engine</InputLabel>
                <Select
                  value={newPrompt.engine}
                  onChange={(e: SelectChangeEvent) => 
                    setNewPrompt(prev => ({ ...prev, engine: e.target.value }))
                  }
                >
                  <MenuItem value="LLM">LLM</MenuItem>
                  <MenuItem value="ML">ML</MenuItem>
                  <MenuItem value="FINETUNING">Fine-tuning</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Tipo</InputLabel>
                <Select
                  value={newPrompt.prompt_type}
                  onChange={(e: SelectChangeEvent) => 
                    setNewPrompt(prev => ({ ...prev, prompt_type: e.target.value }))
                  }
                >
                  <MenuItem value="SYSTEM">System</MenuItem>
                  <MenuItem value="TEMPLATE">Template</MenuItem>
                  <MenuItem value="SPECIALIZED">Specialized</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <TextField
              fullWidth
              label="Nome Prompt"
              value={newPrompt.prompt_name}
              onChange={(e) => setNewPrompt(prev => ({ ...prev, prompt_name: e.target.value }))}
              sx={{ mb: 2 }}
              size="small"
            />

            <TextField
              fullWidth
              label="Contenuto Prompt"
              value={newPrompt.content}
              onChange={(e) => setNewPrompt(prev => ({ ...prev, content: e.target.value }))}
              multiline
              rows={8}
              sx={{ mb: 2 }}
            />

            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                onClick={copyPromptsFromHumanitas}
                startIcon={<Add />}
                color="primary"
                sx={{ flexGrow: 1 }}
              >
                🔄 Copia da Template
              </Button>
              <Button
                variant="outlined"
                onClick={createPrompt}
                color="secondary"
                sx={{ flexGrow: 1 }}
              >
                SALVA
              </Button>
              <Button
                variant="outlined"
                onClick={() => setShowAddForm(false)}
                startIcon={<Cancel />}
              >
                Annulla
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Lista prompt esistenti */}
      {loading ? (
        <Box display="flex" justifyContent="center" p={3}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {prompts.length === 0 ? (
            <Alert severity="info">
              Nessun prompt configurato per questo tenant.
              Clicca "Nuovo Prompt" per iniziare.
            </Alert>
          ) : (
            prompts.map((prompt) => (
              <Box key={prompt.id} sx={{ position: 'relative', mb: 1, '&:hover .action-buttons': { opacity: 1 } }}>
                <Accordion sx={{ mb: 0 }}>
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    sx={{ pr: 8 }} // Space for external buttons
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="subtitle1" fontWeight="medium">
                          {prompt.prompt_name}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                          <Chip
                            label={prompt.engine}
                            size="small"
                            color="primary"
                            variant="outlined"
                          />
                          <Chip
                            label={prompt.prompt_type}
                            size="small"
                            color="secondary"
                            variant="outlined"
                          />
                          <Chip
                            label={prompt.is_active ? "Attivo" : "Inattivo"}
                            size="small"
                            color={prompt.is_active ? "success" : "default"}
                          />
                        </Box>
                      </Box>
                    </Box>
                  </AccordionSummary>
                
                  <AccordionDetails>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Ultimo aggiornamento: {new Date(prompt.updated_at).toLocaleString('it-IT')}
                    </Typography>
                    
                    {editingPrompt?.id === prompt.id ? (
                      <Box>
                        <TextField
                          fullWidth
                          label="Contenuto Prompt"
                          value={editingPrompt.content}
                          onChange={(e) => 
                          setEditingPrompt(prev => prev ? { ...prev, content: e.target.value } : null)
                        }
                        multiline
                        rows={10}
                        sx={{ mb: 2 }}
                      />
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button
                          variant="contained"
                          onClick={() => savePrompt(editingPrompt)}
                        >
                          SALVA
                        </Button>
                        <Button
                          variant="outlined"
                          onClick={() => setEditingPrompt(null)}
                          startIcon={<Cancel />}
                        >
                          Annulla
                        </Button>
                      </Box>
                    </Box>
                  ) : (
                    <Box 
                      sx={{ 
                        backgroundColor: 'grey.50',
                        p: 2,
                        borderRadius: 1,
                        fontFamily: 'monospace',
                        fontSize: '0.875rem',
                        whiteSpace: 'pre-wrap',
                        maxHeight: 300,
                        overflow: 'auto'
                      }}
                    >
                      {prompt.content}
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>
              
              {/* Action buttons positioned OUTSIDE the Accordion structure - NO HTML NESTING! */}
              <Box
                className="action-buttons"
                sx={{ 
                  position: 'absolute',
                  right: 48,
                  top: 12,
                  display: 'flex',
                  gap: 0.5,
                  opacity: 0,
                  transition: 'opacity 0.2s',
                  zIndex: 10
                }}
              >
                <Tooltip title="Anteprima">
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      previewPrompt(prompt);
                    }}
                    sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
                  >
                    <Visibility />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Modifica">
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      setEditingPrompt(prompt);
                    }}
                    sx={{ bgcolor: 'background.paper', boxShadow: 1 }}
                  >
                    <Edit />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Elimina">
                  <IconButton
                    size="small"
                    color="error"
                    onClick={(e) => {
                      e.stopPropagation();
                      e.preventDefault();
                      deletePrompt(prompt.id);
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

      {/* Dialog anteprima */}
      <Dialog 
        open={previewOpen} 
        onClose={() => setPreviewOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Anteprima Prompt</DialogTitle>
        <DialogContent>
          <Box 
            sx={{ 
              backgroundColor: 'grey.50',
              p: 2,
              borderRadius: 1,
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              whiteSpace: 'pre-wrap',
              minHeight: 200
            }}
          >
            {previewContent}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Chiudi</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PromptManager;
