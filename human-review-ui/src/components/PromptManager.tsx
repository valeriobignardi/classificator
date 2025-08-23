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
   * Crea nuovo prompt
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
          engine: newPrompt.engine,
          prompt_type: newPrompt.prompt_type,
          prompt_name: newPrompt.prompt_name,
          content: newPrompt.content,
          variables: {},
          is_active: true
        }),
      });

      if (!response.ok) {
        throw new Error(`Errore creazione: ${response.statusText}`);
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
    if (!window.confirm('Sei sicuro di voler eliminare questo prompt?')) {
      return;
    }

    try {
      const response = await fetch(`/api/prompts/${promptId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Errore eliminazione: ${response.statusText}`);
      }

      await loadPrompts();
    } catch (err) {
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

            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                onClick={createPrompt}
                startIcon={<Save />}
              >
                Crea Prompt
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
              <Accordion key={prompt.id} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', pr: 2 }}>
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
                    
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Tooltip title="Anteprima">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            previewPrompt(prompt);
                          }}
                        >
                          <Visibility />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Modifica">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingPrompt(prompt);
                          }}
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
                            deletePrompt(prompt.id);
                          }}
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
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
                          startIcon={<Save />}
                        >
                          Salva
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
