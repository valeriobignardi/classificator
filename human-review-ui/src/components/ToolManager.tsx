/**
 * ToolManager.tsx - Gestione Tools/Funzioni LLM per tenant
 * Autore: Valerio Bignardi
 * Data: 2025-08-22
 * 
 * Componente React per la gestione dei tools/funzioni disponibili per il classificatore LLM
 * di uno specifico tenant. Permette di visualizzare, creare, modificare ed eliminare tools
 * che vengono utilizzati dal sistema di classificazione.
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Switch,
  FormControlLabel,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Tooltip,
  Paper
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  Code as CodeIcon,
  Functions as FunctionsIcon,
  Save as SaveIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';

interface Tool {
  id: number;
  tool_name: string;
  display_name: string;
  description: string;
  function_schema: any; // JSON schema per la funzione
  is_active: boolean;
  tenant_id: string;
  tenant_name: string;
  created_at: string;
  updated_at: string;
}

interface ToolManagerProps {
  open: boolean;
}

const ToolManager: React.FC<ToolManagerProps> = ({ open }) => {
  // Stato componente
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingTool, setEditingTool] = useState<Tool | null>(null);
  const [expandedAccordion, setExpandedAccordion] = useState<string | false>(false);
  
  // Form data per nuovo/modifica tool
  const [formData, setFormData] = useState({
    tool_name: '',
    display_name: '',
    description: '',
    function_schema: '',
    is_active: true
  });

  // Context tenant
  const { selectedTenant } = useTenant();

  /**
   * Carica tutti i tools per il tenant corrente
   */
  const loadTools = useCallback(async () => {
    if (!selectedTenant) {
      setTools([]);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/tools/tenant/${selectedTenant.tenant_id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Errore HTTP: ${response.status}`);
      }

      const data = await response.json();
      setTools(Array.isArray(data) ? data : []);

    } catch (err) {
      console.error('Errore caricamento tools:', err);
      setError(err instanceof Error ? err.message : 'Errore sconosciuto nel caricamento tools');
      setTools([]);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant]);

  /**
   * Carica tools quando il componente viene aperto o il tenant cambia
   */
  useEffect(() => {
    if (open && selectedTenant) {
      loadTools();
    }
  }, [open, selectedTenant, loadTools]);

  /**
   * Salva modifiche tool esistente
   */
  const saveTool = async (tool: Tool) => {
    console.log('💾 saveTool called with:', tool);
    
    try {
      console.log('🌐 Making PUT request to:', `/api/tools/${tool.id}`);
      
      const requestBody = {
        display_name: tool.display_name,
        description: tool.description,
        function_schema: tool.function_schema,
        is_active: tool.is_active
      };
      
      console.log('📤 Request body:', requestBody);

      const response = await fetch(`/api/tools/${tool.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('📡 Response status:', response.status);
      console.log('📡 Response ok:', response.ok);

      if (!response.ok) {
        const errorText = await response.text();
        console.log('❌ Response error text:', errorText);
        throw new Error(`Errore HTTP: ${response.status} - ${errorText}`);
      }

      const updatedTool = await response.json();
      console.log('✅ Updated tool received:', updatedTool);
      
      // Aggiorna lo stato locale
      setTools(prevTools => 
        prevTools.map(t => t.id === tool.id ? updatedTool : t)
      );

      setSuccess('Tool salvato con successo!');
      setTimeout(() => setSuccess(null), 3000);
      closeDialog();

    } catch (err) {
      console.error('❌ Errore salvataggio tool:', err);
      setError(err instanceof Error ? err.message : 'Errore nel salvataggio tool');
      setTimeout(() => setError(null), 5000);
    }
  };

  /**
   * Crea nuovo tool
   */
  const createTool = async () => {
    console.log('➕ createTool called');
    console.log('🏢 Selected tenant:', selectedTenant);
    
    if (!selectedTenant) {
      console.log('❌ No tenant selected!');
      setError('Nessun tenant selezionato');
      return;
    }

    try {
      // Validazione schema JSON
      let parsedSchema;
      try {
        parsedSchema = JSON.parse(formData.function_schema);
        console.log('✅ Parsed schema:', parsedSchema);
      } catch (e) {
        console.log('❌ JSON parsing error:', e);
        setError('Schema funzione deve essere un JSON valido');
        return;
      }

      const requestBody = {
        tool_name: formData.tool_name,
        display_name: formData.display_name,
        description: formData.description,
        function_schema: parsedSchema,
        is_active: formData.is_active,
        tenant_id: selectedTenant.tenant_id
      };
      
      console.log('📤 POST request body:', requestBody);
      console.log('🌐 Making POST request to: /api/tools');

      const response = await fetch('/api/tools', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('📡 Response status:', response.status);
      console.log('📡 Response ok:', response.ok);

      if (!response.ok) {
        const errorData = await response.text();
        console.log('❌ Response error:', errorData);
        throw new Error(`Errore HTTP: ${response.status} - ${errorData}`);
      }

      const newTool = await response.json();
      console.log('✅ New tool created:', newTool);
      
      setTools(prevTools => [...prevTools, newTool]);
      setSuccess('Tool creato con successo!');
      setTimeout(() => setSuccess(null), 3000);
      closeDialog();

    } catch (err) {
      console.error('❌ Errore creazione tool:', err);
      setError(err instanceof Error ? err.message : 'Errore nella creazione tool');
      setTimeout(() => setError(null), 5000);
    }
  };

  /**
   * Elimina tool
   */
  const deleteTool = async (toolId: number) => {
    if (!window.confirm('Sei sicuro di voler eliminare questo tool?')) {
      return;
    }

    try {
      const response = await fetch(`/api/tools/${toolId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Errore HTTP: ${response.status}`);
      }

      setTools(prevTools => prevTools.filter(t => t.id !== toolId));
      setSuccess('Tool eliminato con successo!');
      setTimeout(() => setSuccess(null), 3000);

    } catch (err) {
      console.error('Errore eliminazione tool:', err);
      setError(err instanceof Error ? err.message : 'Errore nell\'eliminazione tool');
      setTimeout(() => setError(null), 5000);
    }
  };

  /**
   * Apre dialog per nuovo tool
   */
  const openNewToolDialog = () => {
    setFormData({
      tool_name: '',
      display_name: '',
      description: '',
      function_schema: '{\n  "type": "function",\n  "function": {\n    "name": "",\n    "description": "",\n    "parameters": {\n      "type": "object",\n      "properties": {},\n      "required": []\n    }\n  }\n}',
      is_active: true
    });
    setEditingTool(null);
    setDialogOpen(true);
  };

  /**
   * Apre dialog per modifica tool
   */
  const openEditToolDialog = (tool: Tool) => {
    setFormData({
      tool_name: tool.tool_name,
      display_name: tool.display_name,
      description: tool.description,
      function_schema: JSON.stringify(tool.function_schema, null, 2),
      is_active: tool.is_active
    });
    setEditingTool(tool);
    setDialogOpen(true);
  };

  /**
   * Chiude dialog
   */
  const closeDialog = () => {
    setDialogOpen(false);
    setEditingTool(null);
    setFormData({
      tool_name: '',
      display_name: '',
      description: '',
      function_schema: '',
      is_active: true
    });
  };

  /**
   * Gestisce submit form
   */
  const handleSubmit = () => {
    console.log('🔥 handleSubmit called!');
    console.log('📋 Form data:', formData);
    console.log('✏️ Editing tool:', editingTool);
    
    // Validazione campi obbligatori
    if (!formData.tool_name || !formData.display_name || !formData.description) {
      console.log('❌ Validation failed - missing required fields');
      setError('Tutti i campi obbligatori devono essere compilati');
      return;
    }

    // Validazione JSON schema
    try {
      JSON.parse(formData.function_schema);
      console.log('✅ JSON schema validation passed');
    } catch (e) {
      console.log('❌ JSON schema validation failed:', e);
      setError('Schema funzione deve essere un JSON valido');
      return;
    }

    if (editingTool) {
      console.log('📝 Updating existing tool...');
      // Modifica tool esistente
      const updatedTool = {
        ...editingTool,
        display_name: formData.display_name,
        description: formData.description,
        function_schema: JSON.parse(formData.function_schema),
        is_active: formData.is_active
      };
      console.log('🔄 Calling saveTool with:', updatedTool);
      saveTool(updatedTool);
    } else {
      console.log('➕ Creating new tool...');
      // Crea nuovo tool
      createTool();
    }
  };

  /**
   * Gestisce accordion expand
   */
  const handleAccordionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedAccordion(isExpanded ? panel : false);
  };

  if (!open) {
    return null;
  }

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" component="h2" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <FunctionsIcon sx={{ mr: 1 }} />
          Gestione Tools LLM
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Configura le funzioni disponibili per il classificatore LLM di {selectedTenant?.tenant_name || 'questo tenant'}
        </Typography>
      </Box>

      {/* Messages */}
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

      {/* Loading */}
      {loading && (
        <Box display="flex" justifyContent="center" py={4}>
          <CircularProgress />
        </Box>
      )}

      {/* No tenant selected */}
      {!selectedTenant && (
        <Alert severity="info">
          Seleziona un tenant per gestire i suoi tools
        </Alert>
      )}

      {/* Tools list */}
      {selectedTenant && !loading && (
        <>
          {/* Add new tool button */}
          <Box sx={{ mb: 3 }}>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={openNewToolDialog}
              color="primary"
            >
              Nuovo Tool
            </Button>
          </Box>

          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: 2 }}>
            {tools.map((tool) => (
              <Box key={tool.id}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    {/* Header con nome e stato */}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                      <Typography variant="h6" component="h3">
                        {tool.display_name}
                      </Typography>
                      <Chip
                        label={tool.is_active ? 'Attivo' : 'Inattivo'}
                        color={tool.is_active ? 'success' : 'default'}
                        size="small"
                      />
                    </Box>

                    {/* Tool name tecnico */}
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontFamily: 'monospace' }}>
                      {tool.tool_name}
                    </Typography>

                    {/* Descrizione */}
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      {tool.description}
                    </Typography>

                    {/* Schema function (accordion) */}
                    <Accordion 
                      expanded={expandedAccordion === `tool-${tool.id}`}
                      onChange={handleAccordionChange(`tool-${tool.id}`)}
                      sx={{ boxShadow: 'none', '&:before': { display: 'none' } }}
                    >
                      <AccordionSummary 
                        expandIcon={<ExpandMoreIcon />}
                        sx={{ px: 0, minHeight: 36 }}
                      >
                        <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
                          <CodeIcon sx={{ mr: 1, fontSize: 16 }} />
                          Schema Funzione
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails sx={{ px: 0, py: 1 }}>
                        <Paper sx={{ p: 1, backgroundColor: 'grey.50' }}>
                          <Typography 
                            variant="body2" 
                            component="pre" 
                            sx={{ 
                              fontFamily: 'monospace', 
                              fontSize: '0.75rem',
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-all'
                            }}
                          >
                            {JSON.stringify(tool.function_schema, null, 2)}
                          </Typography>
                        </Paper>
                      </AccordionDetails>
                    </Accordion>

                    {/* Metadata */}
                    <Box sx={{ mt: 2 }}>
                      <Divider sx={{ mb: 1 }} />
                      <Typography variant="caption" color="text.secondary" display="block">
                        Creato: {new Date(tool.created_at).toLocaleString('it-IT')}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Modificato: {new Date(tool.updated_at).toLocaleString('it-IT')}
                      </Typography>
                    </Box>
                  </CardContent>

                  <CardActions>
                    <Tooltip title="Modifica tool">
                      <IconButton onClick={() => openEditToolDialog(tool)} size="small">
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Elimina tool">
                      <IconButton onClick={() => deleteTool(tool.id)} size="small" color="error">
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </CardActions>
                </Card>
              </Box>
            ))}
          </Box>

          {/* Empty state */}
          {tools.length === 0 && (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <FunctionsIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Nessun tool configurato
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Crea il primo tool per abilitare funzioni personalizzate nel classificatore LLM
              </Typography>
              <Button variant="outlined" startIcon={<AddIcon />} onClick={openNewToolDialog}>
                Aggiungi Tool
              </Button>
            </Paper>
          )}
        </>
      )}

      {/* Dialog per nuovo/modifica tool */}
      <Dialog open={dialogOpen} onClose={closeDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingTool ? 'Modifica Tool' : 'Nuovo Tool'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
                <TextField
                  fullWidth
                  label="Nome Tecnico Tool"
                  value={formData.tool_name}
                  onChange={(e) => setFormData(prev => ({ ...prev, tool_name: e.target.value }))}
                  disabled={!!editingTool} // Non modificabile in edit
                  required
                  helperText="Nome univoco per la funzione (es: search_patient_info)"
                />
                <TextField
                  fullWidth
                  label="Nome Visualizzato"
                  value={formData.display_name}
                  onChange={(e) => setFormData(prev => ({ ...prev, display_name: e.target.value }))}
                  required
                  helperText="Nome leggibile per l'interfaccia"
                />
              </Box>
              <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label="Descrizione"
                  value={formData.description}
                  onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  required
                  helperText="Descrizione dettagliata del tool e del suo utilizzo"
                />
              <TextField
                fullWidth
                multiline
                rows={12}
                label="Schema Funzione (JSON)"
                value={formData.function_schema}
                onChange={(e) => setFormData(prev => ({ ...prev, function_schema: e.target.value }))}
                required
                helperText="Schema OpenAI Function Calling in formato JSON"
                sx={{ '& .MuiInputBase-input': { fontFamily: 'monospace', fontSize: '0.875rem' } }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.is_active}
                    onChange={(e) => setFormData(prev => ({ ...prev, is_active: e.target.checked }))}
                  />
                }
                label="Tool Attivo"
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={closeDialog} startIcon={<CancelIcon />}>
            Annulla
          </Button>
          <Button onClick={handleSubmit} variant="contained" startIcon={<SaveIcon />}>
            SALVA
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ToolManager;
