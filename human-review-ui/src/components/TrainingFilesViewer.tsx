import React, { useEffect, useMemo, useState } from 'react';
import { Box, Typography, Select, MenuItem, FormControl, InputLabel, Paper, CircularProgress, Button, Stack, Alert, Switch, FormControlLabel } from '@mui/material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import { TrainingFileInfo } from '../types/TrainingFile';

const DEFAULT_LIMIT = 500;

const TrainingFilesViewer: React.FC = () => {
  const { selectedTenant } = useTenant();
  const [files, setFiles] = useState<TrainingFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [content, setContent] = useState<string>('');
  const [rawContent, setRawContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [limit, setLimit] = useState<number>(DEFAULT_LIMIT);
  const [truncated, setTruncated] = useState<boolean>(false);
  const [totalLines, setTotalLines] = useState<number>(0);
  const [formatEnabled, setFormatEnabled] = useState<boolean>(true);

  const tenantId = selectedTenant?.tenant_id;

  useEffect(() => {
    const loadFiles = async () => {
      if (!tenantId) return;
      setLoading(true);
      setError(null);
      setFiles([]);
      setSelectedFile('');
      setContent('');
      try {
        const res = await apiService.listTrainingFiles(tenantId);
        const list = res.files || [];
        setFiles(list);
        if (list.length === 1) {
          setSelectedFile(list[0].name);
        }
      } catch (e: any) {
        setError(e?.message || 'Errore nel caricamento dei file di training');
      } finally {
        setLoading(false);
      }
    };
    loadFiles();
  }, [tenantId]);

  const formatTrainingFileContent = (rawContent: string): string => {
    if (!rawContent) return rawContent;
    
    try {
      // Divide il contenuto in righe (ogni riga è un JSON)
      const lines = rawContent.split('\n').filter(line => line.trim());
      const formattedLines: string[] = [];
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        try {
          // Prova a parsare ogni riga come JSON
          const jsonObj = JSON.parse(line);
          
          // Formatta il JSON con indentazione
          let formattedJson = JSON.stringify(jsonObj, null, 2);
          
          // Converte i \n nei campi di testo in vere nuove righe
          formattedJson = formattedJson.replace(/\\n/g, '\n');
          
          // Aggiungi header per ogni record
          const recordHeader = `📝 Record ${i + 1}/${lines.length}`;
          formattedLines.push(`${recordHeader}\n${formattedJson}`);
        } catch {
          // Se non è JSON valido, mantieni la riga originale con header
          formattedLines.push(`📄 Riga ${i + 1}/${lines.length} (testo grezzo)\n${line}`);
        }
      }
      
      // Unisce le righe con separatore visivo
      return formattedLines.join('\n\n' + '='.repeat(80) + '\n\n');
    } catch {
      // Se c'è un errore generale, ritorna il contenuto originale
      return rawContent;
    }
  };

  const loadContent = async (fileName: string, customLimit?: number) => {
    if (!tenantId || !fileName) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.getTrainingFileContent(tenantId, fileName, customLimit ?? limit);
      const rawContentStr = res.content || '';
      const formattedContent = formatTrainingFileContent(rawContentStr);
      
      setRawContent(rawContentStr);
      setContent(formattedContent);
      setTruncated(res.truncated);
      setTotalLines(res.total_lines || 0);
      // Aggiorna limit effettivo (potrebbe essere diverso se passato)
      if (customLimit) setLimit(customLimit);
    } catch (e: any) {
      setError(e?.message || 'Errore nel caricamento del contenuto');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedFile) {
      loadContent(selectedFile, DEFAULT_LIMIT);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFile]);

  const handleLoadMore = () => {
    const next = Math.min(totalLines || limit * 2, (limit || DEFAULT_LIMIT) * 2);
    loadContent(selectedFile, next);
  };

  const header = useMemo(() => {
    if (!selectedTenant) return 'File di training';
    return `File di training per ${selectedTenant.tenant_name}`;
  }, [selectedTenant]);

  return (
    <Box display="flex" flexDirection="column" gap={2}>
      <Typography variant="h5" fontWeight={600}>{header}</Typography>
      {!selectedTenant && (
        <Alert severity="info">Seleziona un tenant per proseguire.</Alert>
      )}

      {error && <Alert severity="error">{error}</Alert>}

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
        <FormControl sx={{ minWidth: 260 }} size="small">
          <InputLabel id="training-file-select-label">Seleziona file</InputLabel>
          <Select
            labelId="training-file-select-label"
            label="Seleziona file"
            value={selectedFile}
            onChange={(e) => setSelectedFile(e.target.value)}
            disabled={!files.length}
          >
            {files.map((f) => (
              <MenuItem key={f.name} value={f.name}>
                {f.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControlLabel
          control={
            <Switch 
              checked={formatEnabled} 
              onChange={(e) => setFormatEnabled(e.target.checked)}
              size="small"
            />
          }
          label="Formatta JSON"
        />

        {files.length > 1 && (
          <Typography variant="body2" color="text.secondary">
            Trovati {files.length} file. Scegline uno per visualizzarlo.
          </Typography>
        )}
      </Stack>

      <Paper variant="outlined" sx={{ p: 2, bgcolor: '#0b1020', color: '#eaeefb', minHeight: 300 }}>
        {loading && <CircularProgress size={24} />}
        {!loading && !selectedFile && (
          <Typography variant="body2" color="#aab2d5">Nessun file selezionato.</Typography>
        )}
        {!loading && selectedFile && (
          <Box 
            component="pre" 
            sx={{ 
              m: 0, 
              whiteSpace: 'pre-wrap', 
              wordBreak: 'break-word', 
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', 
              maxHeight: 520, 
              overflow: 'auto',
              fontSize: '0.85rem',
              lineHeight: 1.4,
              '& .json-separator': {
                color: '#64748b',
                borderTop: '1px solid #334155',
                margin: '12px 0',
                paddingTop: '12px'
              }
            }}
          >
            {(formatEnabled ? content : rawContent) || '(File vuoto)'}
          </Box>
        )}
      </Paper>

      {selectedFile && !loading && truncated && (
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Visualizzate le prime {limit} righe su {totalLines}. 
          </Typography>
          <Button variant="outlined" onClick={handleLoadMore}>Carica più righe</Button>
        </Box>
      )}
    </Box>
  );
};

export default TrainingFilesViewer;

