import React, { useEffect, useMemo, useState } from 'react';
import { Box, Typography, Accordion, AccordionSummary, AccordionDetails, Chip, Card, CardContent, Button, Tooltip, TextField } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Star as StarIcon, Link as LinkIcon, People as PeopleIcon } from '@mui/icons-material';
import { ReviewCase } from '../types/ReviewCase';

export interface ClusterGroupProps {
  clusterId: string;
  representatives: ReviewCase[];            // rappresentanti noti (es. pending in review)
  propagated?: ReviewCase[];                // opzionale: propagate note/pending
  extraRepresentatives?: ReviewCase[];      // altri rappresentanti (caricati da "tutte le sessioni")
  onLoadMoreReps?: (clusterId: string) => Promise<void> | void; // lazy loader
  onConfirmMajority?: (clusterId: string, selectedLabel: string) => Promise<void> | void; // azione batch
}

const getMajorityLabel = (items: ReviewCase[]): { label: string; ratio: number } => {
  const counts: Record<string, number> = {};
  let total = 0;
  for (const it of items) {
    const lbl = (it.classification || it.ml_prediction || it.llm_prediction || 'N/A').toUpperCase();
    if (!lbl) continue;
    counts[lbl] = (counts[lbl] || 0) + 1;
    total += 1;
  }
  if (total === 0) return { label: 'N/A', ratio: 0 };
  const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return { label: majority[0], ratio: majority[1] / total };
};

export const ClusterGroupAccordion: React.FC<ClusterGroupProps> = ({
  clusterId,
  representatives,
  propagated = [],
  extraRepresentatives = [],
  onLoadMoreReps,
  onConfirmMajority
}) => {
  const [expanded, setExpanded] = useState<boolean>(false);

  const repAll = [...representatives, ...extraRepresentatives];
  const majority = getMajorityLabel(repAll);
  const [selectedLabel, setSelectedLabel] = useState<string>(() =>
    majority.label && majority.label !== 'N/A' ? majority.label : ''
  );
  useEffect(() => {
    const defaultLabel = majority.label && majority.label !== 'N/A' ? majority.label : '';
    setSelectedLabel((current) => {
      if (!current.trim()) {
        return defaultLabel;
      }
      if (defaultLabel && current.trim().toUpperCase() === defaultLabel.toUpperCase()) {
        return defaultLabel;
      }
      return current;
    });
  }, [clusterId, majority.label]);
  const normalizedSelectedLabel = selectedLabel.trim().toUpperCase();
  const availableLabels = useMemo(() => {
    const unique = new Set<string>();
    repAll.forEach((rc) => {
      const lbl = (rc.classification || rc.ml_prediction || rc.llm_prediction || '').trim();
      if (lbl) {
        unique.add(lbl.toUpperCase());
      }
    });
    return Array.from(unique);
  }, [repAll]);
  const handleSelectLabel = (label: string) => {
    setSelectedLabel(label);
  };
  const handleCustomLabelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedLabel(event.target.value);
  };

  const handleChange = (_: any, isExpanded: boolean) => {
    setExpanded(isExpanded);
  };

  return (
    <Accordion expanded={expanded} onChange={handleChange} sx={{ border: '2px solid #e0e0e0', borderRadius: 2, '&:before': { display: 'none' } }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ backgroundColor: expanded ? '#e3f2fd' : '#f8f9fa', borderRadius: '8px 8px 0 0' }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
          <Box display="flex" alignItems="center" gap={2}>
            <StarIcon color="primary" />
            <Box>
              <Typography variant="h6" component="div">
                👑 Cluster {clusterId} - Rappresentanti
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Majority: {majority.label} ({Math.round(majority.ratio * 100)}%) • Reps: {repAll.length} • Propagati: {propagated.length}
              </Typography>
            </Box>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Chip icon={<PeopleIcon />} label={`${repAll.length} Rappresentanti`} color="primary" size="small" />
            {propagated.length > 0 && (
              <Chip icon={<LinkIcon />} label={`${propagated.length} Ereditate`} color="secondary" size="small" variant="outlined" />
            )}
          </Box>
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        {/* Rappresentanti */}
        <Box sx={{ p: 2, backgroundColor: '#f0f8ff', borderBottom: propagated.length ? '1px solid #e0e0e0' : 'none' }}>
          {repAll.length === 0 ? (
            <Typography variant="body2" color="text.secondary">Nessun rappresentante disponibile.</Typography>
          ) : (
            <Box display="flex" gap={2} flexWrap="wrap">
              {repAll.map((rc) => {
                const conversation = rc.conversation_text || '';
                const previewLimit = 160;
                const preview = conversation.slice(0, previewLimit);
                const hasMore = conversation.length > previewLimit;
                const label = (rc.classification || rc.ml_prediction || rc.llm_prediction || 'N/A').trim();
                const labelUpper = label.toUpperCase();
                const isSelected = normalizedSelectedLabel === labelUpper && normalizedSelectedLabel.length > 0;

                return (
                  <Tooltip
                    key={rc.case_id || rc.session_id}
                    title={
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-line', maxWidth: 480 }}>
                        {conversation || 'Conversazione non disponibile'}
                      </Typography>
                    }
                    placement="top-start"
                    arrow
                    enterDelay={400}
                  >
                    <Box sx={{ flex: '1 1 320px' }}>
                      <Card
                        sx={{
                          border: isSelected ? '2px solid #2e7d32' : '2px solid #1976d2',
                          boxShadow: isSelected ? '0 0 0 1px rgba(46, 125, 50, 0.3)' : undefined,
                          transition: 'border-color 0.2s ease'
                        }}
                      >
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1} gap={1}>
                            <Typography variant="subtitle2" color="primary" fontWeight="bold" sx={{ wordBreak: 'break-word' }}>
                              {label || 'N/A'}
                            </Typography>
                            <Chip
                              label={`Conf ${Math.round(((rc.ml_confidence ?? rc.llm_confidence ?? 0) as number) * 100)}%`}
                              size="small"
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {preview}
                            {hasMore ? '…' : ''}
                          </Typography>
                          <Box mt={2} display="flex" justifyContent="flex-end">
                            <Button
                              size="small"
                              variant={isSelected ? 'contained' : 'outlined'}
                              color={isSelected ? 'success' : 'primary'}
                              onClick={() => handleSelectLabel(labelUpper)}
                            >
                              Usa etichetta
                            </Button>
                          </Box>
                        </CardContent>
                      </Card>
                    </Box>
                  </Tooltip>
                );
              })}
            </Box>
          )}
          {onLoadMoreReps && (
            <Box mt={2}>
              <Tooltip title="Carica altri rappresentanti dal dataset delle sessioni">
                <span>
                  <Button variant="outlined" size="small" onClick={() => onLoadMoreReps(clusterId)}>Mostra altri rappresentanti</Button>
                </span>
              </Tooltip>
            </Box>
          )}
          {repAll.length > 0 && (
            <Box mt={3} display="flex" flexDirection="column" gap={1}>
              <Typography variant="subtitle2">Etichetta da applicare</Typography>
              {availableLabels.length > 0 && (
                <Box display="flex" gap={1} flexWrap="wrap">
                  {availableLabels.map((lbl) => {
                    const isCurrent = normalizedSelectedLabel === lbl;
                    return (
                      <Button
                        key={lbl}
                        size="small"
                        variant={isCurrent ? 'contained' : 'outlined'}
                        color={isCurrent ? 'success' : 'primary'}
                        onClick={() => handleSelectLabel(lbl)}
                      >
                        {lbl}
                      </Button>
                    );
                  })}
                </Box>
              )}
              <TextField
                label="Imposta nuova etichetta"
                size="small"
                value={selectedLabel}
                onChange={handleCustomLabelChange}
                placeholder="Digita una nuova etichetta"
                helperText="Scrivi un'etichetta personalizzata o seleziona un'opzione sopra"
              />
              {onConfirmMajority && (
                <Box>
                  <Tooltip
                    title={
                      normalizedSelectedLabel
                        ? `Applica l'etichetta ${normalizedSelectedLabel} ai rappresentanti pending`
                        : 'Inserisci o seleziona un\'etichetta valida'
                    }
                    arrow
                  >
                    <span>
                      <Button
                        color="success"
                        variant="contained"
                        size="small"
                        onClick={() => onConfirmMajority(clusterId, selectedLabel.trim())}
                        disabled={!normalizedSelectedLabel}
                      >
                        ✅ Applica etichetta
                        {normalizedSelectedLabel ? ` (${normalizedSelectedLabel})` : ''}
                      </Button>
                    </span>
                  </Tooltip>
                </Box>
              )}
            </Box>
          )}
        </Box>
        {/* Propagati */}
        {propagated.length > 0 && (
          <Box sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>🔗 Sessioni Ereditate</Typography>
            <Box display="flex" gap={2} flexWrap="wrap">
              {propagated.map((pc) => {
                const conversation = pc.conversation_text || '';
                const previewLimit = 160;
                const preview = conversation.slice(0, previewLimit);
                const hasMore = conversation.length > previewLimit;
                return (
                  <Tooltip
                    key={pc.case_id || pc.session_id}
                    title={
                      <Typography variant="body2" sx={{ whiteSpace: 'pre-line', maxWidth: 480 }}>
                        {conversation || 'Conversazione non disponibile'}
                      </Typography>
                    }
                    placement="top-start"
                    arrow
                    enterDelay={400}
                  >
                    <Box sx={{ flex: '1 1 320px' }}>
                      <Card sx={{ border: '1px dashed #ff9800', backgroundColor: '#fff8e1' }}>
                        <CardContent>
                          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1} gap={1}>
                            <Typography variant="subtitle2" color="secondary" fontWeight="bold" sx={{ wordBreak: 'break-word' }}>
                              {pc.classification || pc.ml_prediction || pc.llm_prediction || 'N/A'}
                            </Typography>
                            <Chip
                              label={`Conf ${Math.round(((pc.ml_confidence ?? pc.llm_confidence ?? 0) as number) * 100)}%`}
                              size="small"
                              color="secondary"
                              variant="outlined"
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {preview}
                            {hasMore ? '…' : ''}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Box>
                  </Tooltip>
                );
              })}
            </Box>
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  );
};

export default ClusterGroupAccordion;
