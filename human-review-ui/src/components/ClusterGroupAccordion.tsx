import React, { useState } from 'react';
import { Box, Typography, Accordion, AccordionSummary, AccordionDetails, Chip, Card, CardContent, Button, Tooltip } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Star as StarIcon, Link as LinkIcon, People as PeopleIcon } from '@mui/icons-material';
import { ReviewCase } from '../types/ReviewCase';

export interface ClusterGroupProps {
  clusterId: string;
  representatives: ReviewCase[];            // rappresentanti noti (es. pending in review)
  propagated?: ReviewCase[];                // opzionale: propagate note/pending
  extraRepresentatives?: ReviewCase[];      // altri rappresentanti (caricati da "tutte le sessioni")
  onLoadMoreReps?: (clusterId: string) => Promise<void> | void; // lazy loader
  onConfirmMajority?: (clusterId: string) => Promise<void> | void; // azione batch
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
              {repAll.map((rc) => (
                <Card key={rc.case_id || rc.session_id} sx={{ flex: '1 1 320px', border: '2px solid #1976d2' }}>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle2" color="primary" fontWeight="bold">
                        {rc.classification || rc.ml_prediction || rc.llm_prediction || 'N/A'}
                      </Typography>
                      <Chip label={`Conf ${(rc.ml_confidence || rc.llm_confidence || 0) * 100 | 0}%`} size="small" />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {(rc.conversation_text || '').slice(0, 160)}{(rc.conversation_text || '').length > 160 ? '…' : ''}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
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
          {onConfirmMajority && repAll.length > 0 && (
            <Box mt={1}>
              <Tooltip title={`Applica etichetta di maggioranza (${majority.label}) ai rappresentanti pending di questo cluster`}>
                <span>
                  <Button color="success" variant="contained" size="small" onClick={() => onConfirmMajority(clusterId)}>
                    ✅ Conferma maggioranza ({majority.label})
                  </Button>
                </span>
              </Tooltip>
            </Box>
          )}
        </Box>
        {/* Propagati */}
        {propagated.length > 0 && (
          <Box sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>🔗 Sessioni Ereditate</Typography>
            <Box display="flex" gap={2} flexWrap="wrap">
              {propagated.map((pc) => (
                <Card key={pc.case_id || pc.session_id} sx={{ flex: '1 1 320px', border: '1px dashed #ff9800', backgroundColor: '#fff8e1' }}>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle2" color="secondary" fontWeight="bold">
                        {pc.classification || pc.ml_prediction || pc.llm_prediction || 'N/A'}
                      </Typography>
                      <Chip label={`Conf ${(pc.ml_confidence || pc.llm_confidence || 0) * 100 | 0}%`} size="small" color="secondary" variant="outlined" />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {(pc.conversation_text || '').slice(0, 160)}{(pc.conversation_text || '').length > 160 ? '…' : ''}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  );
};

export default ClusterGroupAccordion;
