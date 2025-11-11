import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Tooltip,
  Paper,
  Stack,
  IconButton
} from '@mui/material';
import {
  Star as StarIcon,
  Link as LinkIcon,
  OpenInNew as OpenInNewIcon
} from '@mui/icons-material';
import { ClusterContextCase, ClusterContextSummary } from '../types/ReviewCase';
import { formatLabelForDisplay } from '../utils/labelUtils';

interface ClusterCaseStripProps {
  cases: ClusterContextCase[];
  selectedSessionId: string;
  openedSessionId: string;
  onSelectCase: (sessionId: string) => void;
  summary: ClusterContextSummary | null;
}

const ClusterCaseStrip: React.FC<ClusterCaseStripProps> = ({
  cases,
  selectedSessionId,
  openedSessionId,
  onSelectCase,
  summary
}) => {
  if (!summary || cases.length === 0) {
    return null;
  }

  const representativesOnly = cases.filter((item) => item.is_representative);
  const casesToRender = representativesOnly.length > 0 ? representativesOnly : cases;
  const majorityLabelDisplay = summary.majority_label_display || summary.majority_label;

  return (
    <Box sx={{ mb: 3 }}>
      <Stack direction="row" alignItems="center" spacing={2} mb={1}>
        <Typography variant="h6">
          Altri casi nel cluster {summary.cluster_id}
        </Typography>
        {majorityLabelDisplay && (
          <Chip color="secondary" size="small" label={`Maggioranza: ${majorityLabelDisplay}`} />
        )}
        <Chip size="small" variant="outlined" label={`${summary.total_cases} totali`} />
      </Stack>

      <Box
        sx={{
          display: 'flex',
          overflowX: 'auto',
          py: 1,
          px: 0.5,
          gap: 1
        }}
      >
        {casesToRender.map((item) => {
          const isSelected = item.session_id === selectedSessionId;
          const isOpened = item.session_id === openedSessionId;
          const canOpen = Boolean(item.case_id);
          const displayLabel = item.display_label || formatLabelForDisplay(item.label);
          const tooltipContent = item.conversation_text ? (
            <Typography variant="body2" sx={{ whiteSpace: 'pre-line', maxWidth: 360 }}>
              {item.conversation_text}
            </Typography>
          ) : (
            'Conversazione non disponibile'
          );
          return (
            <Tooltip
              key={item.session_id}
              title={tooltipContent}
              placement="bottom-start"
              arrow
              enterDelay={400}
            >
              <Paper
                elevation={isSelected ? 6 : 1}
                sx={{
                  flex: '0 0 240px',
                  borderRadius: 2,
                  border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
                  backgroundColor: isSelected ? 'rgba(25, 118, 210, 0.08)' : 'white',
                  p: 1.5,
                  cursor: canOpen ? 'pointer' : 'default'
                }}
                onClick={() => canOpen && onSelectCase(item.session_id)}
              >
                <Stack direction="row" alignItems="center" spacing={1} mb={1}>
                  {item.is_representative ? (
                    <Tooltip title="Rappresentante del cluster">
                      <StarIcon color="primary" fontSize="small" />
                    </Tooltip>
                  ) : (
                    <Tooltip title="Propagazione">
                      <LinkIcon color="action" fontSize="small" />
                    </Tooltip>
                  )}
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {displayLabel}
                  </Typography>
                </Stack>

                <Stack direction="row" spacing={1} flexWrap="wrap" mb={1}>
                  {item.status && (
                    <Chip size="small" variant="outlined" label={item.status} />
                  )}
                  {item.confidence !== undefined && (
                    <Chip size="small" variant="outlined" label={`Conf ${(item.confidence * 100).toFixed(0)}%`} />
                  )}
                  {isOpened && (
                    <Chip size="small" color="primary" label="In revisione" />
                  )}
                </Stack>

                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                  Session {item.session_id.substring(0, 12)}…
                </Typography>

                <Stack direction="row" justifyContent="flex-end">
                  <Tooltip title={canOpen ? 'Apri caso' : 'Caso non presente nella coda di review'}>
                    <span>
                      <IconButton
                        size="small"
                        color={isSelected ? 'primary' : 'default'}
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectCase(item.session_id);
                        }}
                        disabled={!canOpen}
                      >
                        <OpenInNewIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Stack>
              </Paper>
            </Tooltip>
          );
        })}
      </Box>
    </Box>
  );
};

export default ClusterCaseStrip;
