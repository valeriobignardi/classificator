import React, { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Tooltip,
  CircularProgress,
  Alert,
  Stack,
  Button,
  Collapse,
  IconButton
} from '@mui/material';
import {
  Group as GroupIcon,
  Star as StarIcon,
  Link as LinkIcon,
  CheckCircle as CheckCircleIcon,
  PendingActions as PendingActionsIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon
} from '@mui/icons-material';
import { ClusterContextCase, ClusterContextSummary } from '../types/ReviewCase';
import { formatLabelForDisplay } from '../utils/labelUtils';

interface ClusterContextPanelProps {
  summary: ClusterContextSummary | null;
  cases: ClusterContextCase[];
  loading: boolean;
  error?: string | null;
  selectedSessionId: string;
  openedSessionId: string;
  onSelectCase: (sessionId: string) => void;
  navigatingSessionId?: string | null;
}

const renderStatusChip = (item: ClusterContextCase) => {
  if (item.status === 'reviewed') {
    return (
      <Chip
        size="small"
        color="success"
        icon={<CheckCircleIcon fontSize="small" />}
        label="Review completata"
      />
    );
  }

  if (item.status === 'in_review_queue') {
    return (
      <Chip
        size="small"
        color="warning"
        icon={<PendingActionsIcon fontSize="small" />}
        label="In coda review"
      />
    );
  }

  return (
    <Chip
      size="small"
      icon={<GroupIcon fontSize="small" />}
      label="Disponibile"
    />
  );
};

const ClusterContextPanel: React.FC<ClusterContextPanelProps> = ({
  summary,
  cases,
  loading,
  error,
  selectedSessionId,
  openedSessionId,
  onSelectCase,
  navigatingSessionId = null
}) => {
  const PAGE_SIZE = 2;
  const [showPropagated, setShowPropagated] = useState(false);
  const [representativesPage, setRepresentativesPage] = useState(0);
  const [propagatedPage, setPropagatedPage] = useState(0);
  const majorityLabelDisplay = summary?.majority_label_display || summary?.majority_label;

  const representatives = useMemo(
    () => cases.filter((item) => item.is_representative),
    [cases]
  );
  const others = useMemo(
    () => cases.filter((item) => !item.is_representative),
    [cases]
  );

  useEffect(() => {
    setRepresentativesPage(0);
  }, [representatives.length]);

  useEffect(() => {
    setPropagatedPage(0);
  }, [others.length, showPropagated]);

  const representativesTotalPages = Math.ceil(representatives.length / PAGE_SIZE);
  const propagatedTotalPages = Math.ceil(others.length / PAGE_SIZE);

  const representativesSlice = useMemo(() => {
    const start = representativesPage * PAGE_SIZE;
    return representatives.slice(start, start + PAGE_SIZE);
  }, [representatives, representativesPage]);

  const propagatedSlice = useMemo(() => {
    const start = propagatedPage * PAGE_SIZE;
    return others.slice(start, start + PAGE_SIZE);
  }, [others, propagatedPage]);

  if (loading) {
    return (
      <Card sx={{ position: 'sticky', top: 96 }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 240 }}>
          <CircularProgress size={32} />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ position: 'sticky', top: 96 }}>
        {error}
      </Alert>
    );
  }

  if (!summary || cases.length === 0) {
    return (
      <Card sx={{ position: 'sticky', top: 96 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Nessun contesto disponibile
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Per questo cluster non sono disponibili altre sessioni non outlier. Verifica i dati di clustering.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const renderCaseEntry = (item: ClusterContextCase, index: number) => {
    const isSelected = item.session_id === selectedSessionId;
    const isOpened = item.session_id === openedSessionId;
    const isNavigating = navigatingSessionId === item.session_id;
    const displayLabel = item.display_label || formatLabelForDisplay(item.label);
    const tooltipContent = item.conversation_text ? (
      <Typography variant="body2" sx={{ whiteSpace: 'pre-line', maxWidth: 360 }}>
        {item.conversation_text}
      </Typography>
    ) : (
      'Conversazione non disponibile'
    );

    return (
    <ListItem disablePadding key={`${item.session_id}-${index}`}>
      <Tooltip
        title={tooltipContent}
        placement="left"
        arrow
        enterDelay={300}
      >
        <ListItemButton
          selected={isSelected}
          onClick={() => onSelectCase(item.session_id)}
          sx={{
            borderRadius: 1,
            border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
            mb: 1,
            opacity: isNavigating ? 0.6 : 1,
            alignItems: 'flex-start'
          }}
        >
          <ListItemText
            primary={
              <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
                {item.is_representative ? (
                  <Tooltip title="Rappresentante del cluster">
                    <StarIcon color="primary" fontSize="small" />
                  </Tooltip>
                ) : (
                  <Tooltip title="Propagazione dal rappresentante">
                    <LinkIcon color="action" fontSize="small" />
                  </Tooltip>
                )}
                <Typography fontWeight={600}>
                  {displayLabel}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {item.session_id.substring(0, 8)}…
                </Typography>
                {isOpened && (
                  <Chip size="small" color="primary" label="In revisione" />
                )}
              </Stack>
            }
            secondary={
              <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                {renderStatusChip(item)}
                {item.method && (
                  <Chip size="small" variant="outlined" label={item.method} />
                )}
                {item.confidence !== undefined && (
                  <Chip size="small" variant="outlined" label={`Conf ${(item.confidence * 100).toFixed(0)}%`} />
                )}
                {item.human_decision && (
                  <Chip size="small" color="info" label={`Decisione umana: ${item.human_decision}`} />
                )}
                {isNavigating && (
                  <CircularProgress size={14} />
                )}
              </Stack>
            }
          />
        </ListItemButton>
      </Tooltip>
    </ListItem>
  );
  };

  return (
    <Card sx={{ position: 'sticky', top: 96 }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <GroupIcon color="primary" />
          <Typography variant="h6">
            Cluster {summary.cluster_id}
          </Typography>
        </Stack>

        <Stack
          direction="row"
          spacing={1}
          flexWrap="wrap"
          sx={{ rowGap: 1 }}
        >
          <Chip label={`Totali: ${summary.total_cases}`} color="primary" size="small" />
          <Chip label={`Reviewate: ${summary.reviewed_cases}`} color="success" size="small" />
          <Chip label={`Pending: ${summary.pending_cases}`} color="warning" size="small" />
          <Chip label={`Rappresentanti: ${summary.representatives}`} size="small" variant="outlined" />
          <Chip label={`Propagate: ${summary.propagated}`} size="small" variant="outlined" />
          {majorityLabelDisplay && (
            <Chip
              color="secondary"
              size="small"
              label={`Maggioranza: ${majorityLabelDisplay}`}
            />
          )}
        </Stack>

        <Divider />

        <Box display="flex" flexDirection="column" gap={1.5}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="subtitle2">
              Rappresentanti
            </Typography>
            {representativesTotalPages > 1 && (
              <Stack direction="row" spacing={1} alignItems="center">
                <IconButton
                  size="small"
                  onClick={() => setRepresentativesPage((page) => Math.max(page - 1, 0))}
                  disabled={representativesPage === 0}
                  aria-label="Mostra rappresentanti precedenti"
                >
                  <ChevronLeftIcon fontSize="small" />
                </IconButton>
                <Typography variant="caption" color="text.secondary">
                  {representativesPage + 1} / {representativesTotalPages}
                </Typography>
                <IconButton
                  size="small"
                  onClick={() =>
                    setRepresentativesPage((page) =>
                      Math.min(page + 1, representativesTotalPages - 1)
                    )
                  }
                  disabled={representativesPage >= representativesTotalPages - 1}
                  aria-label="Mostra rappresentanti successivi"
                >
                  <ChevronRightIcon fontSize="small" />
                </IconButton>
              </Stack>
            )}
          </Box>
          {representatives.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              Nessun rappresentante trovato per questo cluster.
            </Typography>
          ) : (
            <List disablePadding>
              {representativesSlice.map((item, idx) =>
                renderCaseEntry(item, representativesPage * PAGE_SIZE + idx)
              )}
            </List>
          )}
        </Box>

        <Divider />

        <Box display="flex" flexDirection="column" gap={1.5}>
          {others.length > 0 ? (
            <>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Typography variant="subtitle2">
                  Propagazioni e altri casi
                </Typography>
                <Button size="small" onClick={() => setShowPropagated((prev) => !prev)}>
                  {showPropagated ? 'Nascondi' : `Mostra altri (${others.length})`}
                </Button>
              </Box>
              <Collapse in={showPropagated}>
                <>
                  {propagatedTotalPages > 1 && (
                    <Stack direction="row" spacing={1} alignItems="center" justifyContent="flex-end">
                      <IconButton
                        size="small"
                        onClick={() => setPropagatedPage((page) => Math.max(page - 1, 0))}
                        disabled={propagatedPage === 0}
                        aria-label="Mostra casi propagati precedenti"
                      >
                        <ChevronLeftIcon fontSize="small" />
                      </IconButton>
                      <Typography variant="caption" color="text.secondary">
                        {propagatedPage + 1} / {propagatedTotalPages}
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={() =>
                          setPropagatedPage((page) =>
                            Math.min(page + 1, propagatedTotalPages - 1)
                          )
                        }
                        disabled={propagatedPage >= propagatedTotalPages - 1}
                        aria-label="Mostra casi propagati successivi"
                      >
                        <ChevronRightIcon fontSize="small" />
                      </IconButton>
                    </Stack>
                  )}
                  <List disablePadding>
                    {propagatedSlice.map((item, idx) =>
                      renderCaseEntry(item, propagatedPage * PAGE_SIZE + idx)
                    )}
                  </List>
                </>
              </Collapse>
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Nessun altro caso nel cluster.
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ClusterContextPanel;
