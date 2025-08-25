/**
 * Componente per la visualizzazione dei risultati del test clustering HDBSCAN
 * 
 * Funzionalità:
 * - Visualizza statistiche cluster (numero cluster, outliers, qualità)
 * - Mostra cluster dettagliati con conversazioni rappresentative
 * - Analizza outliers con campioni
 * - Fornisce metriche di qualità e raccomandazioni
 * 
 * Autore: Sistema di Classificazione
 * Data creazione: 2025-08-25
 * Ultima modifica: 2025-08-25 - Aggiornamento interfaccia per compatibilità backend
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  Chip,
  Alert
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Close as CloseIcon
} from '@mui/icons-material';

interface ClusteringTestResult {
  success: boolean;
  error?: string;
  execution_time?: number;
  statistics?: {
    total_conversations: number;
    n_clusters: number;
    n_outliers: number;
    clustering_ratio: number;
  };
  quality_metrics?: {
    silhouette_score: number;
    calinski_harabasz_score: number;
    davies_bouldin_score: number;
  };
  recommendations?: string[];
  detailed_clusters?: {
    clusters: Array<{
      cluster_id: number;
      size: number;
      conversations: Array<{
        session_id: string;
        text: string;
        text_length: number;
      }>;
    }>;
  };
  outlier_analysis?: {
    count: number;
    percentage: number;
    samples: Array<{
      session_id: string;
      text: string;
      text_length: number;
    }>;
  };
}

interface ClusteringTestResultsProps {
  open: boolean;
  onClose: () => void;
  result: ClusteringTestResult | null;
  isLoading: boolean;
}

/**
 * Componente principale per visualizzare i risultati del test clustering
 */
const ClusteringTestResults: React.FC<ClusteringTestResultsProps> = ({
  open,
  onClose,
  result,
  isLoading
}) => {
  if (!result) {
    return null;
  }

  const renderSuccessResult = () => {
    if (!result.statistics) return null;

    return (
      <Box>
        {/* Statistiche principali */}
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>📊 Risultati Clustering</Typography>
            <Box display="flex" flexWrap="wrap" gap={2} mb={2}>
              <Chip 
                label={`${result.statistics.total_conversations} conversazioni`} 
                color="info" 
                variant="outlined" 
              />
              <Chip 
                label={`${result.statistics.n_clusters} cluster`} 
                color="success" 
                variant="outlined" 
              />
              <Chip 
                label={`${result.statistics.n_outliers} outliers (${Math.round((result.statistics.n_outliers / result.statistics.total_conversations) * 100)}%)`} 
                color="warning" 
                variant="outlined" 
              />
              <Chip 
                label={`${result.execution_time?.toFixed(2)}s esecuzione`} 
                color="primary" 
                variant="outlined" 
              />
            </Box>
          </CardContent>
        </Card>

        {/* Metriche qualità */}
        {result.quality_metrics && (
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>🎯 Qualità Clustering</Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                <Chip 
                  label={`Silhouette: ${result.quality_metrics.silhouette_score?.toFixed(3) || 'N/A'}`} 
                  size="small" 
                  variant="outlined" 
                />
                <Chip 
                  label={`Calinski-Harabasz: ${result.quality_metrics.calinski_harabasz_score?.toFixed(1) || 'N/A'}`} 
                  size="small" 
                  variant="outlined" 
                />
                <Chip 
                  label={`Davies-Bouldin: ${result.quality_metrics.davies_bouldin_score?.toFixed(3) || 'N/A'}`} 
                  size="small" 
                  variant="outlined" 
                />
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Raccomandazioni */}
        {result.recommendations && result.recommendations.length > 0 && (
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>💡 Raccomandazioni</Typography>
              {result.recommendations.map((rec, index) => (
                <Alert 
                  key={index} 
                  severity={rec.includes('✅') ? 'success' : rec.includes('⚠️') ? 'warning' : 'info'}
                  sx={{ mb: 1 }}
                >
                  {rec}
                </Alert>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Esempi cluster principali */}
        {result.detailed_clusters && result.detailed_clusters.clusters.length > 0 && (
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>🔍 Cluster Principali (primi 5)</Typography>
              {result.detailed_clusters.clusters.slice(0, 5).map((cluster, index) => (
                <Box key={cluster.cluster_id} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Cluster {cluster.cluster_id} ({cluster.size} conversazioni)
                  </Typography>
                  {cluster.conversations?.slice(0, 2).map((conv, convIndex) => (
                    <Typography key={convIndex} variant="body2" sx={{ mb: 1, fontStyle: 'italic' }}>
                      "{conv.text}"
                    </Typography>
                  ))}
                  {cluster.size > 2 && (
                    <Typography variant="caption" color="text.secondary">
                      ...e altre {cluster.size - 2} conversazioni simili
                    </Typography>
                  )}
                </Box>
              ))}
            </CardContent>
          </Card>
        )}
      </Box>
    );
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box display="flex" alignItems="center" gap={1}>
          {result.success ? (
            <CheckCircleIcon color="success" />
          ) : (
            <ErrorIcon color="error" />
          )}
          <Typography variant="h6">
            {result.success ? 'Test Clustering Completato' : 'Errore Test Clustering'}
          </Typography>
        </Box>
        <Button onClick={onClose} color="inherit">
          <CloseIcon />
        </Button>
      </DialogTitle>

      <DialogContent dividers>
        {result.success ? (
          renderSuccessResult()
        ) : (
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>❌ Errore</Typography>
            <Typography>{result.error || 'Errore sconosciuto durante il test clustering'}</Typography>
          </Alert>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="contained">
          Chiudi
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ClusteringTestResults;
