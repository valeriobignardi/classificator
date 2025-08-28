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
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Close as CloseIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import ClusterVisualizationComponent from './ClusterVisualizationComponent';

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
    davies_bouldin_score: number;
    calinski_harabasz_score: number;
    outlier_ratio: number;
    cluster_balance: string;
    quality_assessment: string;
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
  // 🆕 Dati visualizzazione dal backend
  visualization_data?: {
    points: Array<{
      x: number;
      y: number;
      z?: number;
      cluster_id: number;
      cluster_label: string;
      session_id: string;
      text_preview: string;
    }>;
    cluster_colors: Record<number, string>;
    statistics: {
      total_points: number;
      n_clusters: number;
      n_outliers: number;
      dimensions: number;
    };
    coordinates: {
      tsne_2d: Array<[number, number]>;
      pca_2d: Array<[number, number]>;
      pca_3d: Array<[number, number, number]>;
    };
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
  const [activeTab, setActiveTab] = React.useState(0);
  
  // Hook sempre chiamati in modo consistente
  React.useEffect(() => {
    if (open && result) {
      console.log('🔍 [CLUSTERING DIALOG] Dialog opened with result');
    }
  }, [open, result]);
  
  // Early return DOPO tutti gli hook
  if (!open || !result) {
    return null;
  }

  const renderSuccessResult = () => {
    // Verifica se abbiamo le statistiche
    if (!result.statistics) {
      return <Alert severity="warning">Nessuna statistica disponibile</Alert>;
    }

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
              <Typography variant="h6" gutterBottom>🎯 Metriche di Qualità Clustering</Typography>
              
              {/* Silhouette Score */}
              <Box sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Chip 
                    label={`Silhouette: ${
                      result.quality_metrics.silhouette_score !== undefined && result.quality_metrics.silhouette_score !== null 
                        ? result.quality_metrics.silhouette_score.toFixed(3) 
                        : 'N/A'
                    }`} 
                    color={
                      result.quality_metrics.silhouette_score !== undefined && result.quality_metrics.silhouette_score !== null
                        ? (result.quality_metrics.silhouette_score > 0.5 ? 'success' : result.quality_metrics.silhouette_score > 0.25 ? 'warning' : 'error')
                        : 'default'
                    }
                    size="medium" 
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  <strong>Silhouette Score:</strong> Misura quanto ogni punto è simile al suo cluster rispetto agli altri cluster. 
                  Valori: -1 (pessimo) a +1 (ottimo). Soglia consigliata: ≥0.5 per clustering di alta qualità.
                </Typography>
              </Box>

              {/* Davies-Bouldin Index */}
              <Box sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Chip 
                    label={`Davies-Bouldin: ${
                      result.quality_metrics.davies_bouldin_score !== undefined && result.quality_metrics.davies_bouldin_score !== null 
                        ? result.quality_metrics.davies_bouldin_score.toFixed(3) 
                        : 'N/A'
                    }`} 
                    color={
                      result.quality_metrics.davies_bouldin_score !== undefined && result.quality_metrics.davies_bouldin_score !== null
                        ? (result.quality_metrics.davies_bouldin_score < 1 ? 'success' : result.quality_metrics.davies_bouldin_score < 2 ? 'warning' : 'error')
                        : 'default'
                    }
                    size="medium" 
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  <strong>Davies-Bouldin Index:</strong> Misura la compattezza interna dei cluster e la separazione tra cluster. 
                  Valori più bassi sono migliori (0 = perfetto). Soglia consigliata: &lt;1.0 per clustering eccellente.
                </Typography>
              </Box>

              {/* Calinski-Harabasz Index */}
              <Box sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Chip 
                    label={`Calinski-Harabasz: ${
                      result.quality_metrics.calinski_harabasz_score !== undefined && result.quality_metrics.calinski_harabasz_score !== null
                        ? result.quality_metrics.calinski_harabasz_score.toFixed(1) 
                        : 'N/A'
                    }`} 
                    color={
                      result.quality_metrics.calinski_harabasz_score !== undefined && result.quality_metrics.calinski_harabasz_score !== null
                        ? (result.quality_metrics.calinski_harabasz_score > 100 ? 'success' : result.quality_metrics.calinski_harabasz_score > 50 ? 'warning' : 'error')
                        : 'default'
                    }
                    size="medium" 
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  <strong>Calinski-Harabasz Index:</strong> Rapporto tra dispersione tra-cluster e intra-cluster. 
                  Valori più alti indicano cluster ben definiti e separati. Soglia consigliata: ≥100 per buona separazione.
                </Typography>
              </Box>

              {/* Riepilogo qualitativo */}
              {result.quality_metrics.quality_assessment && (
                <Box sx={{ mt: 2, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                  <Typography variant="body2" fontWeight="bold">
                    📋 Valutazione Complessiva: {result.quality_metrics.quality_assessment}
                  </Typography>
                </Box>
              )}
              
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
        {/* Tabs per organizzare informazioni */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs 
            value={activeTab} 
            onChange={(_, newValue) => setActiveTab(newValue)}
            aria-label="Risultati clustering tabs"
          >
            <Tab 
              label="📊 Statistiche" 
              icon={<AssessmentIcon />}
              iconPosition="start"
            />
            <Tab 
              label="🎨 Visualizzazione" 
              icon={<VisibilityIcon />}
              iconPosition="start"
              disabled={!result.visualization_data}
            />
          </Tabs>
        </Box>

        {/* Tab Statistiche */}
        {activeTab === 0 && (
          result.success ? (
            renderSuccessResult()
          ) : (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="h6" gutterBottom>❌ Errore</Typography>
              <Typography>{result.error || 'Errore sconosciuto durante il test clustering'}</Typography>
            </Alert>
          )
        )}

        {/* Tab Visualizzazione */}
        {activeTab === 1 && result.visualization_data && (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Visualizzazione Interattiva:</strong> Usa i controlli per esplorare i cluster in 2D/3D. 
                Hover sui punti per dettagli conversazioni.
              </Typography>
            </Alert>
            
            <ClusterVisualizationComponent
              mode="parameters"
              visualizationData={result.visualization_data}
              height={600}
            />
          </Box>
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
