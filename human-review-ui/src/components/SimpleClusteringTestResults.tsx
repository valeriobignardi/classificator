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
  Alert,
  LinearProgress,
  Chip
} from '@mui/material';

interface SimpleClusteringTestResultsProps {
  open: boolean;
  onClose: () => void;
  result: any;
  isLoading: boolean;
}

/**
 * Componente semplificato per visualizzare i risultati del test clustering
 * Versione base senza Grid complesse per evitare conflitti MUI
 * 
 * Autore: Sistema di Classificazione
 * Data: 2025-08-25
 */
const SimpleClusteringTestResults: React.FC<SimpleClusteringTestResultsProps> = ({
  open,
  onClose,
  result,
  isLoading
}) => {
  if (!open) return null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          height: '80vh',
          maxHeight: '80vh',
        }
      }}
    >
      <DialogTitle>
        <Typography variant="h5" component="span">
          🧪 Risultati Test Clustering
        </Typography>
      </DialogTitle>

      <DialogContent dividers>
        {isLoading && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Test clustering in corso...
            </Typography>
            <LinearProgress />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Analisi parametri e generazione cluster...
            </Typography>
          </Box>
        )}

        {!isLoading && result && !result.success && (
          <Alert severity="error" sx={{ mb: 3 }}>
            <Typography variant="h6">❌ Test Fallito</Typography>
            <Typography variant="body2">
              {result.error || 'Errore sconosciuto durante il test clustering'}
            </Typography>
          </Alert>
        )}

        {!isLoading && result && result.success && (
          <>
            {/* Statistiche Generali */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📊 Statistiche Generali
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                  <Chip 
                    label={`⏱️ ${result.execution_time}s`}
                    color="primary"
                    sx={{ fontSize: '1rem', p: 2, height: 'auto' }}
                  />
                  <Chip 
                    label={`📊 ${result.sample_info.total_conversations} conversazioni`}
                    color="success"
                    sx={{ fontSize: '1rem', p: 2, height: 'auto' }}
                  />
                  <Chip 
                    label={`🔗 ${result.cluster_statistics.n_clusters} cluster`}
                    color="info"
                    sx={{ fontSize: '1rem', p: 2, height: 'auto' }}
                  />
                  <Chip 
                    label={`🔍 ${result.cluster_statistics.n_outliers} outliers`}
                    color="warning"
                    sx={{ fontSize: '1rem', p: 2, height: 'auto' }}
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary">
                  Outliers: {(result.quality_metrics.outlier_ratio * 100).toFixed(1)}% del totale
                </Typography>
              </CardContent>
            </Card>

            {/* Qualità */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  🎯 Qualità Clustering
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body1">
                    <strong>Valutazione:</strong> {result.quality_metrics.quality_assessment}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Silhouette Score:</strong> {result.quality_metrics.silhouette_score}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Bilanciamento:</strong> {result.quality_metrics.cluster_balance}
                  </Typography>
                </Box>

                <Alert severity="info">
                  <Typography variant="body2">
                    {result.outlier_analysis.recommendation}
                  </Typography>
                </Alert>
              </CardContent>
            </Card>

            {/* Cluster Trovati */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  📁 Cluster Trovati ({result.detailed_clusters.length})
                </Typography>
                
                {result.detailed_clusters.length === 0 ? (
                  <Alert severity="warning">
                    Nessun cluster trovato. I parametri potrebbero essere troppo restrittivi.
                  </Alert>
                ) : (
                  <Box>
                    {result.detailed_clusters.slice(0, 5).map((cluster: any) => (
                      <Card key={cluster.cluster_id} variant="outlined" sx={{ mb: 2 }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography variant="subtitle1">
                              {cluster.label}
                            </Typography>
                            <Chip 
                              label={`${cluster.size} conversazioni`}
                              size="small" 
                              color="primary"
                            />
                          </Box>
                          
                          <Typography variant="body2" color="text.secondary">
                            Cluster ID: {cluster.cluster_id}
                          </Typography>
                          
                          {/* Prime 2 conversazioni */}
                          {cluster.conversations.slice(0, 2).map((conv: any, idx: number) => (
                            <Box key={idx} sx={{ mt: 1, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                              <Typography variant="caption" color="text.secondary">
                                {conv.session_id} {conv.is_representative && '(Rappresentante)'}
                              </Typography>
                              <Typography variant="body2">
                                {conv.testo_completo.substring(0, 150)}...
                              </Typography>
                            </Box>
                          ))}
                          
                          {cluster.conversations.length > 2 && (
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                              ... e altre {cluster.conversations.length - 2} conversazioni
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                    
                    {result.detailed_clusters.length > 5 && (
                      <Alert severity="info">
                        Mostrati i primi 5 cluster. Totale: {result.detailed_clusters.length}
                      </Alert>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>

            {/* Parametri Utilizzati */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  ⚙️ Parametri Utilizzati
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(result.sample_info.parameters_used).map(([key, value]) => (
                    <Chip 
                      key={key}
                      label={`${key}: ${value}`}
                      variant="outlined"
                      size="small"
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="outlined">
          Chiudi
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SimpleClusteringTestResults;
