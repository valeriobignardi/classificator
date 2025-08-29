/**
 * File: ClusteringStatisticsManager.tsx
 * Autore: Sistema di Classificazione
 * Data: 2025-08-26
 * Descrizione: Componente per visualizzazione statistiche avanzate con clustering + classificazioni
 * 
 * Storia aggiornamenti:
 * - 2025-08-26: Creazione componente per sezione STATISTICHE
 * - Integrazione con endpoint /api/statistics/{tenant}/clustering
 * - Visualizzazioni grafiche 2D/3D con dati storici + cluster labels
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  Alert,
  Chip,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  DateRange as DateRangeIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import ClusterVisualizationComponent from './ClusterVisualizationComponent';

interface StatisticsData {
  success: boolean;
  visualization_data: {
    points: Array<{
      x: number;
      y: number;
      z?: number;
      cluster_id: number;
      cluster_label: string;
      session_id: string;
      text_preview: string;
      classification: string;
      confidence: number;
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
  tenant_id: string;
  execution_time: number;
  error?: string;
}

/**
 * Componente per analisi statistiche avanzate clustering + classificazioni
 * 
 * Funzionalità:
 * - Visualizzazione dati storici con cluster labels
 * - Grafici interattivi 2D/3D con Plotly.js
 * - Analisi distribuzione cluster vs classificazioni finali
 * - Filtri per periodo temporale
 * - Metriche di qualità clustering su dati reali
 * 
 * Props: Nessuna (usa TenantContext)
 * 
 * Ultima modifica: 2025-08-26
 */
const ClusteringStatisticsManager: React.FC = () => {
  const { selectedTenant } = useTenant();
  const [statisticsData, setStatisticsData] = useState<StatisticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Stati per filtri temporali
  const [timeRange, setTimeRange] = useState<{ start_date: string; end_date: string } | null>(null);
  const [predefinedRange, setPredefinedRange] = useState<string>('last_30_days');

  /**
   * Carica statistiche clustering con visualizzazioni
   * 
   * Input: selectedTenant e timeRange opzionale
   * Output: Aggiorna statisticsData
   */
  const loadStatistics = useCallback(async () => {
    if (!selectedTenant?.tenant_id) return;

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getClusteringStatistics(
        selectedTenant.tenant_id,
        timeRange || undefined
      );

      if (response.success && response.visualization_data) {
        setStatisticsData(response);
        setSuccess(`Statistiche caricate: ${response.visualization_data.statistics.total_points} punti, ${response.visualization_data.statistics.n_clusters} cluster`);
      } else {
        setError(response.error || 'Errore caricamento statistiche');
      }
    } catch (err: any) {
      console.error('❌ [STATISTICS] Errore:', err);
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant?.tenant_id, timeRange]);

  /**
   * Aggiorna range temporale predefinito
   * 
   * Input: Tipo di range (last_7_days, last_30_days, etc.)
   * Output: Aggiorna timeRange
   */
  const updatePredefinedRange = (range: string) => {
    setPredefinedRange(range);
    
    const now = new Date();
    const start = new Date();
    
    switch (range) {
      case 'last_7_days':
        start.setDate(now.getDate() - 7);
        break;
      case 'last_30_days':
        start.setDate(now.getDate() - 30);
        break;
      case 'last_90_days':
        start.setDate(now.getDate() - 90);
        break;
      case 'all_time':
        setTimeRange(null);
        return;
      default:
        start.setDate(now.getDate() - 30);
    }
    
    setTimeRange({
      start_date: start.toISOString(),
      end_date: now.toISOString()
    });
  };

  /**
   * Renderizza metriche riassuntive
   */
  const renderSummaryMetrics = () => {
    if (!statisticsData?.visualization_data) return null;

    const stats = statisticsData.visualization_data.statistics;
    const points = statisticsData.visualization_data.points;
    
    // Calcola distribuzione classificazioni finali
    const classificationCounts = points.reduce((acc, point) => {
      acc[point.classification] = (acc[point.classification] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const topClassifications = Object.entries(classificationCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5);
    
    // Calcola metriche cluster
    const clusterSizes = points.reduce((acc, point) => {
      if (point.cluster_id !== -1) {  // Escludi outliers
        acc[point.cluster_id] = (acc[point.cluster_id] || 0) + 1;
      }
      return acc;
    }, {} as Record<number, number>);
    
    const avgClusterSize = Object.values(clusterSizes).length > 0 
      ? Object.values(clusterSizes).reduce((a, b) => a + b, 0) / Object.values(clusterSizes).length 
      : 0;
    
    return (
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <AnalyticsIcon sx={{ mr: 1 }} />
          Metriche Riassuntive
        </Typography>
        
        <Box display="flex" flexDirection="column" gap={2}>
          {/* Metriche Clustering */}
          <Box>
            <Typography variant="subtitle2" color="primary" gutterBottom>
              📊 Clustering
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
              <Chip label={`${stats.total_points} conversazioni`} variant="outlined" />
              <Chip label={`${stats.n_clusters} cluster`} color="primary" variant="outlined" />
              <Chip label={`${stats.n_outliers} outliers`} color="warning" variant="outlined" />
              <Chip label={`Dim. media: ${Math.round(avgClusterSize)}`} variant="outlined" />
            </Box>
          </Box>
          
          {/* Top Classificazioni */}
          <Box>
            <Typography variant="subtitle2" color="secondary" gutterBottom>
              🎯 Top Classificazioni
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1}>
              {topClassifications.map(([classification, count]) => (
                <Chip 
                  key={classification}
                  label={`${classification} (${count})`}
                  color="secondary"
                  variant="outlined"
                  size="small"
                />
              ))}
            </Box>
          </Box>
        </Box>
      </Paper>
    );
  };

  /**
   * Renderizza controlli filtri
   */
  const renderTimeFilters = () => (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box display="flex" flexWrap="wrap" gap={2} alignItems="center">
        <DateRangeIcon color="primary" />
        <Typography variant="subtitle2">Periodo Analisi:</Typography>
        
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel>Range</InputLabel>
          <Select
            value={predefinedRange}
            label="Range"
            onChange={(e) => updatePredefinedRange(e.target.value)}
          >
            <MenuItem value="last_7_days">Ultimi 7 giorni</MenuItem>
            <MenuItem value="last_30_days">Ultimi 30 giorni</MenuItem>
            <MenuItem value="last_90_days">Ultimi 90 giorni</MenuItem>
            <MenuItem value="all_time">Tutti i dati</MenuItem>
          </Select>
        </FormControl>
        
        {timeRange && (
          <>
            <Typography variant="caption" color="text.secondary">
              Dal: {new Date(timeRange.start_date).toLocaleDateString('it-IT')}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Al: {new Date(timeRange.end_date).toLocaleDateString('it-IT')}
            </Typography>
          </>
        )}
        
        <Button
          variant="outlined"
          size="small"
          startIcon={<RefreshIcon />}
          onClick={loadStatistics}
          disabled={loading}
        >
          Aggiorna
        </Button>
      </Box>
    </Paper>
  );

  // Carica dati al mount e quando cambia tenant
  useEffect(() => {
    if (selectedTenant?.tenant_id) {
      // Imposta range default solo al primo caricamento
      updatePredefinedRange('last_30_days');
    }
  }, [selectedTenant?.tenant_id]);

  // Carica statistiche quando cambia timeRange (ma non al primo mount)
  useEffect(() => {
    if (selectedTenant?.tenant_id && (timeRange !== null || predefinedRange === 'all_time')) {
      loadStatistics();
    }
  }, [selectedTenant?.tenant_id, timeRange, predefinedRange, loadStatistics]);

  // Auto-clear success/error messages
  useEffect(() => {
    if (success || error) {
      const timer = setTimeout(() => {
        setSuccess(null);
        setError(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [success, error]);

  if (!selectedTenant) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Seleziona un tenant per visualizzare le statistiche clustering
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center">
              <AssessmentIcon sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" component="h1">
                Statistiche Clustering Avanzate
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip 
                label={selectedTenant.tenant_name} 
                color="primary" 
                variant="outlined" 
              />
              <Chip
                label="Dati Storici + Cluster"
                color="secondary"
                variant="filled"
                size="small"
              />
            </Box>
          </Box>

          <Alert severity="info">
            <Typography variant="body2">
              <strong>Analisi Avanzata:</strong> Visualizza come i dati storici si distribuiscono nei cluster 
              e confronta con le classificazioni finali. Utile per validare la qualità del clustering su dati reali.
            </Typography>
          </Alert>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }}>
          {success}
        </Alert>
      )}

      {/* Controlli filtri */}
      {renderTimeFilters()}

      {loading && (
        <Box display="flex" alignItems="center" justifyContent="center" p={4}>
          <CircularProgress sx={{ mr: 2 }} />
          <Typography>Caricamento statistiche clustering...</Typography>
        </Box>
      )}

      {/* Metriche riassuntive */}
      {statisticsData && renderSummaryMetrics()}

      {/* Visualizzazione principale */}
      {statisticsData?.visualization_data && (
        <ClusterVisualizationComponent
          mode="statistics"
          tenantId={selectedTenant.tenant_id}
          visualizationData={statisticsData.visualization_data}
          onRefresh={loadStatistics}
          loading={loading}
          height={700}
        />
      )}

      {/* Stato vuoto */}
      {!loading && !statisticsData && !error && (
        <Alert severity="info">
          Nessun dato disponibile per il periodo selezionato. Prova ad espandere il range temporale.
        </Alert>
      )}
    </Box>
  );
};

export default ClusteringStatisticsManager;
