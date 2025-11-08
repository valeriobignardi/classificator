/**
 * File: ClusterVisualizationComponent.tsx
 * Autore: Sistema di Classificazione
 * Data: 2025-08-26
 * Descrizione: Componente per visualizzazioni grafiche 2D/3D interattive dei cluster
 * 
 * Storia aggiornamenti:
 * - 2025-08-26: Creazione componente con supporto t-SNE, PCA 2D/3D
 * - Visualizzazioni interattive con Plotly.js
 * - Modalità dual: "Parameters" (solo clustering) e "Statistics" (con classificazioni finali)
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Alert,
  Chip,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Slider,
  Paper
} from '@mui/material';
import {
  ScatterPlot as ScatterPlotIcon,
  ThreeDRotation as ViewIn3DIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

interface VisualizationData {
  points: Array<{
    x: number;
    y: number;
    z?: number;
    cluster_id: number;
    cluster_label: string;
    label_text?: string;
    session_id: string;
    text_preview: string;
    classification?: string;  // Solo in modalità Statistics
    confidence?: number;      // Solo in modalità Statistics
  }>;
  cluster_colors: Record<number, string>;
  statistics: {
    total_points: number;
    n_clusters: number;
    n_outliers: number;
    dimensions: number;
  };
  coordinates: {
    tsne_2d?: Array<[number, number]>;
    pca_2d?: Array<[number, number]>;
    pca_3d?: Array<[number, number, number]>;
  };
}

interface ClusterVisualizationProps {
  mode: 'parameters' | 'statistics';  // Dual mode support
  tenantId?: string;
  visualizationData?: VisualizationData;
  onRefresh?: () => void;
  loading?: boolean;
  height?: number;
}

/**
 * Componente principale per visualizzazioni cluster interattive 2D/3D
 * 
 * Modalità:
 * - "parameters": Solo dati clustering (per test parametri)
 * - "statistics": Cluster + classificazioni finali (per analisi statistiche)
 * 
 * Supporta:
 * - t-SNE 2D per riduzione dimensionalità
 * - PCA 2D/3D per analisi componenti principali
 * - Visualizzazione interattiva con hover e zoom
 * - Filtri per cluster e outliers
 * - Colorazione automatica per cluster
 * 
 * Ultima modifica: 2025-08-26
 */
const ClusterVisualizationComponent: React.FC<ClusterVisualizationProps> = ({
  mode = 'parameters',
  tenantId,
  visualizationData,
  onRefresh,
  loading = false,
  height = 500
}) => {
  // Stati per controlli visualizzazione
  const [visualizationType, setVisualizationType] = useState<'tsne_2d' | 'pca_2d' | 'pca_3d'>('tsne_2d');
  const [showOutliers, setShowOutliers] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [selectedClusters, setSelectedClusters] = useState<number[]>([]);
  const [pointSize, setPointSize] = useState(8);

  /**
   * Effetto per inizializzare cluster selezionati
   * 
   * Input: visualizationData
   * Output: Aggiorna selectedClusters con tutti i cluster
   */
  useEffect(() => {
    if (visualizationData?.points) {
      const uniqueClusters = Array.from(
        new Set(visualizationData.points.map(p => p.cluster_id))
      );
      setSelectedClusters(uniqueClusters);
    }
  }, [visualizationData]);

  /**
   * Filtra i punti in base ai controlli utente
   * 
   * Input: Tutti i punti e filtri attivi
   * Output: Punti filtrati per visualizzazione
   */
  const getFilteredPoints = () => {
    if (!visualizationData?.points) return [];

    return visualizationData.points.filter(point => {
      // Filtro cluster selezionati
      if (!selectedClusters.includes(point.cluster_id)) return false;
      
      // Filtro outliers (cluster_id = -1)
      if (!showOutliers && point.cluster_id === -1) return false;
      
      return true;
    });
  };

  /**
   * Prepara i dati per Plotly in base al tipo di visualizzazione
   * 
   * Input: Tipo visualizzazione e punti filtrati
   * Output: Array di trace per Plotly
   */
  const getPlotlyData = () => {
    const filteredPoints = getFilteredPoints();
    if (filteredPoints.length === 0) return [];

    // Raggruppa per cluster per colorazione
    const clusterGroups = filteredPoints.reduce((acc, point) => {
      if (!acc[point.cluster_id]) {
        acc[point.cluster_id] = [];
      }
      acc[point.cluster_id].push(point);
      return acc;
    }, {} as Record<number, typeof filteredPoints>);

    // Crea una trace per ogni cluster
    return Object.entries(clusterGroups).map(([clusterId, points]) => {
      const clusterIdNum = parseInt(clusterId);
      const isOutlier = clusterIdNum === -1;
      const clusterLabel = points[0]?.cluster_label || `Cluster ${clusterId}`;
      const pointLabelTexts = points.map(p => p.label_text || clusterLabel);
      
      // Determina coordinate in base al tipo visualizzazione
      let coordinates: any = {};
      
      switch (visualizationType) {
        case 'tsne_2d':
          coordinates = {
            x: points.map(p => p.x),
            y: points.map(p => p.y),
            // Mostra etichette opzionali in 2D
            mode: showLabels ? 'markers+text' : 'markers',
            text: showLabels ? pointLabelTexts : undefined,
            textposition: showLabels ? 'top center' : undefined,
            type: 'scatter'
          };
          break;
          
        case 'pca_2d':
          coordinates = {
            x: points.map(p => p.x),
            y: points.map(p => p.y),
            // Mostra etichette opzionali in 2D
            mode: showLabels ? 'markers+text' : 'markers',
            text: showLabels ? pointLabelTexts : undefined,
            textposition: showLabels ? 'top center' : undefined,
            type: 'scatter'
          };
          break;
          
        case 'pca_3d':
          coordinates = {
            x: points.map(p => p.x),
            y: points.map(p => p.y),
            z: points.map(p => p.z || 0),
            // In 3D, Plotly supporta text ma non textposition
            mode: showLabels ? 'markers+text' : 'markers',
            text: showLabels ? pointLabelTexts : undefined,
            type: 'scatter3d'
          };
          break;
      }

      // Tooltip personalizzato
      const hoverText = points.map(p => {
        let text = `<b>Cluster:</b> ${p.cluster_label}<br>`;
        text += `<b>Session:</b> ${p.session_id}<br>`;
        text += `<b>Preview:</b> ${p.text_preview.substring(0, 100)}...`;
        
        // Se in modalità Statistics, aggiungi info classificazione
        if (mode === 'statistics' && p.classification) {
          text += `<br><b>Classificazione:</b> ${p.classification}`;
          text += `<br><b>Confidenza:</b> ${(p.confidence || 0).toFixed(2)}`;
        }
        
        return text;
      });

      return {
        ...coordinates,
        name: isOutlier ? `Outliers (${points.length})` : `${clusterLabel} (${points.length})`,
        // Usa customdata per i tooltip così 'text' resta libero per le etichette visuali
        customdata: hoverText,
        hovertemplate: '%{customdata}<extra></extra>',
        marker: {
          size: pointSize,
          color: visualizationData?.cluster_colors[clusterIdNum] || '#999999',
          opacity: isOutlier ? 0.4 : 0.8,
          line: {
            width: 1,
            color: 'white'
          }
        },
        showlegend: true,
        legendgroup: clusterId
      };
    });
  };

  /**
   * Configurazione layout Plotly
   * 
   * Input: Tipo visualizzazione
   * Output: Oggetto layout per Plotly
   */
  const getPlotlyLayout = () => {
    const baseLayout = {
      title: {
        text: `${mode === 'statistics' ? 'Analisi Statistiche' : 'Parametri Clustering'} - ${visualizationType.replace('_', ' ').toUpperCase()}`,
        font: { size: 16 }
      },
      height: height,
      margin: { l: 50, r: 50, t: 50, b: 50 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      hovermode: 'closest' as const
    };

    if (visualizationType === 'pca_3d') {
      return {
        ...baseLayout,
        scene: {
          xaxis: { title: { text: 'PC1' } },
          yaxis: { title: { text: 'PC2' } },
          zaxis: { title: { text: 'PC3' } },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 }
          }
        }
      };
    } else {
      return {
        ...baseLayout,
        xaxis: { 
          title: { text: visualizationType.includes('tsne') ? 't-SNE 1' : 'PC1' },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.1)'
        },
        yaxis: { 
          title: { text: visualizationType.includes('tsne') ? 't-SNE 2' : 'PC2' },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.1)'
        }
      };
    }
  };

  // Memo dei dati Plotly per usarli anche negli handler eventi legenda
  const plotlyData = useMemo(() => getPlotlyData(), [visualizationData, selectedClusters, showOutliers, showLabels, visualizationType, pointSize]);

  // Calcola lista completa dei cluster disponibili
  const allClusterIds = useMemo(() => {
    if (!visualizationData?.points) return [] as number[];
    return Array.from(new Set(visualizationData.points.map(p => p.cluster_id)));
  }, [visualizationData]);

  // Handler: click sulla legenda per isolare un cluster
  const handleLegendClick = (e: any) => {
    try {
      const curveNumber = e?.curveNumber;
      if (curveNumber === undefined || !plotlyData[curveNumber]) return false;
      const legendGroup = plotlyData[curveNumber]?.legendgroup;
      const clusterId = parseInt(String(legendGroup));
      if (Number.isNaN(clusterId)) return false;

      setSelectedClusters(prev => {
        // Se già isolato, ripristina tutti
        if (prev.length === 1 && prev[0] === clusterId) {
          return allClusterIds;
        }
        return [clusterId];
      });
      // Previeni il toggle di default di Plotly
      return false;
    } catch {
      return false;
    }
  };

  // Handler: doppio click sulla legenda per ripristinare tutti i cluster
  const handleLegendDoubleClick = (_e: any) => {
    setSelectedClusters(allClusterIds);
    return false;
  };

  /**
   * Toggle selezione cluster
   * 
   * Input: ID cluster da toggle
   * Output: Aggiorna selectedClusters
   */
  const toggleCluster = (clusterId: number) => {
    if (selectedClusters.includes(clusterId)) {
      setSelectedClusters(prev => prev.filter(id => id !== clusterId));
    } else {
      setSelectedClusters(prev => [...prev, clusterId]);
    }
  };

  /**
   * Renderizza controlli visualizzazione
   */
  const renderControls = () => {
    if (!visualizationData) return null;

    const availableClusters = Array.from(
      new Set(visualizationData.points.map(p => p.cluster_id))
    ).sort((a, b) => a - b);

    return (
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box display="flex" flexWrap="wrap" gap={2} alignItems="center">
          {/* Tipo visualizzazione */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Tipo</InputLabel>
            <Select
              value={visualizationType}
              label="Tipo"
              onChange={(e) => setVisualizationType(e.target.value as any)}
            >
              <MenuItem value="tsne_2d">t-SNE 2D</MenuItem>
              <MenuItem value="pca_2d">PCA 2D</MenuItem>
              <MenuItem value="pca_3d">PCA 3D</MenuItem>
            </Select>
          </FormControl>

          {/* Switch outliers */}
          <FormControlLabel
            control={
              <Switch
                checked={showOutliers}
                onChange={(e) => setShowOutliers(e.target.checked)}
                size="small"
              />
            }
            label="Outliers"
          />

          {/* Switch etichette */}
          <FormControlLabel
            control={
              <Switch
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                size="small"
              />
            }
            label="Etichette"
          />

          {/* Dimensione punti */}
          <Box sx={{ width: 120 }}>
            <Typography variant="caption">Dim. Punti</Typography>
            <Slider
              value={pointSize}
              onChange={(_, value) => setPointSize(value as number)}
              min={4}
              max={15}
              size="small"
            />
          </Box>

          {/* Refresh */}
          {onRefresh && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshIcon />}
              onClick={onRefresh}
              disabled={loading}
            >
              Refresh
            </Button>
          )}
        </Box>

        {/* Filtri cluster */}
        <Box mt={2}>
          <Typography variant="subtitle2" mb={1}>Cluster Visibili:</Typography>
          <Box 
            display="flex" 
            flexWrap="nowrap" 
            gap={1}
            sx={{ 
              overflowX: 'auto',
              overflowY: 'hidden',
              paddingBottom: 1,
              '&::-webkit-scrollbar': {
                height: '6px',
              },
              '&::-webkit-scrollbar-track': {
                background: '#f1f1f1',
                borderRadius: '3px',
              },
              '&::-webkit-scrollbar-thumb': {
                background: '#c1c1c1',
                borderRadius: '3px',
              },
              '&::-webkit-scrollbar-thumb:hover': {
                background: '#a8a8a8',
              }
            }}
          >
            {availableClusters.map(clusterId => {
              const isSelected = selectedClusters.includes(clusterId);
              const isOutlier = clusterId === -1;
              const clusterPoints = visualizationData.points.filter(p => p.cluster_id === clusterId);
              const clusterLabel = clusterPoints[0]?.cluster_label || `Cluster ${clusterId}`;
              
              return (
                <Chip
                  key={clusterId}
                  label={isOutlier ? `Outliers (${clusterPoints.length})` : `${clusterLabel} (${clusterPoints.length})`}
                  onClick={() => toggleCluster(clusterId)}
                  variant={isSelected ? "filled" : "outlined"}
                  color={isOutlier ? "default" : "primary"}
                  size="small"
                  sx={{
                    backgroundColor: isSelected 
                      ? visualizationData.cluster_colors[clusterId] || '#999'
                      : 'transparent',
                    flexShrink: 0,
                    minWidth: 'auto',
                    '&:hover': {
                      opacity: 0.8
                    }
                  }}
                />
              );
            })}
          </Box>
        </Box>
      </Paper>
    );
  };

  // Loading state
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="center" p={4}>
            <CircularProgress sx={{ mr: 2 }} />
            <Typography>Generazione visualizzazioni...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (!visualizationData) {
    return (
      <Alert severity="info">
        Nessun dato di visualizzazione disponibile. {onRefresh && 'Prova a fare refresh.'}
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header con statistiche */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center">
              {visualizationType === 'pca_3d' ? <ViewIn3DIcon sx={{ mr: 1 }} /> : <ScatterPlotIcon sx={{ mr: 1 }} />}
              <Typography variant="h6">
                Visualizzazione {mode === 'statistics' ? 'Statistiche' : 'Clustering'}
              </Typography>
            </Box>
            <Box display="flex" gap={1}>
              <Chip label={`${visualizationData.statistics?.total_points || 0} punti`} size="small" />
              <Chip label={`${visualizationData.statistics?.n_clusters || 0} cluster`} size="small" color="primary" />
              <Chip label={`${visualizationData.statistics?.n_outliers || 0} outliers`} size="small" color="warning" />
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Controlli */}
      {renderControls()}

      {/* Grafico principale */}
      <Card>
        <CardContent>
          <Plot
            data={plotlyData}
            layout={getPlotlyLayout()}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
              toImageButtonOptions: {
                format: 'png',
                filename: `cluster_visualization_${mode}_${visualizationType}`,
                height: height,
                width: 800,
                scale: 1
              }
            }}
            onLegendClick={handleLegendClick}
            onLegendDoubleClick={handleLegendDoubleClick}
            style={{ width: '100%' }}
          />
        </CardContent>
      </Card>
    </Box>
  );
};

export default ClusterVisualizationComponent;
