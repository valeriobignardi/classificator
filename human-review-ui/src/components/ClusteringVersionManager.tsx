/*
 * ClusteringVersionManager.tsx
 * 
 * Componente per la gestione delle versioni dei risultati di clustering
 * 
 * Funzionalità:
 * - Visualizzazione cronologia risultati clustering salvati
 * - Selezione e visualizzazione di versioni specifiche
 * - Confronto tra due versioni
 * - Trend analysis delle metriche nel tempo
 * 
 * Autore: Sistema di Classificazione AI
 * Data: 26/12/2024
 * Ultima modifica: 26/12/2024
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  Button,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider
} from '@mui/material';
import {
  History as HistoryIcon,
  Compare as CompareIcon,
  TrendingUp as TrendIcon,
  Visibility as ViewIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';
import ClusterVisualizationComponent from './ClusterVisualizationComponent';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';

interface ClusteringHistoryItem {
  id: number;
  version_number: number;
  created_at: string;
  n_clusters: number;
  n_outliers: number;
  silhouette_score: number;
  davies_bouldin_score?: number;
  calinski_harabasz_score?: number;
  execution_time: number;
  parameters_summary: string;
}

interface ClusteringVersionDetail {
  id: number;
  version_number: number;
  tenant_id: string;
  created_at: string;
  results_data: any;  // Cambiato da results_json a results_data
  parameters_data: any;  // Cambiato da parameters_json a parameters_data
  n_clusters: number;
  n_outliers: number;
  silhouette_score: number;
  davies_bouldin_score?: number;
  calinski_harabasz_score?: number;
  execution_time: number;
}

interface ClusteringComparison {
  version1: ClusteringVersionDetail;
  version2: ClusteringVersionDetail;
  comparison: {
    metrics_delta: {
      n_clusters: number;
      n_outliers: number;
      silhouette_score: number;
      davies_bouldin_score: number;
      calinski_harabasz_score: number;
      execution_time: number;
    };
    parameters_diff: Array<{
      parameter: string;
      value1: any;
      value2: any;
      changed: boolean;
    }>;
    quality_assessment: string;
  };
}

interface TrendData {
  trend_data: Array<{
    version_number: number;
    created_at: string;
    n_clusters: number;
    n_outliers: number;
    silhouette_score: number;
    execution_time: number;
    clustering_ratio: number;
    n_conversations: number;
  }>;
  metrics_summary: {
    avg_clusters: number;
    avg_outliers: number;
    avg_silhouette: number;
    best_silhouette: number;
    best_version: number;
    total_versions: number;
  };
  has_data: boolean;
}

/**
 * Componente principale per la gestione versioni clustering
 */
const ClusteringVersionManager: React.FC = () => {
  const { selectedTenant } = useTenant();
  
  // Stati principali
  const [activeTab, setActiveTab] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Stati per cronologia
  const [history, setHistory] = useState<ClusteringHistoryItem[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState<number | null>(null);
  const [selectedVersionDetail, setSelectedVersionDetail] = useState<ClusteringVersionDetail | null>(null);
  
  // Stati per confronto
  const [compareVersion1, setCompareVersion1] = useState<number | null>(null);
  const [compareVersion2, setCompareVersion2] = useState<number | null>(null);
  const [comparison, setComparison] = useState<ClusteringComparison | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  
  // Stati per trend
  const [trendData, setTrendData] = useState<TrendData | null>(null);
  const [trendDays, setTrendDays] = useState(30);
  
  // Stati per dialog
  const [viewDialogOpen, setViewDialogOpen] = useState(false);

  /**
   * Carica la cronologia dei risultati clustering
   */
  const loadHistory = useCallback(async () => {
    if (!selectedTenant?.tenant_id) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getClusteringHistory(selectedTenant.tenant_id, 100);
      if (response.success && response.data) {
        setHistory(response.data);
        
        // Seleziona automaticamente la versione più recente se non ne è selezionata una
        if (response.data.length > 0 && !selectedVersionId) {
          setSelectedVersionId(response.data[0].version_number);  // Usa version_number invece di id
        }
      } else {
        setError(response.error || 'Errore caricamento cronologia clustering');
      }
    } catch (err: any) {
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant?.tenant_id]); // Aggiungiamo la dipendenza corretta

  /**
   * Carica i dettagli di una versione specifica
   */
  const loadVersionDetail = useCallback(async (versionNumber: number) => {
    if (!selectedTenant?.tenant_id) return;
    
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getClusteringVersion(selectedTenant.tenant_id, versionNumber);
      if (response.success && response.data) {
        setSelectedVersionDetail(response.data);
      } else {
        setError(response.error || 'Errore caricamento dettagli versione');
      }
    } catch (err: any) {
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant?.tenant_id]);

  /**
   * Confronta due versioni
   */
  const compareVersions = async () => {
    if (!compareVersion1 || !compareVersion2) return;

    setCompareLoading(true);
    setError(null);

    try {
      const response = await apiService.compareClusteringVersions(
        selectedTenant?.tenant_id || '', 
        compareVersion1, 
        compareVersion2
      );
      
      if (response.success) {
        // Mappiamo la risposta del backend alla struttura attesa dal componente
        const comparisonData: ClusteringComparison = {
          version1: {
            id: response.version1.number,
            version_number: response.version1.number,
            tenant_id: response.tenant_id,
            created_at: response.version1.metadata.created_at,
            results_data: response.version1.data,
            parameters_data: response.version1.parameters,
            n_clusters: response.version1.data.statistics?.n_clusters || 0,
            n_outliers: response.version1.data.statistics?.n_outliers || 0,
            silhouette_score: response.version1.data.quality_metrics?.silhouette_score || 0,
            davies_bouldin_score: response.version1.data.quality_metrics?.davies_bouldin_score,
            calinski_harabasz_score: response.version1.data.quality_metrics?.calinski_harabasz_score,
            execution_time: response.version1.metadata.execution_time
          },
          version2: {
            id: response.version2.number,
            version_number: response.version2.number,
            tenant_id: response.tenant_id,
            created_at: response.version2.metadata.created_at,
            results_data: response.version2.data,
            parameters_data: response.version2.parameters,
            n_clusters: response.version2.data.statistics?.n_clusters || 0,
            n_outliers: response.version2.data.statistics?.n_outliers || 0,
            silhouette_score: response.version2.data.quality_metrics?.silhouette_score || 0,
            davies_bouldin_score: response.version2.data.quality_metrics?.davies_bouldin_score,
            calinski_harabasz_score: response.version2.data.quality_metrics?.calinski_harabasz_score,
            execution_time: response.version2.metadata.execution_time
          },
          comparison: {
            metrics_delta: {
              n_clusters: response.comparison_metrics.clusters_diff,
              n_outliers: response.comparison_metrics.outliers_diff,
              silhouette_score: response.comparison_metrics.silhouette_diff,
              davies_bouldin_score: (response.version2.data.quality_metrics?.davies_bouldin_score || 0) - (response.version1.data.quality_metrics?.davies_bouldin_score || 0),
              calinski_harabasz_score: (response.version2.data.quality_metrics?.calinski_harabasz_score || 0) - (response.version1.data.quality_metrics?.calinski_harabasz_score || 0),
              execution_time: response.comparison_metrics.execution_time_diff
            },
            parameters_diff: [], // TODO: Implementare se necessario
            quality_assessment: response.comparison_metrics.silhouette_diff > 0 
              ? "✅ Miglioramento qualità clustering" 
              : response.comparison_metrics.silhouette_diff < 0 
              ? "⚠️ Peggioramento qualità clustering"
              : "➡️ Qualità clustering stabile"
          }
        };
        setComparison(comparisonData);
      } else {
        setError(response.error || 'Errore confronto versioni');
      }
    } catch (err: any) {
      console.error('❌ Errore confronto versioni:', err);
      setError(`Errore: ${err.message}`);
    } finally {
      setCompareLoading(false);
    }
  };

  /**
   * Carica i dati del trend
   */
  const loadTrendData = useCallback(async () => {
    if (!selectedTenant?.tenant_id) return;

    setLoading(true);
    setError(null);

    try {
      console.log('🔍 [DEBUG] Chiamata API getClusteringMetricsTrend per tenant:', selectedTenant.tenant_id, 'giorni:', trendDays);
      const response = await apiService.getClusteringMetricsTrend(selectedTenant.tenant_id, trendDays);
      
      console.log('📊 [DEBUG] Risposta API trend:', response);
      
      if (response.success && response.data) {
        console.log('✅ [DEBUG] Dati trend ricevuti:', response.data);
        console.log('📈 [DEBUG] Trend data array:', response.data.trend_data?.length || 0, 'elementi');
        console.log('📋 [DEBUG] Metrics summary:', response.data.metrics_summary);
        setTrendData(response.data);
      } else {
        console.error('❌ [DEBUG] Errore risposta API:', response.error);
        setError(response.error || 'Errore caricamento trend');
      }
    } catch (err: any) {
      console.error('❌ [DEBUG] Eccezione chiamata API:', err);
      setError(`Errore: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant?.tenant_id, trendDays]);

  /**
   * Visualizza una versione nel dialog
   */
  const viewVersion = (versionNumber: number) => {
    loadVersionDetail(versionNumber);
    setViewDialogOpen(true);
  };

  /**
   * Formatta la data in formato leggibile
   */
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('it-IT');
  };

  /**
   * Ottiene il colore per il chip del silhouette score
   */
  const getSilhouetteColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 0.5) return 'success';
    if (score >= 0.2) return 'warning';
    return 'error';
  };

  /**
   * Ottiene il colore per il chip del Davies-Bouldin Index
   * Più basso = migliore (0 = perfetto, <1 = buono, >2 = problematico)
   */
  const getDaviesBouldinColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score <= 1.0) return 'success';
    if (score <= 2.0) return 'warning';
    return 'error';
  };

  /**
   * Ottiene il colore per il chip del Calinski-Harabasz Index
   * Più alto = migliore (>300 = ottimo, >100 = buono, <100 = problematico)
   */
  const getCalinskiHarabaszColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 300) return 'success';
    if (score >= 100) return 'warning';
    return 'error';
  };

  // Effect per caricare la cronologia quando cambia il tenant
  useEffect(() => {
    loadHistory();
  }, [selectedTenant]); // ✅ Dipende solo dal tenant

  // Effect per caricare i dettagli quando cambia la versione selezionata
  useEffect(() => {
    if (selectedVersionId) {
      loadVersionDetail(selectedVersionId);
    }
  }, [selectedVersionId, loadVersionDetail]); // ✅ Includiamo loadVersionDetail

  // Effect per caricare il trend quando cambia tab
  useEffect(() => {
    if (activeTab === 2) {
      loadTrendData();
    }
  }, [activeTab, loadTrendData]);

  if (!selectedTenant) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Seleziona un tenant per visualizzare la cronologia clustering
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Card>
        <CardContent>
          {/* Header */}
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center">
              <HistoryIcon sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="h5" component="h1">
                Cronologia Clustering
              </Typography>
            </Box>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip 
                label={selectedTenant.nome} 
                color="primary" 
                variant="outlined" 
              />
              <Chip
                icon={<HistoryIcon />}
                label={`${history.length} versioni`}
                color="info"
                size="small"
              />
            </Box>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {loading && <LinearProgress sx={{ mb: 2 }} />}

          {/* Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
              <Tab 
                label="📋 Cronologia" 
                icon={<HistoryIcon />}
                iconPosition="start"
              />
              <Tab 
                label="🔍 Confronto" 
                icon={<CompareIcon />}
                iconPosition="start"
              />
              <Tab 
                label="📈 Trend" 
                icon={<TrendIcon />}
                iconPosition="start"
              />
            </Tabs>
          </Box>

          {/* Tab Cronologia */}
          {activeTab === 0 && (
            <Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                {/* Selezione versione */}
                <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
                  <FormControl fullWidth>
                    <InputLabel>Seleziona Versione</InputLabel>
                    <Select
                      value={selectedVersionId || ''}
                      onChange={(e) => setSelectedVersionId(Number(e.target.value))}
                      label="Seleziona Versione"
                    >
                      {history.map((item) => (
                        <MenuItem key={item.version_number} value={item.version_number}>
                          V{item.version_number} - {formatDate(item.created_at)}
                          <Chip
                            label={`${item.n_clusters} cluster`}
                            size="small"
                            sx={{ ml: 1 }}
                          />
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                {/* Azioni */}
                <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
                  <Box display="flex" gap={1} height="100%" alignItems="center">
                    <Button
                      variant="outlined"
                      startIcon={<ViewIcon />}
                      onClick={() => selectedVersionId && viewVersion(selectedVersionId)}
                      disabled={!selectedVersionId}
                    >
                      Visualizza
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={loadHistory}
                      disabled={loading}
                    >
                      Aggiorna
                    </Button>
                  </Box>
                </Box>
              </Box>

              {/* Tabella cronologia */}
              <TableContainer component={Paper} sx={{ mt: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Versione</TableCell>
                      <TableCell>Data</TableCell>
                      <TableCell align="center">Cluster</TableCell>
                      <TableCell align="center">Outliers</TableCell>
                      <TableCell align="center">
                        <Tooltip title="Coerenza interna cluster (-1 a 1, più alto = migliore)">
                          <span>Silhouette</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title="Separazione cluster (0+, più basso = migliore)">
                          <span>Davies-Bouldin</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title="Densità e separazione (0+, più alto = migliore)">
                          <span>Calinski-Harabasz</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="center">Tempo</TableCell>
                      <TableCell align="center">Azioni</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {history.map((item) => (
                      <TableRow 
                        key={item.version_number}
                        selected={selectedVersionId === item.version_number}
                        sx={{ cursor: 'pointer' }}
                        onClick={() => setSelectedVersionId(item.version_number)}
                      >
                        <TableCell>
                          <Chip label={`V${item.version_number}`} size="small" />
                        </TableCell>
                        <TableCell>{formatDate(item.created_at)}</TableCell>
                        <TableCell align="center">{item.n_clusters}</TableCell>
                        <TableCell align="center">{item.n_outliers}</TableCell>
                        <TableCell align="center">
                          <Chip
                            label={item.silhouette_score.toFixed(3)}
                            color={getSilhouetteColor(item.silhouette_score)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={item.davies_bouldin_score?.toFixed(3) || 'N/A'}
                            color={item.davies_bouldin_score ? getDaviesBouldinColor(item.davies_bouldin_score) : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={item.calinski_harabasz_score?.toFixed(1) || 'N/A'}
                            color={item.calinski_harabasz_score ? getCalinskiHarabaszColor(item.calinski_harabasz_score) : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="center">{item.execution_time.toFixed(1)}s</TableCell>
                        <TableCell align="center">
                          <Tooltip title="Visualizza dettagli">
                            <IconButton onClick={(e) => { e.stopPropagation(); viewVersion(item.version_number); }}>
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}

          {/* Tab Confronto */}
          {activeTab === 1 && (
            <Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <FormControl fullWidth>
                    <InputLabel>Prima Versione</InputLabel>
                    <Select
                      value={compareVersion1 || ''}
                      onChange={(e) => setCompareVersion1(Number(e.target.value))}
                      label="Prima Versione"
                    >
                      {history.map((item) => (
                        <MenuItem key={item.version_number} value={item.version_number}>
                          V{item.version_number} - {formatDate(item.created_at)}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <FormControl fullWidth>
                    <InputLabel>Seconda Versione</InputLabel>
                    <Select
                      value={compareVersion2 || ''}
                      onChange={(e) => setCompareVersion2(Number(e.target.value))}
                      label="Seconda Versione"
                    >
                      {history.map((item) => (
                        <MenuItem key={item.version_number} value={item.version_number}>
                          V{item.version_number} - {formatDate(item.created_at)}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>

                <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                  <Button
                    variant="contained"
                    startIcon={<CompareIcon />}
                    onClick={compareVersions}
                    disabled={!compareVersion1 || !compareVersion2 || compareLoading}
                    fullWidth
                    sx={{ height: '56px' }}
                  >
                    {compareLoading ? 'Confrontando...' : 'Confronta'}
                  </Button>
                </Box>
              </Box>

              {/* Risultati confronto */}
              {comparison && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    📊 Risultati Confronto
                  </Typography>
                  
                  {/* Metriche Delta */}
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {comparison.comparison.metrics_delta.n_clusters > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.n_clusters}
                          </Typography>
                          <Typography variant="body2">Cluster</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="secondary">
                            {comparison.comparison.metrics_delta.n_outliers > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.n_outliers}
                          </Typography>
                          <Typography variant="body2">Outliers</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="success.main">
                            {comparison.comparison.metrics_delta.silhouette_score > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.silhouette_score.toFixed(3)}
                          </Typography>
                          <Typography variant="body2">Silhouette</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography 
                            variant="h6" 
                            color={comparison.comparison.metrics_delta.davies_bouldin_score < 0 ? "success.main" : comparison.comparison.metrics_delta.davies_bouldin_score > 0 ? "error.main" : "text.primary"}
                          >
                            {comparison.comparison.metrics_delta.davies_bouldin_score > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.davies_bouldin_score.toFixed(3)}
                          </Typography>
                          <Typography variant="body2">Davies-Bouldin</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography 
                            variant="h6" 
                            color={comparison.comparison.metrics_delta.calinski_harabasz_score > 0 ? "success.main" : comparison.comparison.metrics_delta.calinski_harabasz_score < 0 ? "error.main" : "text.primary"}
                          >
                            {comparison.comparison.metrics_delta.calinski_harabasz_score > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.calinski_harabasz_score.toFixed(1)}
                          </Typography>
                          <Typography variant="body2">Calinski-Harabasz</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6">
                            {comparison.comparison.metrics_delta.execution_time > 0 ? '+' : ''}
                            {comparison.comparison.metrics_delta.execution_time.toFixed(1)}s
                          </Typography>
                          <Typography variant="body2">Tempo</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                  </Box>

                  {/* Valutazione qualità */}
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Valutazione:</strong> {comparison.comparison.quality_assessment}
                    </Typography>
                  </Alert>

                  {/* Differenze parametri */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="h6">
                        ⚙️ Differenze Parametri ({comparison.comparison.parameters_diff.filter(p => p.changed).length} modificati)
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <TableContainer>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Parametro</TableCell>
                              <TableCell align="center">Versione 1</TableCell>
                              <TableCell align="center">Versione 2</TableCell>
                              <TableCell align="center">Stato</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {comparison.comparison.parameters_diff.map((diff) => (
                              <TableRow key={diff.parameter}>
                                <TableCell>{diff.parameter}</TableCell>
                                <TableCell align="center">{String(diff.value1)}</TableCell>
                                <TableCell align="center">{String(diff.value2)}</TableCell>
                                <TableCell align="center">
                                  <Chip
                                    label={diff.changed ? 'Modificato' : 'Identico'}
                                    color={diff.changed ? 'warning' : 'success'}
                                    size="small"
                                  />
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              )}
            </Box>
          )}

          {/* Tab Trend */}
          {activeTab === 2 && (
            <Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
                  <FormControl fullWidth>
                    <InputLabel>Periodo Analisi</InputLabel>
                    <Select
                      value={trendDays}
                      onChange={(e) => setTrendDays(Number(e.target.value))}
                      label="Periodo Analisi"
                    >
                      <MenuItem value={7}>Ultimi 7 giorni</MenuItem>
                      <MenuItem value={15}>Ultimi 15 giorni</MenuItem>
                      <MenuItem value={30}>Ultimo mese</MenuItem>
                      <MenuItem value={60}>Ultimi 2 mesi</MenuItem>
                      <MenuItem value={90}>Ultimi 3 mesi</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              </Box>

              {trendData && (
                <Box>
                  {/* Statistiche summary */}
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            {trendData.metrics_summary.avg_clusters.toFixed(1)}
                          </Typography>
                          <Typography variant="body2">Media Cluster</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {trendData.metrics_summary.total_versions} versioni
                          </Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="secondary">
                            {trendData.metrics_summary.avg_outliers.toFixed(1)}
                          </Typography>
                          <Typography variant="body2">Media Outliers</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="success.main">
                            {trendData.metrics_summary.avg_silhouette.toFixed(3)}
                          </Typography>
                          <Typography variant="body2">Media Silhouette</Typography>
                        </CardContent>
                      </Card>
                    </Box>
                    <Box sx={{ flex: '1 1 200px', minWidth: '200px' }}>
                      <Card>
                        <CardContent sx={{ textAlign: 'center' }}>
                          <Typography variant="h6" color="info.main">
                            {trendData.metrics_summary.best_silhouette.toFixed(3)}
                          </Typography>
                          <Typography variant="body2">Best Silhouette</Typography>
                          <Typography variant="caption" color="text.secondary">
                            v{trendData.metrics_summary.best_version}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Box>
                  </Box>

                  {/* Grafici trend */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      📈 Evoluzione Metriche nel Tempo
                    </Typography>
                    
                    {/* Grafico principale: Cluster e Outliers */}
                    <Box sx={{ height: 300, mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ mb: 1 }}>
                        Cluster e Outliers
                      </Typography>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trendData.trend_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="version_number" 
                            label={{ value: 'Versione', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis />
                          <RechartsTooltip 
                            labelFormatter={(value: string | number) => `Versione ${value}`}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="n_clusters"
                            stroke="#1976d2"
                            strokeWidth={2}
                            name="N. Cluster"
                          />
                          <Line
                            type="monotone"
                            dataKey="n_outliers"
                            stroke="#d32f2f"
                            strokeWidth={2}
                            name="N. Outliers"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>

                    {/* Grafico Metriche di Qualità */}
                    <Box sx={{ height: 350, mb: 3 }}>
                      <Typography variant="subtitle1" sx={{ mb: 1 }}>
                        Metriche di Qualità Clustering
                      </Typography>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trendData.trend_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="version_number" 
                            label={{ value: 'Versione', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis yAxisId="silhouette" domain={[0, 1]} />
                          <YAxis yAxisId="davies" orientation="right" domain={[0, 'dataMax']} />
                          <RechartsTooltip 
                            labelFormatter={(value: string | number) => `Versione ${value}`}
                            formatter={(value: any, name: string) => {
                              if (name === 'Silhouette Score') return [value?.toFixed(3) || 'N/A', name];
                              if (name === 'Davies-Bouldin Index') return [value?.toFixed(3) || 'N/A', name];
                              return [value, name];
                            }}
                          />
                          <Legend />
                          <Line
                            yAxisId="silhouette"
                            type="monotone"
                            dataKey="silhouette_score"
                            stroke="#2e7d32"
                            strokeWidth={3}
                            name="Silhouette Score (più alto = migliore)"
                            connectNulls={false}
                            dot={{ fill: '#2e7d32', strokeWidth: 2, r: 4 }}
                          />
                          <Line
                            yAxisId="davies"
                            type="monotone"
                            dataKey="davies_bouldin_score"
                            stroke="#ff9800"
                            strokeWidth={3}
                            name="Davies-Bouldin Index (più basso = migliore)"
                            connectNulls={false}
                            dot={{ fill: '#ff9800', strokeWidth: 2, r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>

                    {/* Grafico Calinski-Harabasz separato */}
                    <Box sx={{ height: 300 }}>
                      <Typography variant="subtitle1" sx={{ mb: 1 }}>
                        Calinski-Harabasz Index (più alto = migliore)
                      </Typography>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trendData.trend_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="version_number" 
                            label={{ value: 'Versione', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis domain={[0, 'dataMax']} />
                          <RechartsTooltip 
                            labelFormatter={(value: string | number) => `Versione ${value}`}
                            formatter={(value: any, name: string) => {
                              return [value?.toFixed(1) || 'N/A', name];
                            }}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="calinski_harabasz_score"
                            stroke="#9c27b0"
                            strokeWidth={3}
                            name="Calinski-Harabasz Index"
                            connectNulls={false}
                            dot={{ fill: '#9c27b0', strokeWidth: 2, r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Dialog per visualizzazione dettagliata */}
      <Dialog 
        open={viewDialogOpen} 
        onClose={() => setViewDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">
              {selectedVersionDetail 
                ? `🔍 Clustering v${selectedVersionDetail.version_number} - Dettagli`
                : 'Caricamento...'
              }
            </Typography>
            <IconButton onClick={() => setViewDialogOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        
        <DialogContent dividers>
          {selectedVersionDetail ? (
            <Box>
              {/* Info versione */}
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
                <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
                  <Typography variant="body2"><strong>Data:</strong> {formatDate(selectedVersionDetail.created_at)}</Typography>
                  <Typography variant="body2"><strong>Tenant:</strong> {selectedVersionDetail.tenant_id}</Typography>
                  <Typography variant="body2"><strong>Tempo esecuzione:</strong> {selectedVersionDetail.execution_time.toFixed(1)}s</Typography>
                </Box>
                <Box sx={{ flex: '1 1 300px', minWidth: '300px' }}>
                  <Typography variant="body2"><strong>Cluster:</strong> {selectedVersionDetail.n_clusters}</Typography>
                  <Typography variant="body2"><strong>Outliers:</strong> {selectedVersionDetail.n_outliers}</Typography>
                  <Typography variant="body2"><strong>Silhouette:</strong> {selectedVersionDetail.silhouette_score.toFixed(3)}</Typography>
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Visualizzazione */}
              {selectedVersionDetail.results_data?.visualization_data && 
               selectedVersionDetail.results_data.visualization_data.points && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    🎨 Visualizzazione Clustering
                  </Typography>
                  <ClusterVisualizationComponent
                    mode="parameters"
                    visualizationData={{
                      ...selectedVersionDetail.results_data.visualization_data,
                      statistics: {
                        total_points: selectedVersionDetail.results_data.visualization_data.total_points,
                        n_clusters: selectedVersionDetail.results_data.visualization_data.n_clusters,
                        n_outliers: selectedVersionDetail.results_data.visualization_data.n_outliers
                      }
                    }}
                    height={500}
                  />
                </Box>
              )}
            </Box>
          ) : (
            <Box display="flex" justifyContent="center" p={3}>
              <LinearProgress sx={{ width: '100%' }} />
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setViewDialogOpen(false)}>
            Chiudi
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ClusteringVersionManager;
