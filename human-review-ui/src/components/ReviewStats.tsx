import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  PieChart as PieChartIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import { Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { apiService } from '../services/apiService';
import ClusterVisualizationComponent from './ClusterVisualizationComponent';
import { Tenant } from '../types/Tenant';

// Registra i componenti di Chart.js
ChartJS.register(ArcElement, Tooltip, Legend);

interface LabelStat {
  tag_name: string;
  total_count: number;
  avg_confidence: number;
  unique_sessions: number;
  methods: { [method: string]: number };
}

interface GeneralStats {
  total_classifications: number;
  total_sessions: number;
  total_labels: number;
  avg_confidence_overall: number;
  total_messages?: number;
}

interface ReviewStatsProps {
  tenant: Tenant;
  refreshTrigger: number;
}

const ReviewStats: React.FC<ReviewStatsProps> = ({ tenant, refreshTrigger }) => {
  // FIX: Usa tenant_slug invece di tenant_name per la selezione (slug è minuscolo e corrisponde alle opzioni)
  const [selectedTenant, setSelectedTenant] = useState<string>(tenant?.tenant_slug || '');
  const [availableTenants, setAvailableTenants] = useState<any[]>([]); // Cambiato da string[] a any[]
  const [labelStats, setLabelStats] = useState<LabelStat[]>([]);
  const [generalStats, setGeneralStats] = useState<GeneralStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 🆕 Stato per visualizzazione cluster nella sezione Statistiche
  const [clusterVisLoading, setClusterVisLoading] = useState(false);
  const [clusterVisError, setClusterVisError] = useState<string | null>(null);
  const [clusterVisualizationData, setClusterVisualizationData] = useState<any | null>(null);

  // Carica tenants disponibili
  const loadAvailableTenants = useCallback(async () => {
    try {
      const response = await apiService.getAvailableTenants();
      console.log('🔍 [DEBUG] Tenants ricevuti:', response.tenants);
      
      // FIX: L'API restituisce array di oggetti, non stringhe
      // Estrai il tenant_slug (o tenant_name.toLowerCase()) da ogni oggetto
      const tenantList = response.tenants.map((t: any) => {
        if (typeof t === 'string') {
          return t;
        } else if (t && typeof t === 'object') {
          // Preferisci tenant_slug, poi tenant_name in minuscolo, poi name
          return t.tenant_slug || t.slug || (t.tenant_name && t.tenant_name.toLowerCase()) || (t.name && t.name.toLowerCase()) || String(t);
        }
        return String(t);
      });
      
      console.log('🔍 [DEBUG] Tenant list processata:', tenantList);
      setAvailableTenants(tenantList);
      
      // Validazione: se selectedTenant non è nelle opzioni disponibili, resettalo
      if (selectedTenant && !tenantList.includes(selectedTenant)) {
        console.log('🔍 [DEBUG] selectedTenant non valido, resetting:', selectedTenant);
        setSelectedTenant(tenantList.length > 0 ? tenantList[0] : '');
      }
      // Se non c'è tenant selezionato, usa il primo disponibile
      else if (!selectedTenant && tenantList.length > 0) {
        setSelectedTenant(tenantList[0]);
      }
    } catch (err) {
      console.error('Error loading tenants:', err);
      setError('Errore nel caricamento dei tenant');
    }
  }, [selectedTenant]);

  // Carica statistiche per il tenant selezionato
  const loadLabelStatistics = useCallback(async () => {
    if (!selectedTenant) return;

    setLoading(true);
    setError(null);

    try {
      // FIX: Usa sempre il tenant_id (UUID) per le chiamate API
      // selectedTenant contiene ora il tenant_slug, ma l'API si aspetta il tenant_id
      let tenantIdToUse = selectedTenant;
      
      // Se abbiamo il tenant principale passato come prop e il suo slug corrisponde
      if (tenant && tenant.tenant_slug === selectedTenant) {
        tenantIdToUse = tenant.tenant_id;  // Usa l'UUID per l'API
      }
      
      const response = await apiService.getLabelStatistics(tenantIdToUse);
      setLabelStats(response.labels || []);
      setGeneralStats(response.general_stats || null);
    } catch (err) {
      setError('Errore nel caricamento delle statistiche delle etichette');
      console.error('Error loading label stats:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant, tenant]);

  // 🆕 Carica dati visualizzazione cluster 2D/3D per la sezione Statistiche
  const loadClusterVisualization = useCallback(async () => {
    if (!selectedTenant) return;

    setClusterVisLoading(true);
    setClusterVisError(null);

    try {
      // Usa sempre il tenant_id (UUID) per l'API
      let tenantIdToUse = selectedTenant;
      if (tenant && tenant.tenant_slug === selectedTenant) {
        tenantIdToUse = tenant.tenant_id;
      }

      const response = await apiService.getClusteringStatistics(tenantIdToUse);
      if (response.success && response.visualization_data) {
        setClusterVisualizationData(response.visualization_data);
      } else {
        setClusterVisError(response.error || 'Errore caricamento visualizzazioni cluster');
      }
    } catch (err: any) {
      setClusterVisError(err.message || 'Errore caricamento visualizzazioni cluster');
    } finally {
      setClusterVisLoading(false);
    }
  }, [selectedTenant, tenant]);

  // FIX: Aggiorna selectedTenant quando cambia il prop tenant (usa tenant_slug)  
  useEffect(() => {
    if (tenant?.tenant_slug) {
      setSelectedTenant(tenant.tenant_slug);
    }
  }, [tenant?.tenant_slug]);

  useEffect(() => {
    loadAvailableTenants();
  }, [loadAvailableTenants]);

  useEffect(() => {
    if (selectedTenant) {
      loadLabelStatistics();
      loadClusterVisualization();
    }
  }, [loadLabelStatistics, loadClusterVisualization, refreshTrigger, selectedTenant]);

  // Genera colori distinti per il grafico a torta (senza duplicati evidenti)
  const generateColors = (count: number) => {
    return Array.from({ length: count }, (_, i) => {
      const hue = Math.round((360 / Math.max(count, 1)) * i);
      return `hsl(${hue}, 70%, 55%)`;
    });
  };

  // Ordina le etichette per occorrenze come nella tabella, per coerenza visiva
  const sortedLabelStats = React.useMemo(() => {
    return [...labelStats].sort((a, b) => b.total_count - a.total_count);
  }, [labelStats]);

  // Prepara dati per il grafico a torta (usando l'ordinamento della tabella)
  const pieChartData = {
    labels: sortedLabelStats.map(stat => stat.tag_name),
    datasets: [
      {
        data: sortedLabelStats.map(stat => stat.total_count),
        backgroundColor: generateColors(sortedLabelStats.length),
        borderColor: generateColors(sortedLabelStats.length).map(c => c.replace('hsl(', 'hsla(').replace(')', ', 0.8)')),
        borderWidth: 2,
      },
    ],
  };

  const chartOptions: ChartOptions<'pie'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          boxWidth: 12,
          padding: 15,
          font: {
            size: 11
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed;
            const total = context.dataset.data.reduce((a: number, b: number) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    },
    maintainAspectRatio: false
  };

  if (loading) {
    return (
      <Box display="flex" flexDirection="column" alignItems="center" p={3}>
        <LinearProgress sx={{ width: '100%', mb: 2 }} />
        <Typography>Caricamento statistiche...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        📊 Statistiche Etichette
      </Typography>

      {/* 🆕 Visualizzazione Cluster 2D/3D (come richiesto) */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Box display="flex" alignItems="center" gap={1}>
              <AnalyticsIcon color="primary" />
              <Typography variant="h6">Visualizzazione Cluster 2D / 3D</Typography>
            </Box>
            <Chip label={tenant?.tenant_name || 'Tenant'} size="small" />
          </Box>

          {clusterVisError && (
            <Alert severity="error" sx={{ mb: 2 }}>{clusterVisError}</Alert>
          )}

          <ClusterVisualizationComponent
            mode="statistics"
            visualizationData={clusterVisualizationData || undefined}
            onRefresh={loadClusterVisualization}
            loading={clusterVisLoading}
            height={600}
          />
        </CardContent>
      </Card>

      {/* Selezione Tenant */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <FormControl fullWidth>
            <InputLabel>Seleziona Tenant</InputLabel>
            <Select
              value={selectedTenant && availableTenants.length > 0 ? selectedTenant : ''}
              onChange={(e) => setSelectedTenant(e.target.value)}
              disabled={loading}
            >
              {availableTenants.map((tenant) => {
                // Gestisci sia stringhe che oggetti tenant  
                const tenantValue = String(tenant); // Converti sempre a stringa
                const tenantDisplay = tenantValue && tenantValue.length > 0 
                  ? tenantValue.charAt(0).toUpperCase() + tenantValue.slice(1)
                  : 'Tenant Sconosciuto';
                
                return (
                  <MenuItem key={tenantValue} value={tenantValue}>
                    {tenantDisplay}
                  </MenuItem>
                );
              })}
            </Select>
          </FormControl>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {!selectedTenant && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Seleziona un tenant per visualizzare le statistiche
        </Alert>
      )}

      {selectedTenant && !loading && labelStats.length === 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Nessuna statistica disponibile per il tenant "{selectedTenant}"
        </Alert>
      )}

      {selectedTenant && labelStats.length > 0 && (
        <div>
          {/* Statistiche Generali */}
          {generalStats && (
            <div style={{ marginBottom: '24px' }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <AnalyticsIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">Statistiche Generali - {selectedTenant}</Typography>
                  </Box>
                  
                  <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                    <div style={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {generalStats.total_classifications}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Sessioni Classificate
                      </Typography>
                    </div>
                    <div style={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        {generalStats.total_messages !== undefined
                          ? generalStats.total_messages
                          : generalStats.total_sessions}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {generalStats.total_messages !== undefined
                          ? 'Frasi Analizzate'
                          : 'Sessioni Uniche'}
                      </Typography>
                    </div>
                    <div style={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="warning.main">
                        {generalStats.total_labels}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Etichette Diverse
                      </Typography>
                    </div>
                    <div style={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {(generalStats.avg_confidence_overall * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Confidenza Media
                      </Typography>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            {/* Grafico a Torta */}
            <div style={{ flex: '1 1 500px' }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <PieChartIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">Distribuzione Etichette</Typography>
                  </Box>
                  
                  <Box height={400}>
                    <Pie data={pieChartData} options={chartOptions} />
                  </Box>
                </CardContent>
              </Card>
            </div>

            {/* Tabella Dettagliata */}
            <div style={{ flex: '1 1 500px' }}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <AssessmentIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">Dettagli Etichette</Typography>
                  </Box>
                  
                  <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                    <Table stickyHeader size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell><strong>Etichetta</strong></TableCell>
                          <TableCell align="right"><strong>Occorrenze</strong></TableCell>
                          <TableCell align="right"><strong>Sessioni</strong></TableCell>
                          <TableCell align="right"><strong>Confidenza</strong></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {sortedLabelStats.map((stat, index) => (
                          <TableRow key={stat.tag_name} hover>
                            <TableCell>
                              <Box display="flex" alignItems="center" gap={1}>
                                <Chip 
                                  label={stat.tag_name} 
                                  size="small" 
                                  color="primary"
                                  variant="outlined"
                                />
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" fontWeight="bold">
                                {stat.total_count}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                {stat.unique_sessions}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">
                                {(stat.avg_confidence * 100).toFixed(1)}%
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      )}
    </Box>
  );
};

export default ReviewStats;
