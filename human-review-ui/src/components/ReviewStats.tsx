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
}

interface ReviewStatsProps {
  tenant: Tenant;
  refreshTrigger: number;
}

const ReviewStats: React.FC<ReviewStatsProps> = ({ tenant, refreshTrigger }) => {
  const [selectedTenant, setSelectedTenant] = useState<string>(tenant?.tenant_id || '');
  const [availableTenants, setAvailableTenants] = useState<string[]>([]);
  const [labelStats, setLabelStats] = useState<LabelStat[]>([]);
  const [generalStats, setGeneralStats] = useState<GeneralStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Carica tenants disponibili
  const loadAvailableTenants = useCallback(async () => {
    try {
      const response = await apiService.getAvailableTenants();
      setAvailableTenants(response.tenants);
      
      // Se non c'è tenant selezionato, usa il primo disponibile
      if (!selectedTenant && response.tenants.length > 0) {
        setSelectedTenant(response.tenants[0]);
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
      const response = await apiService.getLabelStatistics(selectedTenant);
      setLabelStats(response.labels || []);
      setGeneralStats(response.general_stats || null);
    } catch (err) {
      setError('Errore nel caricamento delle statistiche delle etichette');
      console.error('Error loading label stats:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedTenant]);

  useEffect(() => {
    loadAvailableTenants();
  }, [loadAvailableTenants]);

  useEffect(() => {
    if (selectedTenant) {
      loadLabelStatistics();
    }
  }, [loadLabelStatistics, refreshTrigger, selectedTenant]);

  // Genera colori per il grafico a torta
  const generateColors = (count: number) => {
    const colors = [
      '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
      '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384',
      '#36A2EB', '#FFCE56', '#FF9F40', '#9966FF', '#C9CBCF'
    ];
    
    return Array.from({ length: count }, (_, i) => colors[i % colors.length]);
  };

  // Prepara dati per il grafico a torta
  const pieChartData = {
    labels: labelStats.map(stat => stat.tag_name),
    datasets: [
      {
        data: labelStats.map(stat => stat.total_count),
        backgroundColor: generateColors(labelStats.length),
        borderColor: generateColors(labelStats.length).map(color => color + '80'),
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

      {/* Selezione Tenant */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <FormControl fullWidth>
            <InputLabel>Seleziona Tenant</InputLabel>
            <Select
              value={selectedTenant}
              onChange={(e) => setSelectedTenant(e.target.value)}
              disabled={loading}
            >
              {availableTenants.map((tenant) => (
                <MenuItem key={tenant} value={tenant}>
                  {tenant.charAt(0).toUpperCase() + tenant.slice(1)}
                </MenuItem>
              ))}
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
                        Classificazioni Totali
                      </Typography>
                    </div>
                    <div style={{ flex: '1 1 200px', textAlign: 'center' }}>
                      <Typography variant="h4" color="success.main">
                        {generalStats.total_sessions}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Sessioni Uniche
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
                        {labelStats
                          .sort((a, b) => b.total_count - a.total_count)
                          .map((stat, index) => (
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