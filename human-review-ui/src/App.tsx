import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  IconButton,
  Chip,
  Fab
} from '@mui/material';
import { MenuOutlined, ChevronRight } from '@mui/icons-material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ReviewDashboard from './components/ReviewDashboard';
import ReviewStats from './components/ReviewStats';
import CaseDetail from './components/CaseDetail';
import PromptManager from './components/PromptManager';
import ExampleManager from './components/ExampleManager';
import ToolManager from './components/ToolManager';
import ClusteringParametersManager from './components/ClusteringParametersManager';
import ClusteringStatisticsManager from './components/ClusteringStatisticsManager';
import AIConfigurationManager from './components/AIConfigurationManager';
import TenantSelector from './components/TenantSelector';
import { TenantProvider, useTenant } from './contexts/TenantContext';
import { apiService } from './services/apiService';
import { ReviewCase } from './types/ReviewCase';

// Tema Material-UI personalizzato
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function AppContent() {
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedCase, setSelectedCase] = useState<ReviewCase | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [tenantSelectorOpen, setTenantSelectorOpen] = useState(true); // Inizia aperto
  
  // Larghezza del drawer
  const drawerWidth = 280;

  // Usa il context per il tenant
  const { selectedTenant } = useTenant();
  // Passa l'oggetto tenant completo anziché solo il nome
  const tenant = selectedTenant || { 
    tenant_id: '015007d9-d413-11ef-86a5-96000228e7fe',  // Default humanitas
    tenant_name: 'humanitas',
    is_active: true 
  };

  const handleCaseSelect = (caseItem: ReviewCase) => {
    setSelectedCase(caseItem);
    setCurrentTab(selectedCase ? 5 : 5); // Switch to case detail tab (index 5 now)
  };

  const handleCaseResolved = () => {
    setSelectedCase(null);
    setCurrentTab(0); // Back to dashboard
    setRefreshTrigger(prev => prev + 1); // Trigger refresh
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    // Non permettere di cambiare al tab caso se non c'è un caso selezionato
    const caseTabIndex = selectedCase ? 5 : -1; // Cambiato da 3 a 5
    if (newValue === caseTabIndex && !selectedCase) {
      return;
    }
    setCurrentTab(newValue);
    // Se cambiamo tab e non stiamo andando al case detail, reset case selection
    if (newValue !== caseTabIndex) {
      setSelectedCase(null);
    }
  };

  const createMockCases = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await apiService.createMockCases(tenant.tenant_id, 3);
      setRefreshTrigger(prev => prev + 1); // Trigger refresh
    } catch (err) {
      setError('Errore nella creazione dei casi mock');
      console.error('Error creating mock cases:', err);
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh ogni 2 minuti (120 secondi) - meno aggressivo
  useEffect(() => {
    const interval = setInterval(() => {
      if (currentTab === 0) { // Solo se siamo nella dashboard
        setRefreshTrigger(prev => prev + 1);
      }
    }, 120000); // 2 minuti invece di 30 secondi

    return () => clearInterval(interval);
  }, [currentTab]);

  return (
    <Box sx={{ display: 'flex' }}>
      {/* Barra laterale fissa */}
      <TenantSelector
        open={tenantSelectorOpen}
        onToggle={() => setTenantSelectorOpen(!tenantSelectorOpen)}
        drawerWidth={drawerWidth}
      />

      {/* Contenuto principale */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { sm: `calc(100% - ${tenantSelectorOpen ? drawerWidth : 0}px)` },
          transition: 'width 0.3s ease',
        }}
      >
        <AppBar 
          position="fixed" 
          sx={{ 
            width: { sm: `calc(100% - ${tenantSelectorOpen ? drawerWidth : 0}px)` },
            ml: { sm: tenantSelectorOpen ? `${drawerWidth}px` : 0 },
            transition: 'width 0.3s ease, margin 0.3s ease',
          }}
        >
          <Toolbar>
            {!tenantSelectorOpen && (
              <IconButton
                edge="start"
                color="inherit"
                aria-label="apri menu tenant"
                onClick={() => setTenantSelectorOpen(true)}
                sx={{ mr: 2 }}
              >
                <MenuOutlined />
              </IconButton>
            )}
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🏥 Human Review Interface
              {selectedTenant && (
                <Chip
                  label={selectedTenant.tenant_name.toUpperCase()}
                  color="secondary"
                  variant="outlined"
                  size="small"
                  sx={{ ml: 1, color: 'white', borderColor: 'white' }}
                />
              )}
            </Typography>
            <Typography variant="body2">
              Sistema di Supervisione Umana
            </Typography>
          </Toolbar>
        </AppBar>

        {/* Spaziatura per AppBar fissa */}
        <Toolbar />

        <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="review tabs">
              <Tab label="Dashboard Revisione" />
              <Tab label="Statistiche" />
              <Tab label="Configurazione" />
              <Tab label="Parametri Clustering" />
              <Tab label="📊 Statistiche Clustering" />
              {selectedCase && (
                <Tab label={`Caso: ${selectedCase.session_id.substring(0, 8)}...`} />
              )}
            </Tabs>
          </Box>

          <TabPanel value={currentTab} index={0}>
            <ReviewDashboard
              tenant={tenant}
              onCaseSelect={handleCaseSelect}
              onCreateMockCases={createMockCases}
              refreshTrigger={refreshTrigger}
              loading={loading}
            />
          </TabPanel>

          <TabPanel value={currentTab} index={1}>
            <ReviewStats
              tenant={tenant}
              refreshTrigger={refreshTrigger}
            />
          </TabPanel>

          <TabPanel value={currentTab} index={2}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {/* Sezione Configurazione AI - NUOVA SEZIONE PRIORITARIA */}
              <Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  ⚙️ Configurazione AI
                </Typography>
                <AIConfigurationManager open={true} />
              </Box>

              {/* Separatore */}
              <Box sx={{ borderTop: '1px solid #e0e0e0', my: 2 }} />

              {/* Sezione Prompts - SPOSTATA PRIMA */}
              <Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  Gestione Prompts
                </Typography>
                <PromptManager open={true} />
              </Box>

              {/* Separatore */}
              <Box sx={{ borderTop: '1px solid #e0e0e0', my: 2 }} />

              {/* Sezione Esempi - NUOVA */}
              <Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  Gestione Esempi
                </Typography>
                <ExampleManager open={true} />
              </Box>

              {/* Separatore */}
              <Box sx={{ borderTop: '1px solid #e0e0e0', my: 2 }} />

              {/* Sezione Tools - SPOSTATA DOPO */}
              <Box>
                <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
                  Gestione Tools
                </Typography>
                <ToolManager open={true} />
              </Box>
            </Box>
          </TabPanel>

          <TabPanel value={currentTab} index={3}>
            <ClusteringParametersManager />
          </TabPanel>

          <TabPanel value={currentTab} index={4}>
            <ClusteringStatisticsManager />
          </TabPanel>

          {selectedCase && (
            <TabPanel value={currentTab} index={5}>
              <CaseDetail
                case={selectedCase}
                tenant={tenant}
                onCaseResolved={handleCaseResolved}
                onBack={() => setCurrentTab(0)}
              />
            </TabPanel>
          )}

          {loading && (
            <Box display="flex" justifyContent="center" mt={2}>
              <CircularProgress />
            </Box>
          )}
        </Container>

        {/* Floating Action Button per aprire la barra quando è chiusa */}
        {!tenantSelectorOpen && (
          <Fab
            color="primary"
            aria-label="apri barra tenant"
            onClick={() => setTenantSelectorOpen(true)}
            sx={{
              position: 'fixed',
              bottom: 24,
              left: 16,
              zIndex: 1200,
            }}
            size="small"
          >
            <ChevronRight />
          </Fab>
        )}
      </Box>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <TenantProvider>
        <AppContent />
      </TenantProvider>
    </ThemeProvider>
  );
}

export default App;
