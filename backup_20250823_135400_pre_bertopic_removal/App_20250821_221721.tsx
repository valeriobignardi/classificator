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
  Chip
} from '@mui/material';
import { MenuOutlined } from '@mui/icons-material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import ReviewDashboard from './components/ReviewDashboard';
import ReviewStats from './components/ReviewStats';
import CaseDetail from './components/CaseDetail';
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
  const [tenantSelectorOpen, setTenantSelectorOpen] = useState(false);

  // Usa il context per il tenant
  const { selectedTenant } = useTenant();
  const tenant = selectedTenant?.nome.toLowerCase() || 'humanitas'; // Fallback per compatibilità

  const handleCaseSelect = (caseItem: ReviewCase) => {
    setSelectedCase(caseItem);
    setCurrentTab(2); // Switch to case detail tab
  };

  const handleCaseResolved = () => {
    setSelectedCase(null);
    setCurrentTab(0); // Back to dashboard
    setRefreshTrigger(prev => prev + 1); // Trigger refresh
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    // Non permettere di cambiare al tab 2 se non c'è un caso selezionato
    if (newValue === 2 && !selectedCase) {
      return;
    }
    setCurrentTab(newValue);
    // Se cambiamo tab e non stiamo andando al case detail, reset case selection
    if (newValue !== 2) {
      setSelectedCase(null);
    }
  };

  const createMockCases = async () => {
    setLoading(true);
    setError(null);
    
    try {
      await apiService.createMockCases(tenant, 3);
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
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="menu"
            onClick={() => setTenantSelectorOpen(true)}
            sx={{ mr: 2 }}
          >
            <MenuOutlined />
          </IconButton>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🏥 Human Review Interface
            {selectedTenant && (
              <Chip
                label={selectedTenant.nome.toUpperCase()}
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

      <TenantSelector
        open={tenantSelectorOpen}
        onClose={() => setTenantSelectorOpen(false)}
      />

      <Container maxWidth="xl" sx={{ mt: 2 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="review tabs">
            <Tab label="Dashboard Revisione" />
            <Tab label="Statistiche" />
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

        {selectedCase && (
          <TabPanel value={currentTab} index={2}>
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
