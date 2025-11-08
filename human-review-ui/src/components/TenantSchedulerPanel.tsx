import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  FormControlLabel,
  Grid,
  TextField,
  MenuItem,
  Button,
  Alert,
  Stack,
  Divider
} from '@mui/material';
import { useTenant } from '../contexts/TenantContext';
import { apiService } from '../services/apiService';

type Unit = 'minutes' | 'hours' | 'days' | 'weeks';

const unitOptions: Array<{ value: Unit; label: string }> = [
  { value: 'minutes', label: 'Minuti' },
  { value: 'hours', label: 'Ore' },
  { value: 'days', label: 'Giorni' },
  { value: 'weeks', label: 'Settimane' },
];

const isoToLocalInput = (iso?: string | null): string => {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const pad = (n: number) => String(n).padStart(2, '0');
    const yyyy = d.getFullYear();
    const mm = pad(d.getMonth() + 1);
    const dd = pad(d.getDate());
    const hh = pad(d.getHours());
    const mi = pad(d.getMinutes());
    return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
  } catch {
    return '';
  }
};

const TenantSchedulerPanel: React.FC = () => {
  const { selectedTenant } = useTenant();
  const tenantId = selectedTenant?.tenant_id;

  const [enabled, setEnabled] = useState(false);
  const [unit, setUnit] = useState<Unit>('hours');
  const [value, setValue] = useState<number>(24);
  const [startAt, setStartAt] = useState<string>(''); // datetime-local
  const [nextRunAt, setNextRunAt] = useState<string>('');
  const [lastRunAt, setLastRunAt] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      if (!tenantId) return;
      setLoading(true);
      setError(null);
      setSuccess(null);
      try {
        const resp = await apiService.getSchedulerConfig(tenantId);
        const cfg = (resp as any).config || {};
        setEnabled(Boolean(cfg.enabled));
        setUnit((cfg.frequency_unit || 'hours') as Unit);
        setValue(Number(cfg.frequency_value || 24));
        setStartAt(isoToLocalInput(cfg.start_at));
        setNextRunAt(cfg.next_run_at || '');
        setLastRunAt(cfg.last_run_at || '');
      } catch (e: any) {
        setError(e?.message || 'Errore caricamento configurazione');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [tenantId]);

  const handleSave = async () => {
    if (!tenantId) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const payload = {
        enabled,
        frequency_unit: unit,
        frequency_value: value,
        start_at: startAt || null,
      };
      const resp = await apiService.setSchedulerConfig(tenantId, payload);
      const cfg = (resp as any).config || {};
      setNextRunAt(cfg.next_run_at || '');
      setLastRunAt(cfg.last_run_at || '');
      setSuccess('Configurazione salvata');
    } catch (e: any) {
      setError(e?.message || 'Errore salvataggio configurazione');
    } finally {
      setSaving(false);
    }
  };

  const handleRunNow = async () => {
    if (!tenantId) return;
    setError(null);
    setSuccess(null);
    try {
      await apiService.runSchedulerNow(tenantId);
      setSuccess('Esecuzione avviata');
    } catch (e: any) {
      setError(e?.message || 'Errore avvio esecuzione');
    }
  };

  return (
    <Card variant="outlined">
      <CardContent>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={2}>
          <Typography variant="h6">🗓️ Scheduler per-tenant</Typography>
          <FormControlLabel
            control={<Switch checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />}
            label={enabled ? 'Abilitato' : 'Disabilitato'}
          />
        </Stack>

        <Box mt={2}>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <TextField
              select
              fullWidth
              label="Unità"
              value={unit}
              onChange={(e) => setUnit(e.target.value as Unit)}
              disabled={loading}
            >
              {unitOptions.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              type="number"
              fullWidth
              label="Frequenza"
              value={value}
              inputProps={{ min: 1 }}
              onChange={(e) => setValue(Math.max(1, Number(e.target.value)))}
              disabled={loading}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Inizio (opzionale)"
              type="datetime-local"
              value={startAt}
              onChange={(e) => setStartAt(e.target.value)}
              InputLabelProps={{ shrink: true }}
              helperText="Lascia vuoto per iniziare subito secondo frequenza"
              disabled={loading}
            />
          </Grid>
        </Grid>

        <Box mt={2}>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Prossima esecuzione"
                value={nextRunAt ? new Date(nextRunAt).toLocaleString() : '—'}
                InputProps={{ readOnly: true }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Ultima esecuzione"
                value={lastRunAt ? new Date(lastRunAt).toLocaleString() : '—'}
                InputProps={{ readOnly: true }}
              />
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Stack direction="row" spacing={2}>
          <Button variant="contained" onClick={handleSave} disabled={saving || loading}>
            Salva configurazione
          </Button>
          <Button variant="outlined" onClick={handleRunNow} disabled={loading}>
            Esegui adesso
          </Button>
        </Stack>
      </CardContent>
    </Card>
  );
};

export default TenantSchedulerPanel;

