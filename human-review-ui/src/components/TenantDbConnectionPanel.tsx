import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography
} from '@mui/material';
import { SelectChangeEvent } from '@mui/material/Select';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import SaveIcon from '@mui/icons-material/Save';
import RefreshIcon from '@mui/icons-material/Refresh';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';

import { useTenant } from '../contexts/TenantContext';
import {
  apiService,
  TenantDbConnectionConfig,
  TenantDbConnectionMetadata,
  TenantDbConnectionSavePayload
} from '../services/apiService';

type AuthMethod = 'password' | 'key' | 'both';

interface TenantDbConnectionForm {
  use_ssh_tunnel: boolean;
  ssh_host: string;
  ssh_port: string;
  ssh_username: string;
  ssh_auth_method: AuthMethod;
  ssh_password: string;
  ssh_key: string;
  ssh_key_name: string;
  ssh_key_passphrase: string;
  db_host: string;
  db_port: string;
  db_database: string;
  db_user: string;
  db_password: string;
}

const DEFAULT_FORM: TenantDbConnectionForm = {
  use_ssh_tunnel: false,
  ssh_host: '',
  ssh_port: '22',
  ssh_username: '',
  ssh_auth_method: 'password',
  ssh_password: '',
  ssh_key: '',
  ssh_key_name: '',
  ssh_key_passphrase: '',
  db_host: '',
  db_port: '3306',
  db_database: '',
  db_user: '',
  db_password: ''
};

const normalizeString = (value?: string | null) => (value ?? '').trim();

const toPortString = (fallback: string, value?: number | null) =>
  value !== undefined && value !== null ? String(value) : fallback;

const parsePort = (value: string): number | null => {
  if (!value) {
    return null;
  }
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const TenantDbConnectionPanel: React.FC = () => {
  const { selectedTenant } = useTenant();
  const tenantId = selectedTenant?.tenant_id ?? null;

  const [form, setForm] = useState<TenantDbConnectionForm>(DEFAULT_FORM);
  const [originalConfig, setOriginalConfig] = useState<TenantDbConnectionConfig | null>(null);
  const [metadata, setMetadata] = useState<TenantDbConnectionMetadata | null>(null);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [hasSshPassword, setHasSshPassword] = useState(false);
  const [hasDbPassword, setHasDbPassword] = useState(false);
  const [hasSshKey, setHasSshKey] = useState(false);
  const [hasSshKeyPassphrase, setHasSshKeyPassphrase] = useState(false);

  const [sshPasswordTouched, setSshPasswordTouched] = useState(false);
  const [dbPasswordTouched, setDbPasswordTouched] = useState(false);
  const [sshKeyTouched, setSshKeyTouched] = useState(false);
  const [sshKeyPassphraseTouched, setSshKeyPassphraseTouched] = useState(false);
  const [sshKeyFileName, setSshKeyFileName] = useState<string | null>(null);

  const resetForm = useCallback((config?: TenantDbConnectionConfig, meta?: TenantDbConnectionMetadata) => {
    if (!config) {
      setForm(DEFAULT_FORM);
      setOriginalConfig(null);
      setMetadata(meta ?? null);
      setHasSshPassword(false);
      setHasDbPassword(false);
      setHasSshKey(false);
      setHasSshKeyPassphrase(false);
    } else {
      setForm({
        use_ssh_tunnel: config.use_ssh_tunnel ?? false,
        ssh_host: config.ssh_host ?? '',
        ssh_port: toPortString(DEFAULT_FORM.ssh_port, config.ssh_port),
        ssh_username: config.ssh_username ?? '',
        ssh_auth_method: (config.ssh_auth_method as AuthMethod) || DEFAULT_FORM.ssh_auth_method,
        ssh_password: '',
        ssh_key: '',
        ssh_key_name: '',
        ssh_key_passphrase: '',
        db_host: config.db_host ?? '',
        db_port: toPortString(DEFAULT_FORM.db_port, config.db_port),
        db_database: config.db_database ?? '',
        db_user: config.db_user ?? '',
        db_password: ''
      });
      setOriginalConfig(config);
      setMetadata(meta ?? null);
      setHasSshPassword(!!config.has_ssh_password);
      setHasDbPassword(!!config.has_db_password);
      setHasSshKey(!!config.has_ssh_key);
      setHasSshKeyPassphrase(!!config.has_ssh_key_passphrase);
    }
    setSshPasswordTouched(false);
    setDbPasswordTouched(false);
    setSshKeyTouched(false);
    setSshKeyPassphraseTouched(false);
    setSshKeyFileName(null);
    setSuccess(null);
    setError(null);
  }, []);

  const fetchConfiguration = useCallback(async () => {
    if (!tenantId) {
      resetForm();
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getTenantDbConnection(tenantId);
      resetForm(response.configuration, response.metadata);
      setSuccess(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore caricamento configurazione database';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [tenantId, resetForm]);

  useEffect(() => {
    fetchConfiguration();
  }, [fetchConfiguration]);

  const tunnelEnabled = form.use_ssh_tunnel;

  const handleToggleTunnel = (event: React.ChangeEvent<HTMLInputElement>) => {
    const checked = event.target.checked;
    setForm(prev => ({ ...prev, use_ssh_tunnel: checked }));
    setSuccess(null);
  };

  const handleInputChange =
    (field: keyof TenantDbConnectionForm, options?: { preserveWhitespace?: boolean; numeric?: boolean }) =>
    (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const { preserveWhitespace = false, numeric = false } = options || {};
      let value = event.target.value;
      if (numeric) {
        value = value.replace(/[^\d]/g, '');
      } else if (!preserveWhitespace) {
        value = value.replace(/\s+/g, ' ').trimStart();
      }

      setForm(prev => ({
        ...prev,
        [field]: value
      }));

      if (field === 'ssh_password') {
        setSshPasswordTouched(true);
        setHasSshPassword(false);
      }
      if (field === 'db_password') {
        setDbPasswordTouched(true);
        setHasDbPassword(false);
      }
      if (field === 'ssh_key') {
        setSshKeyTouched(true);
        setHasSshKey(false);
        setSshKeyFileName(null);
      }
      if (field === 'ssh_key_passphrase') {
        setSshKeyPassphraseTouched(true);
        setHasSshKeyPassphrase(false);
      }
      if (field === 'ssh_key_name') {
        setSshKeyTouched(true);
      }

      setSuccess(null);
    };

  const handleSelectChange =
    (field: keyof TenantDbConnectionForm) => (event: SelectChangeEvent<AuthMethod>) => {
      const value = event.target.value as AuthMethod;
      setForm(prev => ({
        ...prev,
        [field]: value
      }));
      setSuccess(null);
    };

  const handleKeyFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const content = typeof reader.result === 'string' ? reader.result : '';
      setForm(prev => ({
        ...prev,
        ssh_key: content,
        ssh_key_name: file.name
      }));
      setSshKeyTouched(true);
      setHasSshKey(false);
      setSshKeyFileName(file.name);
      setSuccess(null);
    };
    reader.onerror = () => {
      setError('Errore durante la lettura del file della chiave SSH');
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  const handleClearSshKey = () => {
    setForm(prev => ({
      ...prev,
      ssh_key: '',
      ssh_key_name: ''
    }));
    setSshKeyTouched(true);
    setHasSshKey(false);
    setSshKeyFileName(null);
    setSuccess(null);
  };

  const handleClearSshPassword = () => {
    setForm(prev => ({ ...prev, ssh_password: '' }));
    setSshPasswordTouched(true);
    setHasSshPassword(false);
    setSuccess(null);
  };

  const handleClearDbPassword = () => {
    setForm(prev => ({ ...prev, db_password: '' }));
    setDbPasswordTouched(true);
    setHasDbPassword(false);
    setSuccess(null);
  };

  const handleClearSshPassphrase = () => {
    setForm(prev => ({ ...prev, ssh_key_passphrase: '' }));
    setSshKeyPassphraseTouched(true);
    setHasSshKeyPassphrase(false);
    setSuccess(null);
  };

  const buildPayload = useCallback((): TenantDbConnectionSavePayload => {
    const payload: TenantDbConnectionSavePayload = {
      use_ssh_tunnel: form.use_ssh_tunnel
    };

    const original: Partial<TenantDbConnectionConfig> = originalConfig || {};

    if (normalizeString(form.ssh_host) !== normalizeString(original.ssh_host)) {
      payload.ssh_host = form.ssh_host.trim() || null;
    }

    if (normalizeString(form.ssh_username) !== normalizeString(original.ssh_username)) {
      payload.ssh_username = form.ssh_username.trim() || null;
    }

    if ((form.ssh_auth_method || 'password') !== (original.ssh_auth_method || 'password')) {
      payload.ssh_auth_method = form.ssh_auth_method;
    }

    const sshPortNumber = parsePort(form.ssh_port);
    const originalSshPort = original.ssh_port ?? null;
    if (sshPortNumber !== originalSshPort) {
      payload.ssh_port = sshPortNumber;
    }

    const dbPortNumber = parsePort(form.db_port);
    const originalDbPort = original.db_port ?? null;
    if (dbPortNumber !== originalDbPort) {
      payload.db_port = dbPortNumber;
    }

    if (normalizeString(form.db_host) !== normalizeString(original.db_host)) {
      payload.db_host = form.db_host.trim() || null;
    }

    if (normalizeString(form.db_database) !== normalizeString(original.db_database)) {
      payload.db_database = form.db_database.trim() || null;
    }

    if (normalizeString(form.db_user) !== normalizeString(original.db_user)) {
      payload.db_user = form.db_user.trim() || null;
    }

    if (sshPasswordTouched) {
      payload.ssh_password = form.ssh_password ? form.ssh_password : null;
    }

    if (dbPasswordTouched) {
      payload.db_password = form.db_password ? form.db_password : null;
    }

    if (sshKeyTouched) {
      payload.ssh_key = form.ssh_key || null;
      payload.ssh_key_name = form.ssh_key_name || null;
    } else if (normalizeString(form.ssh_key_name) !== normalizeString(original.ssh_key_name)) {
      payload.ssh_key_name = form.ssh_key_name.trim() || null;
    }

    if (sshKeyPassphraseTouched) {
      payload.ssh_key_passphrase = form.ssh_key_passphrase ? form.ssh_key_passphrase : null;
    }

    return payload;
  }, [
    form,
    originalConfig,
    sshPasswordTouched,
    dbPasswordTouched,
    sshKeyTouched,
    sshKeyPassphraseTouched
  ]);

  const handleSave = async () => {
    if (!tenantId) {
      return;
    }

    const payload = buildPayload();

    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await apiService.saveTenantDbConnection(tenantId, payload);
      resetForm(response.configuration, response.metadata);
      setSuccess('Configurazione database aggiornata con successo');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Errore durante il salvataggio della configurazione';
      setError(message);
    } finally {
      setSaving(false);
    }
  };

  const canEdit = Boolean(tenantId);

  const tenantLabel = useMemo(() => {
    if (!selectedTenant) {
      return 'Nessun tenant selezionato';
    }
    return `${selectedTenant.tenant_name} (${selectedTenant.tenant_slug})`;
  }, [selectedTenant]);

  return (
    <Card elevation={1}>
      <CardContent>
        <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', md: 'center' }} spacing={2}>
          <Box>
            <Typography variant="h5" fontWeight={600}>
              🔐 Connessione Database Remoto
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Configura l&apos;accesso al database MySQL del tenant tramite tunnel SSH. Abilitare il tunnel
              implica l&apos;utilizzo esclusivo di queste credenziali per lettura e scrittura delle conversazioni.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={fetchConfiguration}
              disabled={loading || saving || !tenantId}
            >
              Ricarica
            </Button>
            <Button
              variant="contained"
              startIcon={saving ? <CircularProgress color="inherit" size={16} /> : <SaveIcon />}
              onClick={handleSave}
              disabled={saving || loading || !canEdit}
            >
              {saving ? 'Salvataggio...' : 'Salva configurazione'}
            </Button>
          </Stack>
        </Stack>

        <Divider sx={{ my: 2 }} />

        <Stack spacing={2}>
          <Alert severity="info">
            Quando il tunnel SSH è attivo <strong>tutte le letture e scritture</strong> verso il database del tenant
            passano dal server remoto configurato. Compila obbligatoriamente i parametri del database MySQL remoto.
          </Alert>

          {selectedTenant ? (
            <Alert severity="success" variant="outlined">
              Tenant corrente: <strong>{tenantLabel}</strong>
            </Alert>
          ) : (
            <Alert severity="warning" variant="outlined">
              Seleziona un tenant dal menu laterale per modificare la configurazione.
            </Alert>
          )}

          {loading && (
            <Box display="flex" justifyContent="center" py={3}>
              <CircularProgress />
            </Box>
          )}

          {!loading && error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {!loading && success && (
            <Alert severity="success" onClose={() => setSuccess(null)}>
              {success}
            </Alert>
          )}

          <FormControlLabel
            control={
              <Switch
                color="primary"
                checked={form.use_ssh_tunnel}
                onChange={handleToggleTunnel}
                disabled={!canEdit}
              />
            }
            label="Abilita tunnel SSH verso database remoto"
          />

          <Divider />

          <Typography variant="subtitle1" fontWeight={600}>
            Parametri SSH
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                label="IP / Host server SSH"
                value={form.ssh_host}
                onChange={handleInputChange('ssh_host')}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="Porta SSH"
                value={form.ssh_port}
                onChange={handleInputChange('ssh_port', { numeric: true })}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="Username SSH"
                value={form.ssh_username}
                onChange={handleInputChange('ssh_username')}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <FormControl fullWidth disabled={!canEdit}>
                <InputLabel id="ssh-auth-method-label">Metodo autenticazione</InputLabel>
                <Select
                  labelId="ssh-auth-method-label"
                  value={form.ssh_auth_method}
                  label="Metodo autenticazione"
                  onChange={handleSelectChange('ssh_auth_method')}
                >
                  <MenuItem value="password">Password</MenuItem>
                  <MenuItem value="key">Chiave privata</MenuItem>
                  <MenuItem value="both">Password + Chiave</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                label="Password SSH"
                type="password"
                value={form.ssh_password}
                onChange={handleInputChange('ssh_password', { preserveWhitespace: true })}
                fullWidth
                disabled={!canEdit || (form.ssh_auth_method === 'key' && !form.use_ssh_tunnel)}
                helperText={
                  hasSshPassword && !sshPasswordTouched
                    ? 'Lascia vuoto per mantenere la password esistente.'
                    : 'Inserisci una nuova password oppure lascia vuoto.'
                }
              />
              {hasSshPassword && (
                <Button
                  size="small"
                  color="secondary"
                  startIcon={<DeleteForeverIcon />}
                  onClick={handleClearSshPassword}
                  disabled={!canEdit}
                >
                  Rimuovi password
                </Button>
              )}
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                label="Passphrase chiave SSH"
                type="password"
                value={form.ssh_key_passphrase}
                onChange={handleInputChange('ssh_key_passphrase', { preserveWhitespace: true })}
                fullWidth
                disabled={!canEdit || (form.ssh_auth_method === 'password' && !hasSshKey)}
                helperText={
                  hasSshKeyPassphrase && !sshKeyPassphraseTouched
                    ? 'Lascia vuoto per mantenere la passphrase esistente.'
                    : 'Inserisci una nuova passphrase oppure lascia vuoto.'
                }
              />
              {hasSshKeyPassphrase && (
                <Button
                  size="small"
                  color="secondary"
                  startIcon={<DeleteForeverIcon />}
                  onClick={handleClearSshPassphrase}
                  disabled={!canEdit}
                >
                  Rimuovi passphrase
                </Button>
              )}
            </Grid>

            <Grid item xs={12}>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ xs: 'stretch', md: 'center' }}>
                <Button
                  variant="outlined"
                  startIcon={<UploadFileIcon />}
                  component="label"
                  disabled={!canEdit}
                >
                  Carica chiave da file
                  <input type="file" hidden onChange={handleKeyFileUpload} />
                </Button>
                {sshKeyFileName && (
                  <Typography variant="body2" color="text.secondary">
                    File selezionato: <strong>{sshKeyFileName}</strong>
                  </Typography>
                )}
                {hasSshKey && !sshKeyTouched && (
                  <Typography variant="body2" color="text.secondary">
                    È presente una chiave salvata. Caricane una nuova o premi &quot;Rimuovi&quot; per eliminarla.
                  </Typography>
                )}
                {(hasSshKey || sshKeyTouched) && (
                  <Button
                    color="secondary"
                    startIcon={<DeleteForeverIcon />}
                    onClick={handleClearSshKey}
                    disabled={!canEdit}
                  >
                    Rimuovi chiave
                  </Button>
                )}
              </Stack>
            </Grid>

            <Grid item xs={12}>
              <TextField
                label="Contenuto chiave privata (PEM)"
                placeholder="-----BEGIN PRIVATE KEY-----"
                value={form.ssh_key}
                onChange={handleInputChange('ssh_key', { preserveWhitespace: true })}
                fullWidth
                multiline
                minRows={5}
                disabled={!canEdit}
                helperText="Puoi incollare direttamente il contenuto della chiave privata oppure usare il pulsante di upload."
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="Nome chiave (opzionale)"
                value={form.ssh_key_name}
                onChange={handleInputChange('ssh_key_name')}
                fullWidth
                disabled={!canEdit}
              />
            </Grid>
          </Grid>

          <Divider />

          <Typography variant="subtitle1" fontWeight={600}>
            Credenziali Database MySQL Remoto
          </Typography>

          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <TextField
                label="Host Database"
                value={form.db_host}
                onChange={handleInputChange('db_host')}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <TextField
                label="Porta DB"
                value={form.db_port}
                onChange={handleInputChange('db_port', { numeric: true })}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="Nome Database"
                value={form.db_database}
                onChange={handleInputChange('db_database')}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                label="Utente DB"
                value={form.db_user}
                onChange={handleInputChange('db_user')}
                fullWidth
                disabled={!canEdit}
                required={tunnelEnabled}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                label="Password DB"
                type="password"
                value={form.db_password}
                onChange={handleInputChange('db_password', { preserveWhitespace: true })}
                fullWidth
                disabled={!canEdit}
                helperText={
                  hasDbPassword && !dbPasswordTouched
                    ? 'Lascia vuoto per mantenere la password esistente.'
                    : 'Inserisci una nuova password oppure lascia vuoto.'
                }
              />
              {hasDbPassword && (
                <Button
                  size="small"
                  color="secondary"
                  startIcon={<DeleteForeverIcon />}
                  onClick={handleClearDbPassword}
                  disabled={!canEdit}
                >
                  Rimuovi password DB
                </Button>
              )}
            </Grid>
          </Grid>

          {(originalConfig?.updated_at || originalConfig?.created_at) && (
            <Typography variant="caption" color="text.secondary">
              Ultimo aggiornamento configurazione:{' '}
              {originalConfig?.updated_at ?? originalConfig?.created_at}
            </Typography>
          )}
          {metadata && (
            <Typography variant="caption" color="text.secondary">
              Tenant ID: {metadata.tenant_id}
            </Typography>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
};

export default TenantDbConnectionPanel;
