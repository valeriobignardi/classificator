import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, TextInput, Button, Alert, Switch } from 'react-native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';

export default function Scheduler() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<any>(null);
  const [cfg, setCfg] = useState<any>({ enabled: false, frequency_unit: 'hours', frequency_value: 24, start_at: null });

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const st = await api.getSchedulerStatus();
      setStatus(st);
      if (selectedTenant) {
        const c = await api.getSchedulerConfig(selectedTenant.tenant_id);
        setCfg({
          enabled: !!c.config?.enabled,
          frequency_unit: c.config?.frequency_unit || 'hours',
          frequency_value: c.config?.frequency_value || 24,
          start_at: c.config?.start_at || null,
        });
      }
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento scheduler');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  const save = async () => {
    if (!selectedTenant) return;
    try {
      setLoading(true);
      await api.setSchedulerConfig(selectedTenant.tenant_id, cfg);
      Alert.alert('OK', 'Configurazione aggiornata');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Salvataggio fallito');
    } finally {
      setLoading(false);
    }
  };

  const start = async () => { setLoading(true); try { await api.startScheduler(); await load(); } finally { setLoading(false); } };
  const stop = async () => { setLoading(true); try { await api.stopScheduler(); await load(); } finally { setLoading(false); } };
  const runNow = async () => {
    if (!selectedTenant) return;
    setLoading(true);
    try {
      await api.runSchedulerNow(selectedTenant.tenant_slug);
      Alert.alert('OK', 'Esecuzione avviata');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Impossibile avviare esecuzione');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 12 }}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}

      <Text style={styles.title}>Scheduler</Text>
      <View style={styles.card}>
        <Text>Abilitato</Text>
        <View style={{ alignItems: 'flex-start' }}>
          <Switch value={cfg.enabled} onValueChange={(v) => setCfg((p: any) => ({ ...p, enabled: v }))} />
        </View>
        <Text>Frequenza (unit)</Text>
        <TextInput style={styles.input} value={String(cfg.frequency_unit)} onChangeText={(v) => setCfg((p: any) => ({ ...p, frequency_unit: v }))} />
        <Text>Frequenza (valore)</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(cfg.frequency_value)} onChangeText={(v) => setCfg((p: any) => ({ ...p, frequency_value: Number(v)||0 }))} />
        <Button title="Salva" onPress={save} />
      </View>

      <View style={styles.card}>
        <Text style={styles.subtitle}>Controllo</Text>
        <View style={{ flexDirection: 'row', gap: 12 }}>
          <Button title="Start" onPress={start} />
          <Button title="Stop" onPress={stop} />
          <Button title="Run Now" onPress={runNow} />
        </View>
      </View>

      {status && (
        <View style={styles.card}>
          <Text style={styles.subtitle}>Stato</Text>
          <Text>running: {String(status.running)}</Text>
          <Text>threads: {String(status.threads || '')}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 6 },
  subtitle: { fontSize: 16, fontWeight: '600', marginBottom: 8 },
  card: { backgroundColor: '#fff', padding: 12, borderRadius: 8, elevation: 1, marginBottom: 12 },
  input: { backgroundColor: '#fff', borderColor: '#ddd', borderWidth: 1, borderRadius: 6, padding: 10, marginBottom: 10 },
  error: { color: '#b00020' },
});

