import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, TextInput, Button, Alert } from 'react-native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';

export default function LLMConfiguration() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<any[]>([]);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [params, setParams] = useState<any>({});
  const [testPrompt, setTestPrompt] = useState('Scrivi "ok" se sei attivo.');

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true); setError(null);
    try {
      const m = await api.getLLMModels(selectedTenant.tenant_id);
      setModels(m.models || []);
      const p = await api.getLLMParameters(selectedTenant.tenant_id);
      setParams(p.parameters || {});
      setCurrentModel(p.current_model || '');
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento configurazione LLM');
    } finally {
      setLoading(false);
    }
  };

  const save = async () => {
    if (!selectedTenant) return;
    try {
      setLoading(true);
      await api.updateLLMParameters(selectedTenant.tenant_id, params, currentModel || undefined);
      Alert.alert('OK', 'Parametri LLM salvati');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Salvataggio parametri fallito');
    } finally {
      setLoading(false);
    }
  };

  const reset = async () => {
    if (!selectedTenant) return;
    try {
      setLoading(true);
      await api.resetLLMParameters(selectedTenant.tenant_id);
      await load();
      Alert.alert('OK', 'Parametri ripristinati');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Reset fallito');
    } finally {
      setLoading(false);
    }
  };

  const test = async () => {
    if (!selectedTenant || !currentModel) return;
    try {
      setLoading(true);
      const res = await api.testLLMModel(selectedTenant.tenant_id, currentModel, params, testPrompt);
      Alert.alert('Test', res.message || 'Test eseguito');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Test fallito');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 12 }}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}

      <Text style={styles.title}>Configurazione LLM</Text>
      <View style={styles.card}>
        <Text style={styles.subtitle}>Modello</Text>
        <TextInput value={currentModel} onChangeText={setCurrentModel} style={styles.input} placeholder="nome modello" />
        <Text style={{ color: '#555' }}>Modelli disponibili: {models.map(m => m.name || m).join(', ')}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.subtitle}>Parametri Principali</Text>
        <Text>max_tokens</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(params?.generation?.max_tokens ?? '')}
          onChangeText={v => setParams((p: any) => ({ ...p, generation: { ...(p.generation||{}), max_tokens: Number(v)||0 } }))} />
        <Text>temperature</Text>
        <TextInput style={styles.input} keyboardType="decimal-pad" value={String(params?.generation?.temperature ?? '')}
          onChangeText={v => setParams((p: any) => ({ ...p, generation: { ...(p.generation||{}), temperature: Number(v)||0 } }))} />
        <Text>top_p</Text>
        <TextInput style={styles.input} keyboardType="decimal-pad" value={String(params?.generation?.top_p ?? '')}
          onChangeText={v => setParams((p: any) => ({ ...p, generation: { ...(p.generation||{}), top_p: Number(v)||0 } }))} />
      </View>

      <View style={styles.card}>
        <Text style={styles.subtitle}>Test</Text>
        <TextInput style={styles.input} value={testPrompt} onChangeText={setTestPrompt} />
        <View style={{ flexDirection: 'row', gap: 12 }}>
          <Button title="Salva" onPress={save} />
          <Button title="Reset" onPress={reset} />
          <Button title="Test" onPress={test} />
        </View>
      </View>
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

