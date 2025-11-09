import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, TextInput, Button, Alert } from 'react-native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';

export default function ClusteringParameters() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useState<any>({});

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getClusteringParameters(selectedTenant.tenant_id);
      setParams(res.parameters || {});
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento parametri');
    } finally {
      setLoading(false);
    }
  };

  const save = async () => {
    if (!selectedTenant) return;
    try {
      setLoading(true);
      await api.updateClusteringParameters(selectedTenant.tenant_id, params);
      Alert.alert('OK', 'Parametri aggiornati');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Salvataggio fallito');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  const setNum = (key: string, v: string) => setParams((p: any) => ({ ...p, [key]: Number(v) || 0 }));

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 12 }}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      <Text style={styles.title}>Parametri Clustering</Text>
      <View style={styles.card}>
        <Text>min_cluster_size</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(params.min_cluster_size ?? '')} onChangeText={(v) => setNum('min_cluster_size', v)} />
        <Text>min_samples</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(params.min_samples ?? '')} onChangeText={(v) => setNum('min_samples', v)} />
        <Text>metric</Text>
        <TextInput style={styles.input} value={String(params.metric ?? '')} onChangeText={(v) => setParams((p: any) => ({ ...p, metric: v }))} />
        <Button title="Salva" onPress={save} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 6 },
  card: { backgroundColor: '#fff', padding: 12, borderRadius: 8, elevation: 1 },
  input: { backgroundColor: '#fff', borderColor: '#ddd', borderWidth: 1, borderRadius: 6, padding: 10, marginBottom: 10 },
  error: { color: '#b00020' },
});

