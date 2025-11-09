import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, TextInput, Button, Alert, Switch } from 'react-native';
import { useTenant } from '../contexts/TenantContext';
import axios from 'axios';
import { API_BASE_URL } from '../config';

export default function ReviewQueueThresholds() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<any>({});
  const [clustering, setClustering] = useState<any>({});

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true); setError(null);
    try {
      const res = await axios.get(`${API_BASE_URL}/review-queue/${selectedTenant.tenant_id}/thresholds`);
      if (res.data?.success) {
        setThresholds(res.data.thresholds || {});
        setClustering(res.data.clustering_parameters || {});
      } else {
        throw new Error(res.data?.error || 'API error');
      }
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento soglie');
    } finally {
      setLoading(false);
    }
  };

  const save = async () => {
    if (!selectedTenant) return;
    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE_URL}/review-queue/${selectedTenant.tenant_id}/thresholds`, {
        thresholds,
        clustering_parameters: clustering,
      });
      if (!res.data?.success) throw new Error(res.data?.error || 'Salvataggio fallito');
      Alert.alert('OK', 'Soglie aggiornate');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Salvataggio fallito');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  const setNumT = (k: string, v: string) => setThresholds((p: any) => ({ ...p, [k]: Number(v) || 0 }));
  const setBoolT = (k: string, v: boolean) => setThresholds((p: any) => ({ ...p, [k]: v }));

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 12 }}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      <Text style={styles.title}>Soglie Review Queue</Text>
      <View style={styles.card}>
        <Text>enable_smart_review</Text>
        <Switch value={!!thresholds.enable_smart_review} onValueChange={(v) => setBoolT('enable_smart_review', v)} />
        <Text>max_pending_per_batch</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(thresholds.max_pending_per_batch ?? '')} onChangeText={(v) => setNumT('max_pending_per_batch', v)} />
        <Text>minimum_consensus_threshold</Text>
        <TextInput style={styles.input} keyboardType="number-pad" value={String(thresholds.minimum_consensus_threshold ?? '')} onChangeText={(v) => setNumT('minimum_consensus_threshold', v)} />
        <Text>outlier_confidence_threshold</Text>
        <TextInput style={styles.input} keyboardType="decimal-pad" value={String(thresholds.outlier_confidence_threshold ?? '')} onChangeText={(v) => setNumT('outlier_confidence_threshold', v)} />
        <Text>propagated_confidence_threshold</Text>
        <TextInput style={styles.input} keyboardType="decimal-pad" value={String(thresholds.propagated_confidence_threshold ?? '')} onChangeText={(v) => setNumT('propagated_confidence_threshold', v)} />
        <Text>representative_confidence_threshold</Text>
        <TextInput style={styles.input} keyboardType="decimal-pad" value={String(thresholds.representative_confidence_threshold ?? '')} onChangeText={(v) => setNumT('representative_confidence_threshold', v)} />
        <Button title="Salva" onPress={save} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 6 },
  card: { backgroundColor: '#fff', padding: 12, borderRadius: 8, elevation: 1, marginBottom: 12 },
  input: { backgroundColor: '#fff', borderColor: '#ddd', borderWidth: 1, borderRadius: 6, padding: 10, marginBottom: 10 },
  error: { color: '#b00020' },
});

