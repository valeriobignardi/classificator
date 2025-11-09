import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Button } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';
import { ReviewStats } from '../types/ReviewCase';
import TenantSelector from '../components/TenantSelector';

export default function ReviewDashboard() {
  const { selectedTenant } = useTenant();
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nav = useNavigation<any>();

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getReviewStats(selectedTenant.tenant_id);
      setStats(res.stats);
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento statistiche');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  return (
    <View style={styles.container}>
      <TenantSelector />
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      {stats && (
        <View style={styles.card}>
          <Text style={styles.title}>Review Queue</Text>
          <Text>In coda: {stats.review_queue.pending_cases}</Text>
          <Text>Capacità: {stats.review_queue.total_capacity}</Text>
          <Text>Utilizzo: {Math.round(stats.review_queue.queue_utilization * 100)}%</Text>
        </View>
      )}
      <View style={styles.row}>
        <Button title="Vedi Casi" onPress={() => nav.navigate('CasesList')} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  card: { backgroundColor: '#fff', padding: 16, margin: 12, borderRadius: 8, elevation: 1 },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 8 },
  error: { color: '#b00020', paddingHorizontal: 12 },
  row: { paddingHorizontal: 12, marginTop: 8 },
});

