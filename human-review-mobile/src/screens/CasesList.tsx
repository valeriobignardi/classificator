import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, FlatList, Switch, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';
import { ReviewCase } from '../types/ReviewCase';

export default function CasesList() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [includeRepresentatives, setIncludeRepresentatives] = useState(true);
  const [includePropagated, setIncludePropagated] = useState(true);
  const [includeOutliers, setIncludeOutliers] = useState(true);
  const nav = useNavigation<any>();

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getReviewCases(selectedTenant.tenant_id, {
        limit: 50,
        include_representatives: includeRepresentatives,
        include_propagated: includePropagated,
        include_outliers: includeOutliers,
      });
      setCases(res.cases || []);
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento casi');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id, includeRepresentatives, includePropagated, includeOutliers]);

  const renderItem = ({ item }: { item: ReviewCase }) => (
    <TouchableOpacity style={styles.item} onPress={() => nav.navigate('CaseDetail', { caseId: item.case_id })}>
      <Text style={styles.tag}>{item.classification}</Text>
      <Text numberOfLines={3} style={styles.text}>{item.conversation_text}</Text>
      <View style={styles.metaRow}>
        {item.is_representative && <Text style={styles.badge}>REP</Text>}
        {item.propagated_from && <Text style={styles.badge}>PROP</Text>}
        {item.cluster_id && <Text style={styles.cluster}>#{item.cluster_id}</Text>}
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={styles.filters}>
        <View style={styles.filterRow}><Text>Rappresentanti</Text><Switch value={includeRepresentatives} onValueChange={setIncludeRepresentatives} /></View>
        <View style={styles.filterRow}><Text>Propagati</Text><Switch value={includePropagated} onValueChange={setIncludePropagated} /></View>
        <View style={styles.filterRow}><Text>Outliers</Text><Switch value={includeOutliers} onValueChange={setIncludeOutliers} /></View>
      </View>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      <FlatList
        data={cases}
        keyExtractor={(i) => i.case_id}
        renderItem={renderItem}
        contentContainerStyle={{ padding: 12 }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  filters: { backgroundColor: '#fff', padding: 12, marginBottom: 8 },
  filterRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  item: { backgroundColor: '#fff', padding: 12, marginBottom: 10, borderRadius: 8, elevation: 1 },
  text: { color: '#333', marginTop: 6 },
  tag: { fontWeight: '700', color: '#1e88e5' },
  metaRow: { flexDirection: 'row', gap: 8, marginTop: 8 },
  badge: { backgroundColor: '#eee', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 10, overflow: 'hidden' },
  cluster: { marginLeft: 'auto', color: '#555' },
  error: { color: '#b00020', paddingHorizontal: 12 },
});

