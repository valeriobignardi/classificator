import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, ScrollView, TextInput, Button, Alert } from 'react-native';
import { useRoute } from '@react-navigation/native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';
import { ReviewCase } from '../types/ReviewCase';

export default function CaseDetail() {
  const route = useRoute<any>();
  const caseId = route.params?.caseId as string;
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<ReviewCase | null>(null);
  const [humanDecision, setHumanDecision] = useState('');
  const [confidence, setConfidence] = useState('0.9');
  const [notes, setNotes] = useState('');

  const load = async () => {
    if (!selectedTenant || !caseId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getCaseDetail(selectedTenant.tenant_id, caseId);
      setDetails(res.case);
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento caso');
    } finally {
      setLoading(false);
    }
  };

  const resolve = async () => {
    if (!selectedTenant || !caseId) return;
    try {
      setLoading(true);
      await api.resolveCase(selectedTenant.tenant_id, caseId, humanDecision, Number(confidence) || 0.9, notes);
      Alert.alert('OK', 'Caso risolto');
    } catch (e: any) {
      Alert.alert('Errore', e?.message || 'Impossibile risolvere il caso');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id, caseId]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 12 }}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      {details && (
        <View>
          <Text style={styles.title}>Classificazione: {details.classification}</Text>
          <Text style={styles.meta}>Case ID: {details.case_id}</Text>
          <Text style={styles.meta}>Cluster: {details.cluster_id ?? '-'}</Text>
          <Text style={styles.meta}>ML: {details.ml_prediction} ({Math.round(details.ml_confidence*100)}%)</Text>
          <Text style={styles.meta}>LLM: {details.llm_prediction} ({Math.round(details.llm_confidence*100)}%)</Text>
          <Text style={styles.body}>{details.conversation_text}</Text>
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.subtitle}>Decisione Umana</Text>
        <TextInput placeholder="Etichetta (es: prenotazione_esami)" style={styles.input} value={humanDecision} onChangeText={setHumanDecision} />
        <TextInput placeholder="Confidenza (0-1)" style={styles.input} value={confidence} onChangeText={setConfidence} keyboardType="decimal-pad" />
        <TextInput placeholder="Note" style={[styles.input, styles.notes]} value={notes} onChangeText={setNotes} multiline />
        <Button title="Conferma" onPress={resolve} />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  title: { fontSize: 18, fontWeight: '700', marginBottom: 6 },
  subtitle: { fontSize: 16, fontWeight: '600', marginBottom: 8 },
  meta: { color: '#555', marginBottom: 4 },
  body: { marginTop: 10, color: '#333' },
  card: { backgroundColor: '#fff', padding: 12, borderRadius: 8, elevation: 1, marginTop: 16 },
  input: { backgroundColor: '#fff', borderColor: '#ddd', borderWidth: 1, borderRadius: 6, padding: 10, marginBottom: 10 },
  notes: { minHeight: 80, textAlignVertical: 'top' },
  error: { color: '#b00020' },
});

