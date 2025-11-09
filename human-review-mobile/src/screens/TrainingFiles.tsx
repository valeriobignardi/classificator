import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, FlatList, TouchableOpacity, Modal, ScrollView, Button } from 'react-native';
import { api } from '../services/api';
import { useTenant } from '../contexts/TenantContext';

export default function TrainingFiles() {
  const { selectedTenant } = useTenant();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<Array<{ name: string; size: number; modified_at: string }>>([]);
  const [preview, setPreview] = useState<{ file: string; content: string } | null>(null);

  const load = async () => {
    if (!selectedTenant) return;
    setLoading(true); setError(null);
    try {
      const res = await api.listTrainingFiles(selectedTenant.tenant_id);
      setFiles(res.files || []);
    } catch (e: any) {
      setError(e?.message || 'Errore caricamento file di training');
    } finally {
      setLoading(false);
    }
  };

  const open = async (file: string) => {
    if (!selectedTenant) return;
    setLoading(true);
    try {
      const res = await api.getTrainingFileContent(selectedTenant.tenant_id, file, 500);
      setPreview({ file, content: res.content || '' });
    } catch (e: any) {
      setError(e?.message || 'Errore apertura file');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [selectedTenant?.tenant_id]);

  return (
    <View style={styles.container}>
      {loading && <ActivityIndicator />}
      {error && <Text style={styles.error}>{error}</Text>}
      <FlatList
        data={files}
        keyExtractor={(i) => i.name}
        contentContainerStyle={{ padding: 12 }}
        renderItem={({ item }) => (
          <TouchableOpacity style={styles.item} onPress={() => open(item.name)}>
            <Text style={styles.name}>{item.name}</Text>
            <Text style={styles.meta}>{Math.round(item.size/1024)} KB • {item.modified_at}</Text>
          </TouchableOpacity>
        )}
      />

      <Modal visible={!!preview} animationType="slide" onRequestClose={() => setPreview(null)}>
        <View style={{ flex: 1 }}>
          <View style={{ padding: 12, backgroundColor: '#fff', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ fontWeight: '700' }}>{preview?.file}</Text>
            <Button title="Chiudi" onPress={() => setPreview(null)} />
          </View>
          <ScrollView style={{ flex: 1, backgroundColor: '#fafafa' }} contentContainerStyle={{ padding: 12 }}>
            <Text style={{ fontFamily: 'monospace' }}>{preview?.content}</Text>
          </ScrollView>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fafafa' },
  item: { backgroundColor: '#fff', padding: 12, marginBottom: 10, borderRadius: 8, elevation: 1 },
  name: { fontWeight: '700' },
  meta: { color: '#555', marginTop: 4 },
  error: { color: '#b00020', paddingHorizontal: 12 },
});

