import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTenant } from '../contexts/TenantContext';

export default function TenantSelector() {
  const { availableTenants, selectedTenant, setSelectedTenant } = useTenant();

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Tenant:</Text>
      <View style={styles.row}>
        {availableTenants.map((t) => (
          <TouchableOpacity
            key={t.tenant_id}
            style={[styles.pill, selectedTenant?.tenant_id === t.tenant_id && styles.pillActive]}
            onPress={() => setSelectedTenant(t)}
          >
            <Text style={[styles.pillText, selectedTenant?.tenant_id === t.tenant_id && styles.pillTextActive]}>
              {t.tenant_name}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 12, backgroundColor: '#fff' },
  label: { fontWeight: '600', marginBottom: 6, fontSize: 16 },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  pill: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 16, backgroundColor: '#eee', marginRight: 8, marginBottom: 8 },
  pillActive: { backgroundColor: '#1e88e5' },
  pillText: { color: '#333' },
  pillTextActive: { color: '#fff' },
});

