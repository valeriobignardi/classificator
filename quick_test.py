#!/usr/bin/env python3

import sys
sys.path.append('Models')

from documento_processing import DocumentoProcessing

# Test rapido
doc = DocumentoProcessing(
    session_id="test_123",
    testo_completo="Test conversazione"
)

print(f"✅ Oggetto creato: {doc}")
print(f"📝 Session ID: {doc.session_id}")
print(f"📊 Tipo iniziale: {doc.get_document_type()}")

# Test clustering
doc.set_clustering_info(cluster_id=5, cluster_size=10, is_outlier=False)
print(f"🎯 Dopo clustering: Cluster {doc.cluster_id}")

# Test rappresentante
doc.set_as_representative("test_selection")
print(f"👥 Tipo dopo representative: {doc.get_document_type()}")

print("🎉 Classe DocumentoProcessing funziona correttamente!")
