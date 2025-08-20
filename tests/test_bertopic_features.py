"""
Autore: GitHub Copilot
Creato: 2025-08-07
Storia aggiornamenti:
- 2025-08-07: Smoke test iniziale per BERTopicFeatureProvider.

Scopo:
Verifica che la classe si importi e gestisca gracefully l'assenza
 delle dipendenze, e che le forme di output siano coerenti.
"""

import numpy as np
import pytest

from TopicModeling.bertopic_feature_provider import BERTopicFeatureProvider


def test_import_and_availability():
    provider = BERTopicFeatureProvider()
    assert isinstance(provider.is_available(), bool)


def test_transform_shapes_without_fit():
    provider = BERTopicFeatureProvider()
    if not provider.is_available():
        pytest.skip("BERTopic non disponibile nell'ambiente di test")

    texts = ["ciao", "prenotare una visita", "referto pronto"]
    embs = np.random.rand(len(texts), 768).astype(np.float32)

    # Fit e transform
    provider.fit(texts, embs)
    out = provider.transform(texts, embs, return_one_hot=True, top_k=5)

    assert "topic_ids" in out and "topic_probas" in out and "one_hot" in out
    assert out["topic_ids"].shape[0] == len(texts)
    assert out["topic_probas"].shape[0] == len(texts)
    assert out["one_hot"].shape[0] == len(texts)
