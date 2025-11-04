#!/usr/bin/env python3
"""
Embedding Store centralizzato su MongoDB

Scopo:
- Salvare gli embeddings generati in batch (multi-tenant aware)
- Recuperarli al momento del salvataggio della classificazione per evitare ricalcolo
- Opzionalmente consumarli (cancellazione) dopo l'uso e/o TTL automatico

Struttura documento (collection: embedding_cache):
{
  tenant_id: str,
  session_id: str,
  embedding: list[float],  # vettore embedding
  embedding_model: str,    # nome modello embedder
  created_at: datetime,
  expires_at: datetime     # usato dal TTL index (opzionale)
}

Indice unico: (tenant_id, session_id)
Indice TTL su expires_at (se ttl_seconds > 0)

Ultima modifica: 2025-11-04
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.collection import Collection
    from pymongo.errors import DuplicateKeyError
except ImportError as e:
    raise RuntimeError(f"pymongo non disponibile: {e}")


class EmbeddingStore:
    """
    Store centralizzato per embeddings su MongoDB
    """

    def __init__(
        self,
        mongodb_url: str = None,
        database_name: str = None,
        collection_name: str = "embedding_cache",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.mongodb_url = mongodb_url or os.getenv("MONGODB_URL", "mongodb://localhost:27017/classificazioni")
        # Consente sia URL con database incluso che separato
        if "/" in self.mongodb_url.rsplit("/", 1)[-1]:
            # URL include già il db; usa database_name solo se fornito
            self.database_name = database_name or self.mongodb_url.rsplit("/", 1)[-1]
        else:
            self.database_name = database_name or "classificazioni"
        self.collection_name = collection_name

        # TTL predefinito: 7 giorni (configurabile via ENV)
        if ttl_seconds is None:
            ttl_env = os.getenv("EMBEDDING_CACHE_TTL_SECONDS")
            self.ttl_seconds = int(ttl_env) if ttl_env else 7 * 24 * 3600
        else:
            self.ttl_seconds = ttl_seconds

        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self._connect()
        self._ensure_indexes()

    def _connect(self) -> None:
        # Se l'URL contiene già il database, MongoClient lo userà; altrimenti selezioniamo più sotto
        self.client = MongoClient(self.mongodb_url, serverSelectionTimeoutMS=5000)
        # Se l'URL include db, prendi quello; altrimenti usa self.database_name
        if "/" in self.mongodb_url.rsplit("/", 1)[-1]:
            db_name = self.mongodb_url.rsplit("/", 1)[-1]
        else:
            db_name = self.database_name
        self.collection = self.client[db_name][self.collection_name]
        # Ping per validare connessione
        self.client.admin.command('ping')

    def _ensure_indexes(self) -> None:
        # Unico su (tenant_id, session_id)
        try:
            self.collection.create_index(
                [("tenant_id", ASCENDING), ("session_id", ASCENDING)],
                name="uniq_tenant_session",
                unique=True,
                background=True,
            )
        except Exception:
            pass
        # TTL su expires_at (solo se ttl_seconds > 0)
        try:
            if self.ttl_seconds and self.ttl_seconds > 0:
                # Nota: TTL index richiede un campo Date con timezone naïve/UTC
                self.collection.create_index(
                    [("expires_at", ASCENDING)],
                    name="ttl_expires_at",
                    expireAfterSeconds=0,
                    background=True,
                )
        except Exception:
            pass

    def save_embeddings(
        self,
        tenant_id: str,
        session_ids: List[str],
        embeddings,  # np.ndarray o list[list[float]]
        embedding_model: str,
    ) -> Tuple[int, int]:
        """Salva embeddings in modo idempotente (upsert). Ritorna (inserted/updated, skipped)."""
        if embeddings is None or len(session_ids) == 0:
            return (0, 0)

        # Normalizza embeddings in lista di liste
        try:
            import numpy as np
            if isinstance(embeddings, np.ndarray):
                emb_list = embeddings.tolist()
            else:
                emb_list = embeddings
        except Exception:
            emb_list = embeddings

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self.ttl_seconds) if self.ttl_seconds and self.ttl_seconds > 0 else None

        inserted_or_updated = 0
        skipped = 0
        for i, sid in enumerate(session_ids):
            if i >= len(emb_list):
                skipped += 1
                continue
            doc = {
                "tenant_id": tenant_id,
                "session_id": sid,
                "embedding": emb_list[i],
                "embedding_model": embedding_model,
                "created_at": now,
            }
            if expires_at:
                doc["expires_at"] = expires_at

            # Upsert per non duplicare
            self.collection.update_one(
                {"tenant_id": tenant_id, "session_id": sid},
                {"$set": doc},
                upsert=True,
            )
            inserted_or_updated += 1

        return (inserted_or_updated, skipped)

    def get_embedding(
        self,
        tenant_id: str,
        session_id: str,
        consume: bool = True,
    ) -> Optional[Tuple[list, str]]:
        """Recupera (embedding, embedding_model). Se consume=True, cancella il record dopo la lettura."""
        doc = self.collection.find_one({"tenant_id": tenant_id, "session_id": session_id})
        if not doc:
            return None
        embedding = doc.get("embedding")
        model = doc.get("embedding_model", "unknown_embedder")
        if consume:
            try:
                self.collection.delete_one({"_id": doc["_id"]})
            except Exception:
                # Best effort: in caso di errore lascia al TTL l'eliminazione automatica
                pass
        return (embedding, model)

