"""
Autore: Valerio Bignardi
Data creazione: 2025-09-05
Storia aggiornamenti:
  - 2025-09-05: Creazione iniziale del tool di analisi.

Descrizione del file:
  Questo file contiene una sola classe, `ReviewQueueLLMNAAnalyzer`, che
  consente di analizzare i documenti nella Review Queue su MongoDB per
  individuare i casi con predizione LLM mancante (N/A). Lo script legge
  la configurazione da `config.yaml`, calcola i conteggi per categorie
  (rappresentanti, propagati, outlier) e fornisce campioni di ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from pymongo import MongoClient


@dataclass
class AnalyzerConfig:
    """
    Scopo:
      Struttura dati per i parametri di configurazione dell'analizzatore.

    Parametri di input:
      - mongo_url: URL di connessione a MongoDB.
      - database: Nome del database MongoDB.
      - tenant_slug: Slug del tenant (es. "humanitas").
      - tenant_id: UUID del tenant (stringa). Se non presente, opzionale.
      - collection: Nome della collection completa, se già nota.
      - sample_limit: Numero di ID da campionare per categoria.

    Valori di ritorno:
      N/A (è una struttura dati).

    Tracciamento aggiornamenti:
      - Ultima modifica: 2025-09-05
    """

    mongo_url: str
    database: str
    tenant_slug: str
    tenant_id: Optional[str] = None
    collection: Optional[str] = None
    sample_limit: int = 10


class ReviewQueueLLMNAAnalyzer:
    """
    Scopo della classe:
      Eseguire analisi sui documenti della Review Queue per identificare i
      casi con LLM mancante (N/A), suddivisi per rappresentanti, propagati
      e outlier.

    Input:
      - config: Istanza `AnalyzerConfig` con parametri e connessioni.

    Output:
      - Metodi che restituiscono dizionari con conteggi e campioni.

    Errori gestiti:
      - Eccezioni di connessione MongoDB.
      - Mancanza di collection dedotta da tenant_slug/tenant_id.

    Ultima modifica: 2025-09-05
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.client: Optional[MongoClient] = None
        self.collection = None

    # ---------------------------------------------------------------
    # Metodi di inizializzazione/connessione
    # ---------------------------------------------------------------
    def connect(self) -> None:
        """
        Scopo:
          Crea la connessione a MongoDB e risolve la collection.

        Parametri:
          Nessuno (usa `self.config`).

        Ritorno:
          None. Inizializza `self.collection`.

        Ultima modifica: 2025-09-05
        """

        self.client = MongoClient(self.config.mongo_url)
        db = self.client[self.config.database]

        if self.config.collection:
            col_name = self.config.collection
        else:
            if not self.config.tenant_slug and not self.config.tenant_id:
                raise ValueError("Tenant slug o id richiesto per dedurre la "
                                 "collection.")
            if self.config.tenant_id:
                col_name = f"{self.config.tenant_slug}_{self.config.tenant_id}"
            else:
                # Prova a dedurre dalla lista collections (fallback).
                prefix = f"{self.config.tenant_slug}_"
                candidates = [
                    name for name in db.list_collection_names()
                    if name.startswith(prefix)
                ]
                if not candidates:
                    raise ValueError("Impossibile dedurre la collection: "
                                     f"nessuna corrispondenza per '{prefix}'.")
                # Scegli la più recente per nome (euristica semplice).
                col_name = sorted(candidates)[-1]

        self.collection = db[col_name]

    # ---------------------------------------------------------------
    # Query builders
    # ---------------------------------------------------------------
    def _missing_llm_query(self) -> Dict[str, Any]:
        """
        Scopo:
          Costruisce il filtro per "LLM mancante/vuoto".

        Parametri:
          Nessuno.

        Ritorno:
          Dizionario query MongoDB per identificare llm_prediction mancante.

        Ultima modifica: 2025-09-05
        """

        return {
            "$or": [
                {"llm_prediction": {"$exists": False}},
                {"llm_prediction": ""},
                {"llm_prediction": None},
            ]
        }

    def _pending_query(self) -> Dict[str, Any]:
        """
        Scopo:
          Restituisce il filtro comune per review_status = 'pending'.

        Ritorno:
          Dizionario query MongoDB.

        Ultima modifica: 2025-09-05
        """

        return {"review_status": "pending"}

    # ---------------------------------------------------------------
    # Analisi principali
    # ---------------------------------------------------------------
    def compute_counts(self) -> Dict[str, Any]:
        """
        Scopo:
          Calcola i conteggi dei casi pending con LLM mancante, segmentati
          per rappresentanti, propagati e outlier.

        Parametri:
          Nessuno.

        Ritorno:
          Dizionario con conteggi e nomi collection.

        Ultima modifica: 2025-09-05
        """

        if self.collection is None:
            raise RuntimeError("Connessione non inizializzata. Chiama connect().")

        missing = self._missing_llm_query()
        pending = self._pending_query()

        q_rep = {**pending, "metadata.representative": True, **missing}
        q_prop = {**pending, "metadata.propagated": True, **missing}
        q_out = {**pending, "metadata.outlier": True, **missing}

        counts = {
            "collection": self.collection.name,
            "pending_total": self.collection.count_documents(pending),
            "rep_pending_llm_NA": self.collection.count_documents(q_rep),
            "prop_pending_llm_NA": self.collection.count_documents(q_prop),
            "out_pending_llm_NA": self.collection.count_documents(q_out),
        }
        return counts

    def sample_ids(self) -> Dict[str, List[str]]:
        """
        Scopo:
          Estrae un campione di ID per categoria, per ispezione manuale.

        Parametri:
          None. Usa `self.config.sample_limit`.

        Ritorno:
          Dizionario con liste di ID (stringhe) per categoria.

        Ultima modifica: 2025-09-05
        """

        if self.collection is None:
            raise RuntimeError("Connessione non inizializzata. Chiama connect().")

        missing = self._missing_llm_query()
        pending = self._pending_query()
        limit = int(self.config.sample_limit)

        q_rep = {**pending, "metadata.representative": True, **missing}
        q_prop = {**pending, "metadata.propagated": True, **missing}
        q_out = {**pending, "metadata.outlier": True, **missing}

        rep = [str(x.get("_id")) for x in
               self.collection.find(q_rep, {"_id": 1}).limit(limit)]
        prop = [str(x.get("_id")) for x in
                self.collection.find(q_prop, {"_id": 1}).limit(limit)]
        outl = [str(x.get("_id")) for x in
                self.collection.find(q_out, {"_id": 1}).limit(limit)]

        return {"rep_ids": rep, "prop_ids": prop, "out_ids": outl}

    def analyze_representatives_details(self) -> Dict[str, Any]:
        """
        Scopo:
          Analizza i dettagli dei rappresentanti con LLM mancante per
          capire distribuzione cluster, presenza final_decision, ecc.

        Parametri:
          Nessuno.

        Ritorno:
          Dizionario con statistiche dettagliate sui rappresentanti N/A.

        Ultima modifica: 2025-09-05
        """

        if self.collection is None:
            raise RuntimeError("Connessione non inizializzata. Chiama connect().")

        missing = self._missing_llm_query()
        pending = self._pending_query()
        q_rep = {**pending, "metadata.representative": True, **missing}

        # Estrai dettagli rappresentanti
        projection = {
            "_id": 1, "session_id": 1, "metadata.cluster_id": 1,
            "classification": 1, "predicted_label": 1, "confidence": 1,
            "review_reason": 1, "classified_by": 1,
            "llm_prediction": 1, "ml_prediction": 1,
            "llm_confidence": 1, "ml_confidence": 1
        }
        
        cursor = self.collection.find(q_rep, projection)
        details = list(cursor)

        # Analisi statistiche
        cluster_counts = {}
        has_final_decision = 0
        has_any_prediction = 0
        review_reasons = {}
        classified_by_counts = {}

        for doc in details:
            # Conta per cluster_id
            cluster_id = doc.get("metadata", {}).get("cluster_id", "unknown")
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

            # Verifica presenza final_decision
            if (doc.get("classification") or doc.get("predicted_label")):
                has_final_decision += 1

            # Verifica presenza di almeno una predizione
            if (doc.get("ml_prediction") or doc.get("llm_prediction") or
                doc.get("ml_confidence", 0) > 0 or doc.get("llm_confidence", 0) > 0):
                has_any_prediction += 1

            # Conta review_reason
            reason = doc.get("review_reason", "not_specified")
            review_reasons[reason] = review_reasons.get(reason, 0) + 1

            # Conta classified_by
            by = doc.get("classified_by", "not_specified")
            classified_by_counts[by] = classified_by_counts.get(by, 0) + 1

        total = len(details)
        
        return {
            "total_representatives_llm_na": total,
            "cluster_distribution": cluster_counts,
            "top_clusters": dict(sorted(cluster_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]),
            "has_final_decision": has_final_decision,
            "has_final_decision_pct": (has_final_decision / total * 100) if total else 0,
            "has_any_prediction": has_any_prediction,
            "has_any_prediction_pct": (has_any_prediction / total * 100) if total else 0,
            "review_reasons": review_reasons,
            "classified_by_counts": classified_by_counts,
            "sample_docs": [
                {
                    "id": str(doc.get("_id")),
                    "session_id": doc.get("session_id"),
                    "cluster_id": doc.get("metadata", {}).get("cluster_id"),
                    "has_classification": bool(doc.get("classification") or doc.get("predicted_label")),
                    "review_reason": doc.get("review_reason"),
                    "classified_by": doc.get("classified_by")
                } for doc in details[:5]
            ]
        }

    def run(self) -> Dict[str, Any]:
        """
        Scopo:
          Esegue l'analisi completa: connessione, conteggi, campioni e dettagli.

        Parametri:
          Nessuno.

        Ritorno:
          Dizionario completo con risultati; stampa anche JSON leggibile.

        Ultima modifica: 2025-09-05
        """

        self.connect()
        counts = self.compute_counts()
        samples = self.sample_ids()
        rep_details = self.analyze_representatives_details()
        
        result = {
            "counts": counts, 
            "samples": samples,
            "representatives_analysis": rep_details,
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=2))
        return result


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Scopo:
      Carica il file YAML di configurazione e restituisce il dict.

    Parametri:
      - path: Percorso al file YAML (es. config.yaml).

    Ritorno:
      Dizionario con la configurazione.

    Ultima modifica: 2025-09-05
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_analyzer_config(args: argparse.Namespace) -> AnalyzerConfig:
    """
    Scopo:
      Compone `AnalyzerConfig` unendo CLI args e `config.yaml`.

    Parametri:
      - args: Argomenti CLI parsati.

    Ritorno:
      Istanza `AnalyzerConfig` pronta per l'uso.

    Ultima modifica: 2025-09-05
    """

    cfg = _load_yaml_config(args.config)
    mongo = cfg.get("mongodb", {})

    return AnalyzerConfig(
        mongo_url=mongo.get("url", "mongodb://localhost:27017"),
        database=mongo.get("database", "classificazioni"),
        tenant_slug=args.tenant_slug,
        tenant_id=args.tenant_id,
        collection=args.collection,
        sample_limit=args.sample_limit,
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Scopo:
      Definisce e parse gli argomenti da linea di comando.

    Parametri:
      - argv: Lista degli argomenti (tipicamente sys.argv[1:]).

    Ritorno:
      Oggetto argparse.Namespace con i valori parsati.

    Ultima modifica: 2025-09-05
    """

    p = argparse.ArgumentParser(
        description=(
            "Analizza la Review Queue e conta i casi con LLM=N/A per "
            "rappresentanti/propagati/outlier."
        )
    )
    p.add_argument("--config", default="config.yaml",
                   help="Percorso a config.yaml")
    p.add_argument("--tenant-slug", default="humanitas",
                   help="Slug del tenant (es. humanitas)")
    p.add_argument("--tenant-id", default=None,
                   help="UUID del tenant")
    p.add_argument("--collection", default=None,
                   help="Collection completa (se già nota)")
    p.add_argument("--sample-limit", type=int, default=10,
                   help="Numero di ID campione per categoria")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Scopo:
      Entry-point CLI per lanciare l'analisi da terminale.

    Parametri:
      - argv: Argomenti da linea di comando; se None usa sys.argv[1:].

    Ritorno:
      Exit code (0 successo, >0 errore).

    Ultima modifica: 2025-09-05
    """

    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        config = _build_analyzer_config(args)
        analyzer = ReviewQueueLLMNAAnalyzer(config)
        analyzer.run()
        return 0
    except Exception as exc:
        err = {
            "error": str(exc),
            "type": exc.__class__.__name__,
            "when": datetime.now().isoformat(),
        }
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
