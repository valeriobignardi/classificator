import json
import numpy as np

from Pipeline.end_to_end_pipeline import EndToEndPipeline


class FakeAggregator:
    def __init__(self):
        self._store = {}

    def estrai_sessioni_aggregate(self, limit=None):
        texts = [
            "Devo prenotare una visita cardiologica domani",
            "Non riesco ad accedere al portale, errore di login",
            "Come posso scaricare il referto dell'esame del sangue?",
            "Vorrei cambiare l'appuntamento fissato per lunedì",
            "Quali sono gli orari di apertura dell'ambulatorio?",
            "Problema con il pagamento della fattura online",
            "Desidero parlare con un operatore umano",
            "Dove posso parcheggiare vicino all'ospedale?",
            "Non funziona l'app, si blocca all'avvio",
            "Informazioni generali sui servizi disponibili",
        ]
        if limit:
            texts = texts[:limit]
        d = {f"S{i+1:04d}": {"testo_completo": t} for i, t in enumerate(texts)}
        self._store = d
        return d

    def filtra_sessioni_vuote(self, sessioni):
        return {k: v for k, v in sessioni.items() if v.get("testo_completo", "").strip()}

    def get_session_by_id(self, session_id):
        return self._store.get(session_id)

    def chiudi_connessione(self):
        pass


class FakeTagDB:
    def __init__(self):
        self.rows = []

    def connetti(self):
        return True

    def disconnetti(self):
        return True

    def classifica_sessione(self, session_id, tag_name, tenant_slug, confidence_score, method, classified_by, notes):
        self.rows.append({
            'session_id': session_id,
            'tag_name': tag_name,
            'confidence_score': confidence_score
        })
        return True

    def get_classificazioni_by_session_ids(self, ids):
        return [r for r in self.rows if r['session_id'] in ids]

    def esegui_query(self, q, params=None):
        return []

    def esegui_update(self, q, params=None):
        return True

    def get_statistiche_classificazioni(self):
        return {'total_classificazioni': len(self.rows), 'per_tag': []}


class FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        rs = np.random.RandomState(42)
        return rs.rand(len(texts), 384).astype('float32')


def main():
    print('>>> Avvio pipeline sintetica…')
    pipe = EndToEndPipeline(tenant_slug='humanitas', auto_mode=False)

    # Disabilita LLM per rapidità
    if hasattr(pipe.ensemble_classifier, 'llm_classifier'):
        pipe.ensemble_classifier.llm_classifier = None

    # Monkeypatch componenti esterni e embedder
    pipe.aggregator = FakeAggregator()
    pipe.tag_db = FakeTagDB()
    pipe.embedder = FakeEmbedder()
    if hasattr(pipe, 'semantic_memory') and hasattr(pipe.semantic_memory, 'embedder'):
        pipe.semantic_memory.embedder = pipe.embedder

    # Disabilita BERTopic per smoke rapido
    if hasattr(pipe, 'bertopic_config'):
        pipe.bertopic_config['enabled'] = False

    print('>>> Eseguo pipeline end-to-end sintetica…')
    res = pipe.esegui_pipeline_completa(
        giorni_indietro=7,
        limit=10,
        batch_size=8,
        interactive_mode=False,
        use_ensemble=True,
    )

    print('>>> Risultato sintetico:')
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
