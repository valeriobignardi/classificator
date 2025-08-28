"""
Autore: GitHub Copilot
Creato: 2025-08-07
Storia aggiornamenti:
- 2025-08-07: Creazione iniziale del provider feature BERTopic.
- 2025-08-07: Aggiunta riduzione dimensionale opzionale (SVD) sulle full probas.

Descrizione:
Classe BERTopicFeatureProvider per addestrare un modello BERTopic sui testi
(con embeddings LaBSE coerenti con il progetto) e generare feature robuste
per l'ensemble ML: vettori di probabilità dei topic (topic-probas) e opzionale
one-hot del topic-id principale. Include persistenza (save/load) e metodi di
ispezione (top words e rappresentanti per topic) a supporto review/quality.

Note stile/sintassi:
- Una sola classe per file
- Header con autore/data/storia
- Linee <= 80 caratteri
- Commenti funzione con scopo, I/O, return, ultimo aggiornamento
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import time
import numpy as np

# Dipendenze opzionali per non rompere l'ambiente se mancanti
try:
    from bertopic import BERTopic
    import umap
    import hdbscan  # noqa: F401  # usato internamente da BERTopic
    BERTopic_AVAILABLE = True
except Exception:  # ImportError o altri errori runtime
    BERTopic_AVAILABLE = False

# Riduzione dimensionale opzionale (SVD) - import lazy
try:
    from sklearn.decomposition import TruncatedSVD
    _SVD_AVAILABLE = True
except Exception:
    TruncatedSVD = None
    _SVD_AVAILABLE = False


class BERTopicFeatureProvider:
    """
    Fornisce feature di topic (probas, id) tramite BERTopic.

    Uso tipico:
    - fit(texts, embeddings) su corpus storico
    - transform(texts, embeddings) per generare feature
    - save(path) / load(path) per persistenza

    Ultimo aggiornamento: 2025-08-07
    """
    
    def __init__(self,
                 random_state: int = 42,
                 calculate_probabilities: bool = True,
                 n_neighbors: int = 15,
                 min_dist: float = 0.0,
                 metric: str = "cosine",
                 hdbscan_min_cluster_size: int = 5,
                 hdbscan_min_samples: int = 3,
                 top_k_words: int = 10,
                 # Nuovi parametri SVD
                 use_svd: bool = False,
                 svd_components: int = 32,
                 # Nuovo parametro embedder per BERTopic transform
                 embedder: Optional[Any] = None,
                 # Nuovi parametri per configurazione completa da database
                 hdbscan_params: Optional[Dict[str, Any]] = None,
                 umap_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Inizializza provider con iperparametri riproducibili.

        Parametri:
        - random_state: seed per riproducibilità
        - calculate_probabilities: se calcolare p(topic|doc)
        - n_neighbors, min_dist, metric: UMAP (deprecati, usa umap_params)
        - hdbscan_min_cluster_size, hdbscan_min_samples: clustering (deprecati, usa hdbscan_params)
        - top_k_words: parole chiave per topic_info
        - use_svd: abilita riduzione SVD sulle full probas (default 32)
        - svd_components: dimensione target per SVD (default 32)
        - embedder: Embedder per BERTopic transform singoli (default None)
        - hdbscan_params: Dict completo parametri HDBSCAN da database (sovrascrive parametri individuali)
        - umap_params: Dict completo parametri UMAP da database (sovrascrive parametri individuali)
        
        Ultimo aggiornamento: 2025-08-28
        """
        self.available = BERTopic_AVAILABLE
        self.model: Optional[BERTopic] = None
        self.random_state = random_state
        self.calculate_probabilities = calculate_probabilities
        self.embedder = embedder  # Salva embedder per uso interno
        
        # 🆕 GESTIONE PARAMETRI UMAP
        # Se umap_params è fornito, usa quello (configurazione dal database)
        # Altrimenti usa parametri individuali (backward compatibility)
        if umap_params is not None:
            self.umap_params = umap_params.copy()
            # Assicurati che random_state sia impostato per riproducibilità
            if 'random_state' not in self.umap_params:
                self.umap_params['random_state'] = random_state
        else:
            # Backward compatibility: usa parametri individuali
            self.umap_params = {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "n_components": 5,
                "metric": metric,
                "random_state": random_state,
            }
        
        # 🆕 GESTIONE PARAMETRI HDBSCAN
        # Se hdbscan_params è fornito, usa quello (configurazione dal database)
        # Altrimenti usa parametri individuali (backward compatibility)
        if hdbscan_params is not None:
            self.hdbscan_params = hdbscan_params.copy()
        else:
            # Backward compatibility: usa parametri individuali
            self.hdbscan_params = {
                "min_cluster_size": hdbscan_min_cluster_size,
                "min_samples": hdbscan_min_samples,
                "metric": metric,
            }
        
        self.top_k_words = top_k_words
        # Stato SVD
        self.use_svd = use_svd and _SVD_AVAILABLE
        self.svd_components = svd_components
        # Usa Any per evitare dipendenze rigide sul tipo quando sklearn manca
        self._svd_model: Optional[Any] = None

    def is_available(self) -> bool:
        """
        Verifica disponibilità dipendenze BERTopic.

        Ritorno: True se import riuscito, False altrimenti.
        Ultimo aggiornamento: 2025-08-07
        """
        return self.available

    def fit(self,
            texts: List[str],
            embeddings: Optional[np.ndarray] = None) -> "BERTopicFeatureProvider":
        """
        Addestra BERTopic su un corpus. Usa embeddings LaBSE se forniti.
        Adatta automaticamente i parametri per dataset piccoli.

        Parametri:
        - texts: lista testi (len N)
        - embeddings: array (N x D) opzionale

        Ritorno: self
        Ultimo aggiornamento: 2025-08-23 - Fix parametri per dataset piccoli
        """
        if not self.available:
            raise RuntimeError("BERTopic non disponibile: installa dipendenze.")

        n_samples = len(texts)
        print(f"📊 BERTopic training su {n_samples} campioni")
        
        # 🛠️ ADATTA PARAMETRI PER DATASET PICCOLI
        # Per evitare l'errore "k must be less than or equal to the number of training points"
        adapted_umap_params = self.umap_params.copy()
        adapted_hdbscan_params = self.hdbscan_params.copy()
        
        # 🔧 FIX METRICA: HDBSCAN supporta limitatamente 'cosine', usa 'euclidean' per robustezza
        if adapted_hdbscan_params.get('metric') == 'cosine':
            adapted_hdbscan_params['metric'] = 'euclidean'
            print(f"🔧 Metrica HDBSCAN: cosine → euclidean (compatibilità)")
            
        # Assicurati che HDBSCAN abbia parametri compatibili con prediction_data
        # prediction_data viene generato solo con certe configurazioni
        adapted_hdbscan_params['prediction_data'] = True  # Forza generazione prediction_data
        adapted_hdbscan_params['match_reference_implementation'] = True  # Compatibilità
        
        if n_samples < 50:  # Dataset piccolo
            print(f"⚠️ Dataset piccolo ({n_samples} campioni) - Adattamento parametri...")
            
            # UMAP: n_neighbors deve essere <= n_samples - 1
            max_neighbors = max(2, n_samples - 1)
            original_neighbors = adapted_umap_params.get('n_neighbors', 15)
            adapted_umap_params['n_neighbors'] = min(original_neighbors, max_neighbors)
            
            # HDBSCAN: min_cluster_size deve essere ragionevole per il dataset
            max_cluster_size = max(2, n_samples // 3)
            original_cluster_size = adapted_hdbscan_params.get('min_cluster_size', 5)
            adapted_hdbscan_params['min_cluster_size'] = min(original_cluster_size, max_cluster_size)
            
            # HDBSCAN: min_samples deve essere <= min_cluster_size
            adapted_hdbscan_params['min_samples'] = min(
                adapted_hdbscan_params.get('min_samples', 2),
                adapted_hdbscan_params['min_cluster_size']
            )
            
            print(f"   🔧 UMAP n_neighbors: {original_neighbors} → {adapted_umap_params['n_neighbors']}")
            print(f"   🔧 HDBSCAN min_cluster_size: {original_cluster_size} → {adapted_hdbscan_params['min_cluster_size']}")
            print(f"   🔧 HDBSCAN min_samples: {adapted_hdbscan_params['min_samples']}")
            print(f"   🔧 HDBSCAN metric: {adapted_hdbscan_params['metric']}")

        # Crea modelli con parametri adattati
        reducer = umap.UMAP(**adapted_umap_params)
        
        # Crea HDBSCAN con parametri adattati
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(**adapted_hdbscan_params)
        
        # 🛠️ GESTIONE INTELLIGENTE calculate_probabilities
        # Solo disabilita per dataset DAVVERO piccoli dove matematicamente impossibile
        min_safe_size = 20
        
        calculate_probs = self.calculate_probabilities and (n_samples >= min_safe_size)
        
        if not calculate_probs and self.calculate_probabilities:
            if n_samples < min_safe_size:
                print(f"⚠️ Disabilito calculate_probabilities per dataset piccolo ({n_samples} < {min_safe_size} campioni)")
            else:
                print(f"⚠️ Disabilito calculate_probabilities per sicurezza ({n_samples} campioni)")
        
        # 🔧 CONFIGURAZIONE EMBEDDING MODEL per BERTopic transform
        # Se abbiamo un embedder configurato, creamo un wrapper compatibile con BERTopic
        embedding_model = None
        if self.embedder is not None:
            print(f"   🎯 Configurando embedding model per BERTopic: {type(self.embedder).__name__}")
            embedding_model = self._create_bertopic_embedding_wrapper()
        
        self.model = BERTopic(
            umap_model=reducer,
            hdbscan_model=clusterer,  # Usa configurazione personalizzata
            embedding_model=embedding_model,  # ✅ AGGIUNTO: embedding model configurato
            calculate_probabilities=calculate_probs,  # Disabilita per dataset piccoli
            verbose=False,
            low_memory=True,
            nr_topics=None,
            seed_topic_list=None,
        )
        
        try:
            # 🆕 IMPORTANTE: Non passare embeddings se abbiamo embedding_model personalizzato
            # Altrimenti BERTopic ignora il nostro embedding_model e usa il backend interno
            if embedding_model is not None:
                print("   🎯 Usando embedding_model personalizzato, BERTopic genererà embeddings internamente")
                self.model.fit(texts)  # Lascia che BERTopic usi il nostro embedding_model
            else:
                print("   🔧 Usando embeddings pre-computati")
                self.model.fit(texts, embeddings=embeddings)
            print(f"✅ BERTopic FIT completato su {n_samples} campioni")
        except Exception as e:
            error_msg = str(e).lower()
            print(f"❌ Errore BERTopic fit: {e}")
            
            # Fallback 1: Se errore specifico "prediction data", disabilita calculate_probabilities
            if "prediction data" in error_msg or "no prediction data" in error_msg:
                print("🔄 Fallback specifico: errore prediction_data → disabilito calculate_probabilities...")
                try:
                    self.model = BERTopic(
                        umap_model=reducer,
                        hdbscan_model=clusterer,
                        calculate_probabilities=False,  # Disabilita solo per questo errore specifico
                        verbose=False,
                        low_memory=True,
                        nr_topics=None,
                        seed_topic_list=None,
                    )
                    self.model.fit(texts, embeddings=embeddings)
                    print(f"✅ BERTopic FIT fallback completato su {n_samples} campioni (prediction_data fix)")
                    return  # Successo con fallback specifico
                except Exception as e2:
                    print(f"❌ Errore anche con fallback prediction_data: {e2}")
            
            # Fallback 2: Se errore memoria/parametri, prova parametri ultra-conservativi
            if n_samples < 50:
                print("🔄 Tentativo fallback con parametri ultra-conservativi...")
                adapted_umap_params['n_neighbors'] = max(2, min(5, n_samples - 1))
                adapted_hdbscan_params['min_cluster_size'] = max(2, min(3, n_samples // 4))
                adapted_hdbscan_params['min_samples'] = 1
                adapted_hdbscan_params['metric'] = 'euclidean'  # Forza metrica compatibile
                
                # Rimuovi parametri problematici per dataset micro
                if 'cluster_selection_method' in adapted_hdbscan_params:
                    adapted_hdbscan_params['cluster_selection_method'] = 'eom'
                
                reducer = umap.UMAP(**adapted_umap_params)
                clusterer = HDBSCAN(**adapted_hdbscan_params)
                
                self.model = BERTopic(
                    umap_model=reducer,
                    hdbscan_model=clusterer,
                    calculate_probabilities=False,  # SEMPRE disabilita per fallback
                    verbose=False,
                    low_memory=True,
                    nr_topics=None,
                    seed_topic_list=None,
                )
                self.model.fit(texts, embeddings=embeddings)
                print(f"✅ BERTopic FIT completato con parametri fallback")
            else:
                raise

        # Allena SVD sulle full probas se richiesto
        if self.use_svd:
            if not _SVD_AVAILABLE:
                print("⚠️ SVD non disponibile: installa scikit-learn. Procedo senza.")
            else:
                try:
                    # Calcola full probas sul training set
                    _, P = self.model.transform(texts, embeddings=embeddings)
                    if P is None or P.size == 0:
                        print("⚠️ Probabilità non disponibili; skip SVD training")
                    else:
                        k = min(self.svd_components, P.shape[1])
                        if k < 1:
                            print("⚠️ svd_components < 1; skip SVD training")
                        else:
                            self._svd_model = TruncatedSVD(
                                n_components=k,
                                random_state=self.random_state
                            )
                            self._svd_model.fit(P)
                            print(f"✅ SVD addestrato su full probas (k={k})")
                except Exception as e:
                    print(f"⚠️ Errore training SVD: {e}. Procedo senza SVD.")
                    self._svd_model = None
        return self

    def transform(self,
                  texts: List[str],
                  embeddings: Optional[np.ndarray] = None,
                  return_one_hot: bool = False,
                  top_k: Optional[int] = None
                  ) -> Dict[str, Any]:
        """
        Trasforma nuovi testi in feature topic.

        Parametri:
        - texts: lista testi (len N)
        - embeddings: array (N x D) opzionale
        - return_one_hot: se includere one-hot del topic-id
        - top_k: limita lunghezza vettore proba ai top-k topic (facolt.)

        Ritorno:
        {
          "topic_ids": np.ndarray shape (N,),
          "topic_probas": np.ndarray shape (N, T) o (N, K SVD),
          "one_hot": Optional[np.ndarray shape (N, T)]
        }
        Ultimo aggiornamento: 2025-08-07
        """
        if not self.model:
            raise RuntimeError("Modello BERTopic non addestrato.")

        try:
            topic_ids, probs = self.model.transform(texts, embeddings=embeddings)
        except Exception as e:
            print(f"❌ Errore transform BERTopic: {e}")
            # Se il transform fallisce, ritorna topic vuoti
            topic_ids = np.array([-1] * len(texts))  # Tutti outlier
            probs = None

        if probs is None:
            # Probabilità non disponibili (calculate_probabilities=False)
            T = int(max(topic_ids) + 1) if len(topic_ids) and max(topic_ids) >= 0 else 1
            probs = np.zeros((len(texts), T), dtype=float)
            for i, t in enumerate(topic_ids):
                if t >= 0:
                    if t >= probs.shape[1]:
                        # espandi dinamicamente
                        extra = t + 1 - probs.shape[1]
                        probs = np.pad(
                            probs, ((0, 0), (0, extra)),
                            mode="constant"
                        )
                    probs[i, t] = 1.0
        elif probs.ndim == 1:
            # Se probs è 1D (caso raro), convertilo in 2D
            probs = probs.reshape(-1, 1)

        # Se SVD attivo e modello addestrato, applica riduzione
        if self.use_svd and self._svd_model is not None and probs.size > 0:
            try:
                reduced = self._svd_model.transform(probs)
                result_probas = np.asarray(reduced, dtype=float)
                # One-hot calcolato sulle full probas prima della riduzione
                one_hot = None
                if return_one_hot:
                    T = probs.shape[1]
                    one_hot = np.zeros((len(texts), T), dtype=float)
                    if T > 0:
                        rows = np.arange(len(texts))
                        one_hot[rows, np.argmax(probs, axis=1)] = 1.0
                result = {
                    "topic_ids": np.asarray(topic_ids),
                    "topic_probas": result_probas,
                }
                if return_one_hot and one_hot is not None:
                    result["one_hot"] = one_hot
                return result
            except Exception as e:
                print(f"⚠️ Errore SVD transform: {e}. Uso full/top-k probas.")
                # falls-through a logica standard

        # Logica standard (senza SVD): opzionale top_k per documento
        if top_k is not None and probs.size > 0 and probs.ndim == 2:
            idx = np.argsort(-probs, axis=1)[:, :top_k]
            probs = np.take_along_axis(probs, idx, axis=1)

        result: Dict[str, Any] = {
            "topic_ids": np.asarray(topic_ids),
            "topic_probas": np.asarray(probs, dtype=float),
        }

        if return_one_hot:
            T = probs.shape[1]
            one_hot = np.zeros((len(texts), T), dtype=float)
            if T > 0:
                rows = np.arange(len(texts))
                one_hot[rows, np.argmax(probs, axis=1)] = 1.0
            result["one_hot"] = one_hot

        return result

    def get_topic_info(self,
                       topic_id: int,
                       top_k_words: Optional[int] = None
                       ) -> Dict[str, Any]:
        """
        Restituisce info su un topic: top words e rappresentanti.

        Parametri:
        - topic_id: indice del topic
        - top_k_words: override del numero parole

        Ritorno:
        {
          "topic": int,
          "words": List[Tuple[word, score]],
          "representatives": List[str]
        }
        Ultimo aggiornamento: 2025-08-07
        """
        if not self.model:
            raise RuntimeError("Modello BERTopic non addestrato.")

        k = top_k_words or self.top_k_words
        words = self.model.get_topic(topic_id)[:k] if k else []
        reps = self.model.get_representative_docs(topic_id) or []
        return {
            "topic": topic_id,
            "words": [(w, float(s)) for w, s in words],
            "representatives": reps,
        }

    def save(self, path: str) -> None:
        """
        Salva modello e metadati in directory.

        Parametri:
        - path: directory target

        Ritorno: None
        Ultimo aggiornamento: 2025-08-07
        """
        if not self.model:
            raise RuntimeError("Modello BERTopic non addestrato.")

        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "bertopic_model")
        self.model.save(model_path)

        # Salva SVD se presente
        if self._svd_model is not None:
            try:
                import joblib
                joblib.dump(self._svd_model, os.path.join(path, "svd_model.pkl"))
                print("💾 SVD salvato (svd_model.pkl)")
            except Exception as e:
                print(f"⚠️ Errore salvataggio SVD: {e}")

        meta = {
            "random_state": self.random_state,
            "umap_params": self.umap_params,
            "hdbscan_params": self.hdbscan_params,
            "calculate_probabilities": self.calculate_probabilities,
            "top_k_words": self.top_k_words,
            "use_svd": self.use_svd,
            "svd_components": self.svd_components,
            "version": "1.1",
        }
        with open(os.path.join(path, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> "BERTopicFeatureProvider":
        """
        Carica modello e metadati da directory.

        Parametri:
        - path: directory sorgente

        Ritorno: self
        Ultimo aggiornamento: 2025-08-07
        """
        if not self.available:
            raise RuntimeError("BERTopic non disponibile: installa dipendenze.")

        try:
            from bertopic import BERTopic  # reimport sicuro
            model_path = os.path.join(path, "bertopic_model")
            
            print(f"🔄 Caricamento BERTopic model da {model_path}")
            self.model = BERTopic.load(model_path)
            print("✅ BERTopic model caricato con successo")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Errore nel caricamento del modello BERTopic: {error_msg}")
            
            # Riconoscimento errori di incompatibilità numba/joblib
            if any(keyword in error_msg for keyword in [
                "code() argument", 
                "must be str, not int", 
                "numba", 
                "serialize",
                "_unpickle__CustomPickled"
            ]):
                print("🔧 ERRORE DI COMPATIBILITÀ NUMBA RILEVATO!")
                print("📋 Causa: Modello salvato con versione numba incompatibile")
                print("🔄 SOLUZIONE: Rimuovere modello corrotto e permettere ricreazione")
                
                # Rimuovi la directory corrotta
                import shutil
                if os.path.exists(path):
                    backup_path = path + "_corrupted_" + str(int(time.time()))
                    print(f"🗂️ Spostando modello corrotto in: {backup_path}")
                    shutil.move(path, backup_path)
                    
                # Forza ricreazione indicando assenza di modello
                print("✅ Modello corrotto rimosso - il sistema ricreerà automaticamente BERTopic")
                return None  # Indica che non c'è modello da caricare
            else:
                # Altri errori - propaga eccezione
                import traceback
                print(f"❌ Stack trace: {traceback.format_exc()}")
                raise e

        # Carica SVD se presente e previsto
        svd_path = os.path.join(path, "svd_model.pkl")
        if os.path.exists(svd_path):
            try:
                import joblib
                self._svd_model = joblib.load(svd_path)
                print("📥 SVD caricato (svd_model.pkl)")
            except Exception as e:
                print(f"⚠️ Errore caricamento SVD: {e}")
                self._svd_model = None

        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.random_state = meta.get("random_state", self.random_state)
            self.umap_params = meta.get("umap_params", self.umap_params)
            self.hdbscan_params = meta.get("hdbscan_params",
                                           self.hdbscan_params)
            self.calculate_probabilities = meta.get(
                "calculate_probabilities", self.calculate_probabilities
            )
            self.top_k_words = meta.get("top_k_words", self.top_k_words)
            # Ripristina config SVD
            self.use_svd = bool(meta.get("use_svd", self.use_svd)) and _SVD_AVAILABLE
            self.svd_components = int(meta.get("svd_components", self.svd_components))
        return self
    
    def _create_bertopic_embedding_wrapper(self):
        """
        Crea wrapper embedder compatibile con BERTopic
        
        Scopo della funzione: Crea wrapper per embedder che rispetta interfaccia BERTopic
        Parametri di input: None (usa self.embedder)
        Parametri di output: Wrapper embedder BERTopic-compatibile
        Valori di ritorno: Classe wrapper con metodo embed
        Tracciamento aggiornamenti: 2025-08-28 - Creata per fix BERTopic transform
        
        Returns:
            Wrapper embedder per BERTopic
            
        Autore: Valerio Bignardi
        Data: 2025-08-28
        """
        embedder = self.embedder
        
        class BERTopicEmbeddingWrapper:
            """Wrapper per embedder compatibile con BERTopic"""
            
            def __init__(self, embedder):
                self.embedder = embedder
            
            def embed(self, texts, verbose=False):
                """Metodo embed richiesto da BERTopic"""
                if isinstance(texts, str):
                    texts = [texts]
                return self.embedder.encode(texts, show_progress_bar=verbose)
        
        return BERTopicEmbeddingWrapper(embedder)
