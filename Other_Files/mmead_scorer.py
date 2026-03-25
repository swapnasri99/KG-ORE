"""
MMEAD KG Scorer for KG-ORE
==========================

Two modes:
1. Live MMEAD (DuckDB): --use_mmead
2. Cached SQLite:       --use_mmead --mmead_cache mmead_cache.db  (FAST)

Build cache first: python build_mmead_cache.py

Scoring:
    score = α × laff_norm + β × entity_id_overlap + γ × embedding_similarity
"""

import json
import struct
import sqlite3
import time
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
import random


@dataclass
class ScoringComponents:
    laff_raw: float
    laff_norm: float
    entity_overlap: float
    kg_connectivity: float
    combined: float


# ============================================================
# Entity Store — Cached SQLite (FAST)
# ============================================================

class CachedEntityStore:
    """
    Loads entities from mmead_cache.db (pre-built SQLite).
    Instant lookups, no DuckDB overhead.
    """

    def __init__(self, db_path: str):
        print(f"  Loading entity cache from {db_path}...")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size=-200000;")  # 200 MB cache

        count = self.conn.execute("SELECT COUNT(*) FROM passage_entities").fetchone()[0]
        print(f"  ✓ {count:,} passages in cache")

        # In-memory cache for hot passages
        self._cache: Dict[str, Tuple[Set[int], Set[str]]] = {}
        self._hits = 0
        self._misses = 0

    def _load_passage(self, docno: str) -> Tuple[Set[int], Set[str]]:
        if docno in self._cache:
            self._hits += 1
            return self._cache[docno]

        self._misses += 1

        row = self.conn.execute(
            "SELECT entity_ids, entity_names FROM passage_entities WHERE pid=?",
            (docno,)
        ).fetchone()

        if row is None:
            self._cache[docno] = (set(), set())
            return set(), set()

        # Unpack entity_ids from binary
        id_blob = row[0]
        n_ids = len(id_blob) // 4  # 4 bytes per int
        id_list = struct.unpack(f'{n_ids}i', id_blob) if n_ids > 0 else []
        entity_ids = set(id_list)

        # Unpack names from JSON
        entity_names = set(json.loads(row[1])) if row[1] else set()

        self._cache[docno] = (entity_ids, entity_names)
        return entity_ids, entity_names

    def preload_passages(self, docnos: List[str]):
        """Bulk preload from SQLite — fast."""
        needed = [d for d in docnos if d not in self._cache]
        if not needed:
            return

        # Batch query
        for batch_start in range(0, len(needed), 5000):
            batch = needed[batch_start:batch_start + 5000]
            placeholders = ','.join('?' for _ in batch)
            rows = self.conn.execute(
                f"SELECT pid, entity_ids, entity_names FROM passage_entities WHERE pid IN ({placeholders})",
                batch
            ).fetchall()

            for pid, id_blob, names_json in rows:
                n_ids = len(id_blob) // 4
                id_list = struct.unpack(f'{n_ids}i', id_blob) if n_ids > 0 else []
                self._cache[str(pid)] = (set(id_list), set(json.loads(names_json)) if names_json else set())

            # Cache empty for not-found
            for d in batch:
                if d not in self._cache:
                    self._cache[d] = (set(), set())

    def get_passage_ids(self, docno: str) -> Set[int]:
        ids, _ = self._load_passage(docno)
        return ids

    def get_passage_names(self, docno: str) -> Set[str]:
        _, names = self._load_passage(docno)
        return names

    def get_passage_mids(self, docno: str) -> Set[str]:
        _, names = self._load_passage(docno)
        return names

    def passage_count(self) -> int:
        return len(self._cache)

    def get_cache_stats(self) -> Dict:
        return {
            'cached_passages': len(self._cache),
            'cache_hits': self._hits,
            'cache_misses': self._misses,
        }


# ============================================================
# Entity Store — Live MMEAD (DuckDB fallback)
# ============================================================

class LiveEntityStore:
    """
    Loads entities directly from MMEAD DuckDB.
    Slower but no pre-build step needed.
    """

    def __init__(self, linker: str = 'rel'):
        from mmead import get_links
        print(f"  Loading MMEAD entity links (linker={linker})...")
        self.links = get_links('v1', 'passage', linker=linker)
        print(f"  ✓ MMEAD ready")

        self._cache: Dict[str, Tuple[Set[int], Set[str]]] = {}
        self._hits = 0
        self._misses = 0

        # DuckDB connection for bulk queries
        self._conn = None
        self._table_name = f"msmarco_v1_passage_links_{linker}"
        try:
            self._conn = self.links.cursor
        except Exception:
            pass

    def _load_passage(self, docno: str) -> Tuple[Set[int], Set[str]]:
        if docno in self._cache:
            self._hits += 1
            return self._cache[docno]

        self._misses += 1
        try:
            result = self.links.load_links_from_docid(int(docno))
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result

            entity_ids = set()
            entity_names = set()
            for ent in data.get('passage', []):
                eid = ent.get('entity_id')
                name = ent.get('entity', '')
                if eid is not None:
                    entity_ids.add(int(eid))
                if name:
                    entity_names.add(name)

            self._cache[docno] = (entity_ids, entity_names)
            return entity_ids, entity_names
        except Exception:
            self._cache[docno] = (set(), set())
            return set(), set()

    def preload_passages(self, docnos: List[str]):
        needed = [d for d in docnos if d not in self._cache]
        if not needed or self._conn is None:
            # Fallback: load one by one
            for d in needed:
                self._load_passage(d)
            return

        try:
            pid_list = [int(d) for d in needed]
            for batch_start in range(0, len(pid_list), 5000):
                batch = pid_list[batch_start:batch_start + 5000]
                placeholders = ','.join(str(p) for p in batch)
                rows = self._conn.execute(f"""
                    SELECT pid, entity_id, entity
                    FROM {self._table_name}
                    WHERE pid IN ({placeholders})
                """).fetchall()

                from collections import defaultdict
                pid_entities = defaultdict(lambda: (set(), set()))
                for pid, eid, name in rows:
                    pid_str = str(pid)
                    if eid is not None:
                        pid_entities[pid_str][0].add(int(eid))
                    if name:
                        pid_entities[pid_str][1].add(name)

                for pid_str, (ids, names) in pid_entities.items():
                    self._cache[pid_str] = (ids, names)
                for p in batch:
                    if str(p) not in self._cache:
                        self._cache[str(p)] = (set(), set())
        except Exception:
            for d in needed:
                self._load_passage(d)

    def get_passage_ids(self, docno: str) -> Set[int]:
        ids, _ = self._load_passage(docno)
        return ids

    def get_passage_names(self, docno: str) -> Set[str]:
        _, names = self._load_passage(docno)
        return names

    def get_passage_mids(self, docno: str) -> Set[str]:
        _, names = self._load_passage(docno)
        return names

    def passage_count(self) -> int:
        return len(self._cache)

    def get_cache_stats(self) -> Dict:
        return {
            'cached_passages': len(self._cache),
            'cache_hits': self._hits,
            'cache_misses': self._misses,
        }


# ============================================================
# Embedding Similarity
# ============================================================

class EmbeddingSimilarity:
    """
    Entity similarity via Wikipedia2Vec.
    Two modes: cached SQLite or live MMEAD.
    """

    def __init__(self, emb_dim: int = 300, cache_db_path: str = None):
        self._vec_cache: Dict[str, Optional[np.ndarray]] = {}
        self._found = 0
        self._not_found = 0
        self._emb_dim = emb_dim
        self._cache_conn = None
        self._live_emb = None

        if cache_db_path:
            # Load from pre-built SQLite
            print(f"  Loading embeddings from cache DB...")
            self._cache_conn = sqlite3.connect(cache_db_path, check_same_thread=False)
            count = self._cache_conn.execute("SELECT COUNT(*) FROM entity_embeddings").fetchone()[0]
            print(f"  ✓ {count:,} embeddings in cache")
        else:
            # Live MMEAD
            from mmead import get_embeddings
            print(f"  Loading Wikipedia2Vec embeddings ({emb_dim}d)...")
            self._live_emb = get_embeddings(emb_dim)
            print(f"  ✓ Wikipedia2Vec ready")

    def preload_embeddings(self, entity_names: Set[str]):
        """Preload embeddings for a batch of entity names."""
        needed = [n for n in entity_names if n not in self._vec_cache]
        if not needed:
            return

        if self._cache_conn:
            # Bulk load from SQLite
            for batch_start in range(0, len(needed), 5000):
                batch = needed[batch_start:batch_start + 5000]
                placeholders = ','.join('?' for _ in batch)
                rows = self._cache_conn.execute(
                    f"SELECT entity_name, embedding FROM entity_embeddings WHERE entity_name IN ({placeholders})",
                    batch
                ).fetchall()

                for name, emb_blob in rows:
                    vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
                    self._vec_cache[name] = vec
                    self._found += 1

                # Mark not-found
                found_names = {r[0] for r in rows}
                for name in batch:
                    if name not in found_names and name not in self._vec_cache:
                        self._vec_cache[name] = None
                        self._not_found += 1
        else:
            # Live: load one by one
            for name in needed:
                self._get_vector(name)

    def _get_vector(self, entity_name: str) -> Optional[np.ndarray]:
        if entity_name in self._vec_cache:
            return self._vec_cache[entity_name]

        vec = None

        if self._cache_conn:
            row = self._cache_conn.execute(
                "SELECT embedding FROM entity_embeddings WHERE entity_name=?",
                (entity_name,)
            ).fetchone()
            if row:
                vec = np.frombuffer(row[0], dtype=np.float32).copy()
        elif self._live_emb:
            try:
                raw = self._live_emb.load_entity_embedding(entity_name)
                if raw is not None:
                    norm = np.linalg.norm(raw)
                    if norm > 0:
                        vec = raw / norm
            except Exception:
                pass

        if vec is not None:
            self._vec_cache[entity_name] = vec
            self._found += 1
        else:
            self._vec_cache[entity_name] = None
            self._not_found += 1

        return vec

    def compute_similarity(
        self,
        names_a: Set[str],
        names_b: Set[str],
        shared_names: Set[str] = None
    ) -> float:
        """Max-cosine alignment (BERTScore-style)."""
        if shared_names is None:
            shared_names = set()

        only_a = names_a - shared_names
        only_b = names_b - shared_names

        if not only_a and not only_b:
            return 1.0

        vecs_a = {}
        for name in only_a:
            v = self._get_vector(name)
            if v is not None:
                vecs_a[name] = v

        vecs_b = {}
        for name in only_b:
            v = self._get_vector(name)
            if v is not None:
                vecs_b[name] = v

        if not vecs_a or not vecs_b:
            return 0.0

        # A→B
        a_to_b = []
        for vec_a in vecs_a.values():
            best = max(float(np.dot(vec_a, vec_b)) for vec_b in vecs_b.values())
            a_to_b.append(max(0.0, best))

        # B→A
        b_to_a = []
        for vec_b in vecs_b.values():
            best = max(float(np.dot(vec_b, vec_a)) for vec_a in vecs_a.values())
            b_to_a.append(max(0.0, best))

        avg_a = sum(a_to_b) / len(a_to_b) if a_to_b else 0.0
        avg_b = sum(b_to_a) / len(b_to_a) if b_to_a else 0.0

        return (avg_a + avg_b) / 2.0

    def get_stats(self) -> Dict:
        return {
            'embeddings_found': self._found,
            'embeddings_not_found': self._not_found,
            'vec_cache_size': len(self._vec_cache),
        }


# ============================================================
# MMEAD KG Scorer
# ============================================================

class MMEADScorer:
    """
    Drop-in replacement for KGScorerUnified.
    Same interface: rescore_neighbors(docno, neighbor_docnos, laff_weights)
    """

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        emb_dim: int = 300,
        linker: str = 'rel',
        cache_db: str = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        total = alpha + beta + gamma
        if abs(total - 1.0) > 0.01:
            print(f"  Warning: Weights sum to {total:.2f}, normalizing...")
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total

        print("=" * 60)
        print("MMEAD KG Scorer")
        print("=" * 60)
        print(f"  α={self.alpha:.2f} (LAFF)  β={self.beta:.2f} (Entity ID Overlap)  γ={self.gamma:.2f} (Embedding Sim)")
        print(f"  Mode: {'Cached SQLite' if cache_db else 'Live MMEAD (DuckDB)'}")

        # Entity store
        if cache_db:
            self.entity_store = CachedEntityStore(cache_db)
        else:
            self.entity_store = LiveEntityStore(linker=linker)

        # Embedding similarity
        if self.gamma > 0:
            self.emb_sim = EmbeddingSimilarity(emb_dim=emb_dim, cache_db_path=cache_db)
            print(f"  Embedding similarity: ENABLED (max-cosine alignment)")
        else:
            self.emb_sim = None
            print(f"  Embedding similarity: DISABLED (γ=0)")

        print(f"  Final weights: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
        print("=" * 60)

    def compute_entity_overlap(self, ids_a: Set[int], ids_b: Set[int]) -> float:
        if not ids_a or not ids_b:
            return 0.0
        shared = len(ids_a & ids_b)
        return shared / min(len(ids_a), len(ids_b))

    def compute_kg_connectivity(self, names_a: Set[str], names_b: Set[str]) -> float:
        if self.emb_sim is None or not names_a or not names_b:
            return 0.0
        shared = names_a & names_b
        return self.emb_sim.compute_similarity(names_a, names_b, shared_names=shared)

    def score_neighbor(
        self,
        doc_ids: Set[int],
        doc_names: Set[str],
        neighbor_ids: Set[int],
        neighbor_names: Set[str],
        laff_weight: float,
        laff_min: float = None,
        laff_max: float = None
    ) -> ScoringComponents:
        if laff_min is not None and laff_max is not None and laff_max > laff_min:
            laff_norm = (laff_weight - laff_min) / (laff_max - laff_min)
        else:
            laff_norm = laff_weight
        laff_norm = max(0.0, min(1.0, laff_norm))

        entity_overlap = self.compute_entity_overlap(doc_ids, neighbor_ids)
        kg_connectivity = self.compute_kg_connectivity(doc_names, neighbor_names)

        combined = (
            self.alpha * laff_norm +
            self.beta * entity_overlap +
            self.gamma * kg_connectivity
        )

        return ScoringComponents(
            laff_raw=laff_weight,
            laff_norm=laff_norm,
            entity_overlap=entity_overlap,
            kg_connectivity=kg_connectivity,
            combined=combined
        )

    def rescore_neighbors(
        self,
        docno: str,
        neighbor_docnos: List[str],
        laff_weights: np.ndarray
    ) -> List[Tuple[str, float, ScoringComponents]]:
        # Preload all passages at once
        all_docnos = [docno] + list(neighbor_docnos)
        self.entity_store.preload_passages(all_docnos)

        # Preload all embeddings at once
        if self.emb_sim is not None:
            all_names = set()
            for d in all_docnos:
                all_names.update(self.entity_store.get_passage_names(d))
            self.emb_sim.preload_embeddings(all_names)

        doc_ids = self.entity_store.get_passage_ids(docno)
        doc_names = self.entity_store.get_passage_names(docno)

        laff_min = float(np.min(laff_weights))
        laff_max = float(np.max(laff_weights))

        results = []
        for i, neighbor_docno in enumerate(neighbor_docnos):
            n_ids = self.entity_store.get_passage_ids(neighbor_docno)
            n_names = self.entity_store.get_passage_names(neighbor_docno)

            components = self.score_neighbor(
                doc_ids, doc_names,
                n_ids, n_names,
                float(laff_weights[i]), laff_min, laff_max
            )
            results.append((neighbor_docno, components.combined, components))
            if random.random() < 0.0005:  
                print("DBG", docno, neighbor_docno,
          "laff", round(components.laff_norm, 3),
          "overlap", round(components.entity_overlap, 3),
          "kg", round(components.kg_connectivity, 3),
          "combined", round(components.combined, 3))

        results.sort(key=lambda x: x[1], reverse=True)

        
        return results

    def save_caches(self):
        pass

    def get_stats(self) -> Dict:
        stats = {
            'passages_with_entities': self.entity_store.passage_count(),
            'weights': f"α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}",
            'scorer_type': 'MMEAD (Wikipedia2Vec)',
        }
        stats.update(self.entity_store.get_cache_stats())
        if self.emb_sim:
            stats.update(self.emb_sim.get_stats())
        return stats


# ============================================================
# Factory
# ============================================================

def create_mmead_scorer(
    alpha: float = 0.0,
    beta: float = 0.5,
    gamma: float = 0.5,
    emb_dim: int = 300,
    linker: str = 'rel',
    cache_db: str = None,
    **kwargs
) -> MMEADScorer:
    return MMEADScorer(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        emb_dim=emb_dim,
        linker=linker,
        cache_db=cache_db,
    )