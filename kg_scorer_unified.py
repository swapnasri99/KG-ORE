import os
import json
import math
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from entity_store_disk import DiskEntityStore


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ScoringComponents:
    """Container for individual scoring components."""
    laff_raw: float
    laff_norm: float
    entity_overlap: float
    kg_connectivity: float
    combined: float


# Entity Store

class KGPREntityStore:
    """
    Loads and manages KGPR's pre-computed entity linking results.

    Provides:
    - Entity names (for text-level overlap)
    - Freebase MIDs (for KG connectivity)
    """

    def __init__(
        self,
        passage_el_path: str = None,
        full_passage_el_path: str = None,
        query_el_path: str = None
    ):
        # docno -> {"names": set of lowercased entity names, "mids": set of freebase MIDs}
        self._passage_data: Dict[str, Dict[str, Set[str]]] = {}
        self._query_data: Dict[str, Dict[str, Set[str]]] = {}

        if passage_el_path:
            self._load_el_file(passage_el_path, self._passage_data, "passage (BM25 subset)")

        if full_passage_el_path:
            self._load_el_file(full_passage_el_path, self._passage_data, "passage (full corpus)")

        if query_el_path:
            self._load_el_file(query_el_path, self._query_data, "query")

        print(f"  EntityStore: {len(self._passage_data):,} passages, {len(self._query_data):,} queries")

    def _load_el_file(self, path: str, storage: dict, label: str):
        """Load a JSONL entity linking file."""
        if not os.path.exists(path):
            print(f"  WARNING: EL file not found: {path}")
            return

        print(f"  Loading {label} EL from {path}...")
        count_new = 0

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"  Loading {label} EL", mininterval=2.0):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                doc_id = str(item["id"])

                if doc_id in storage:
                    continue

                # Extract entity names (lowercased for overlap matching)
                entity_names = set()
                for name in item.get("entity_name", []):
                    normalized = name.lower().strip()
                    if len(normalized) >= 2:
                        entity_names.add(normalized)

                # Extract Freebase MIDs
                freebase_mids = set(item.get("entity", []))

                if entity_names or freebase_mids:
                    storage[doc_id] = {
                        "names": entity_names,
                        "mids": freebase_mids
                    }
                    count_new += 1

        print(f"    Loaded {count_new:,} new entries (total: {len(storage):,})")

    def get_passage_names(self, docno: str) -> Set[str]:
        """Get entity names for a passage (lowercased)."""
        data = self._passage_data.get(str(docno))
        return data["names"] if data else set()

    def get_passage_mids(self, docno: str) -> Set[str]:
        """Get Freebase MIDs for a passage."""
        data = self._passage_data.get(str(docno))
        return data["mids"] if data else set()

    def get_query_names(self, qid: str) -> Set[str]:
        """Get entity names for a query."""
        data = self._query_data.get(str(qid))
        return data["names"] if data else set()

    def get_query_mids(self, qid: str) -> Set[str]:
        """Get Freebase MIDs for a query."""
        data = self._query_data.get(str(qid))
        return data["mids"] if data else set()

    def has_passage(self, docno: str) -> bool:
        return str(docno) in self._passage_data

    def passage_count(self) -> int:
        return len(self._passage_data)

    def get_coverage(self, docnos: List[str]) -> Dict:
        """Check EL coverage for a list of docnos."""
        total = len(docnos)
        covered = sum(1 for d in docnos if str(d) in self._passage_data)
        return {
            'total': total,
            'covered': covered,
            'missing': total - covered,
            'coverage_pct': f"{100*covered/total:.1f}%" if total > 0 else "N/A"
        }


# ============================================================
# Freebase Graph (local sparse matrix)
# ============================================================

class FreebaseGraph:
    """
    In-memory Freebase KG for instant 1-hop connectivity checks.
    Uses sparse adjacency matrix from KGPR's pre-built triples.
    """

    def __init__(self, freebase_dir: str):
        print(f"  Loading Freebase graph from {freebase_dir}...")

        triples_path = os.path.join(freebase_dir, "subgraph_1hop_triples.npy")
        ent2id_path = os.path.join(freebase_dir, "ent2id.pickle")

        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"Missing: {triples_path}")
        if not os.path.exists(ent2id_path):
            raise FileNotFoundError(f"Missing: {ent2id_path}")

        # Load triples
        triples = np.load(triples_path)
        E = triples.shape[0]

        # Load entity mapping
        with open(ent2id_path, 'rb') as f:
            self.ent2id = pickle.load(f)

        # Build sparse adjacency
        max_id = max(triples[:, 0].max(), triples[:, 2].max()) + 1

        fwd = csr_matrix(
            (np.ones(E, dtype=bool), (triples[:, 0], triples[:, 2])),
            shape=(max_id, max_id)
        )

        bwd = csr_matrix(
            (np.ones(E, dtype=bool), (triples[:, 2], triples[:, 0])),
            shape=(max_id, max_id)
        )

        self.adj = fwd + bwd

        self._cache: Dict[Tuple[int, int], bool] = {}

        print(f"    {E:,} triples, {len(self.ent2id):,} entities")

    def are_connected(self, mid1: str, mid2: str) -> bool:
        """Check 1-hop connectivity between two Freebase MIDs."""
        if mid1 == mid2:
            return True

        id1 = self.ent2id.get(mid1)
        id2 = self.ent2id.get(mid2)
        if id1 is None or id2 is None:
            return False

        key = (min(id1, id2), max(id1, id2))
        if key in self._cache:
            return self._cache[key]

        connected = bool(self.adj[id1, id2])
        self._cache[key] = connected
        return connected

    def has_entity(self, mid: str) -> bool:
        return mid in self.ent2id


# ============================================================
# Unified KG Scorer
# ============================================================

class KGScorerUnified:
    """
    KG-Enhanced Neighbor Scorer (Unified).

    Uses KGPR's entities for everything:
    - Entity Overlap: based on entity_name text matching
    - KG Connectivity: based on Freebase 1-hop paths

    No spaCy, no Wikidata API, no external calls.

    Formula:
        Score = α × LAFF_norm + β × EntityOverlap + γ × KGConnectivity
    """

    VALID_KG_MODES = {'binary', 'count', 'log', 'coverage', 'ratio', 'raw', 'minmax'}

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        kg_score_mode: str = 'log',
        max_connections_cap: int = 10,
        # Data sources
        entity_store: KGPREntityStore = None,
        freebase_graph: FreebaseGraph = None,
        # Paths (if entity_store/freebase_graph not provided)
        freebase_dir: str = None,
        passage_el_path: str = None,
        full_passage_el_path: str = None,
        query_el_path: str = None,
    ):
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kg_score_mode = kg_score_mode
        self.max_connections_cap = max_connections_cap

        # Normalize weights
        total = alpha + beta + gamma
        if abs(total - 1.0) > 0.01:
            print(f"  Warning: Weights sum to {total:.2f}, normalizing...")
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total

        print("=" * 60)
        print("KG Scorer (Unified - KGPR Entities)")
        print("=" * 60)
        print(f"  α={self.alpha:.2f} (LAFF)  β={self.beta:.2f} (Entity Overlap)  γ={self.gamma:.2f} (KG)")
        print(f"  KG score mode: {kg_score_mode}")

        # Entity Store
        if entity_store is not None:
            self.entity_store = entity_store
        else:
            self.entity_store = KGPREntityStore(
                passage_el_path=passage_el_path,
                full_passage_el_path=full_passage_el_path,
                query_el_path=query_el_path
            )

        # Freebase Graph (only if gamma > 0)
        if self.gamma > 0:
            if freebase_graph is not None:
                self.freebase_graph = freebase_graph
            elif freebase_dir:
                self.freebase_graph = FreebaseGraph(freebase_dir)
            else:
                print("  WARNING: γ > 0 but no Freebase graph. Setting γ=0.")
                self.alpha += self.gamma / 2
                self.beta += self.gamma / 2
                self.gamma = 0.0
                self.freebase_graph = None
        else:
            self.freebase_graph = None
            print("  KG connectivity: DISABLED (γ=0)")

        use_kg = self.gamma > 0 and self.freebase_graph is not None
        print(f"  KG connectivity: {'ENABLED (Freebase local)' if use_kg else 'DISABLED'}")
        print(f"  Final weights: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
        print("=" * 60)

    # ----------------------------------------------------------
    # Entity Overlap (using KGPR entity names)
    # ----------------------------------------------------------

    def compute_entity_overlap(
        self,
        names1: Set[str],
        names2: Set[str]
    ) -> float:
        """
        Compute entity overlap from KGPR entity names.

        Uses normalized entity_name field (e.g., "diabetes mellitus").
        Formula: |intersection| / min(|A|, |B|)

        Returns: Score in [0, 1]
        """
        if not names1 or not names2:
            return 0.0

        intersection = len(names1 & names2)
        min_size = min(len(names1), len(names2))
        return intersection / min_size if min_size > 0 else 0.0

    # ----------------------------------------------------------
    # KG Connectivity (using Freebase graph)
    # ----------------------------------------------------------

    def compute_kg_connectivity(
        self,
        mids1: Set[str],
        mids2: Set[str]
    ) -> float:
        """
        Compute KG connectivity between two sets of Freebase entities.

        Returns: Score in [0, 1] for most modes, raw count for 'raw'/'minmax'.
        """
        if self.freebase_graph is None or not mids1 or not mids2:
            return 0.0

        # Filter to entities that exist in graph
        valid1 = {m for m in mids1 if self.freebase_graph.has_entity(m)}
        valid2 = {m for m in mids2 if self.freebase_graph.has_entity(m)}

        if not valid1 or not valid2:
            return 0.0

        # Binary mode: any connection = 1.0
        if self.kg_score_mode == 'binary':
            for m1 in valid1:
                for m2 in valid2:
                    if m1 != m2 and self.freebase_graph.are_connected(m1, m2):
                        return 1.0
            return 0.0

        # Coverage mode: fraction of smaller set that connects
        if self.kg_score_mode == 'coverage':
            if len(valid1) <= len(valid2):
                smaller, larger = valid1, valid2
            else:
                smaller, larger = valid2, valid1

            connected_count = 0
            for ms in smaller:
                for ml in larger:
                    if ms != ml and self.freebase_graph.are_connected(ms, ml):
                        connected_count += 1
                        break

            return connected_count / len(smaller)

        # Count-based modes (count, log, ratio, raw, minmax)
        connected = 0
        checks = 0
        max_checks = 500

        for m1 in valid1:
            for m2 in valid2:
                if checks >= max_checks:
                    break
                if m1 != m2 and self.freebase_graph.are_connected(m1, m2):
                    connected += 1
                checks += 1
            if checks >= max_checks:
                break

        if connected == 0:
            return 0.0

        if self.kg_score_mode == 'count':
            return min(connected / self.max_connections_cap, 1.0)

        if self.kg_score_mode == 'log':
            return min(math.log2(connected + 1) / 5.0, 1.0)

        if self.kg_score_mode == 'ratio':
            total = min(len(valid1) * len(valid2), max_checks)
            return connected / total if total > 0 else 0.0

        if self.kg_score_mode in ('raw', 'minmax'):
            return float(connected)  # raw count; minmax normalizes in rescore_neighbors

        return 0.0

    # ----------------------------------------------------------
    # Combined Scoring
    # ----------------------------------------------------------

    def score_neighbor(
        self,
        doc_names: Set[str],
        doc_mids: Set[str],
        neighbor_names: Set[str],
        neighbor_mids: Set[str],
        laff_weight: float,
        laff_min: float = None,
        laff_max: float = None
    ) -> ScoringComponents:
        """Score a single neighbor using all three signals."""
        # Normalize LAFF to [0, 1]
        if laff_min is not None and laff_max is not None and laff_max > laff_min:
            laff_norm = (laff_weight - laff_min) / (laff_max - laff_min)
        else:
            laff_norm = laff_weight
        laff_norm = max(0.0, min(1.0, laff_norm))

        # Entity overlap
        entity_overlap = self.compute_entity_overlap(doc_names, neighbor_names)

        # KG connectivity
        kg_connectivity = self.compute_kg_connectivity(doc_mids, neighbor_mids)

        # Combined
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
        """
        Rescore all neighbors of a document.

        When kg_mode='minmax', applies min-max normalization on KG
        connectivity across all neighbors of this document.

        Returns:
            List of (neighbor_docno, combined_score, components) sorted DESC
        """
        doc_names = self.entity_store.get_passage_names(docno)
        doc_mids = self.entity_store.get_passage_mids(docno) if self.gamma > 0 else set()

        # LAFF normalization bounds
        laff_min = float(np.min(laff_weights))
        laff_max = float(np.max(laff_weights))

        if self.kg_score_mode == 'minmax' and self.gamma > 0:
            # === MIN-MAX MODE ===
            # Step 1: Compute raw KG counts + entity overlaps for all neighbors
            raw_data = []
            for i, neighbor_docno in enumerate(neighbor_docnos):
                n_names = self.entity_store.get_passage_names(neighbor_docno)
                n_mids = self.entity_store.get_passage_mids(neighbor_docno)

                # LAFF normalization
                if laff_max > laff_min:
                    laff_norm = (float(laff_weights[i]) - laff_min) / (laff_max - laff_min)
                else:
                    laff_norm = float(laff_weights[i])
                laff_norm = max(0.0, min(1.0, laff_norm))

                # Entity overlap (already [0,1])
                entity_overlap = self.compute_entity_overlap(doc_names, n_names)

                # Raw KG connected count
                kg_raw = self.compute_kg_connectivity(doc_mids, n_mids)

                raw_data.append({
                    'neighbor': neighbor_docno,
                    'laff_norm': laff_norm,
                    'laff_raw': float(laff_weights[i]),
                    'entity_overlap': entity_overlap,
                    'kg_raw': kg_raw,
                })

            # Step 2: Min-max normalize KG scores across all neighbors
            kg_values = [d['kg_raw'] for d in raw_data]
            kg_min = min(kg_values)
            kg_max = max(kg_values)

            results = []
            for d in raw_data:
                if kg_max > kg_min:
                    kg_norm = (d['kg_raw'] - kg_min) / (kg_max - kg_min)
                else:
                    kg_norm = 0.0 if d['kg_raw'] == 0 else 1.0

                combined = (
                    self.alpha * d['laff_norm'] +
                    self.beta * d['entity_overlap'] +
                    self.gamma * kg_norm
                )

                components = ScoringComponents(
                    laff_raw=d['laff_raw'],
                    laff_norm=d['laff_norm'],
                    entity_overlap=d['entity_overlap'],
                    kg_connectivity=kg_norm,
                    combined=combined
                )
                results.append((d['neighbor'], combined, components))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        else:
            # === ORIGINAL MODE (log, binary, count, etc.) ===
            results = []
            for i, neighbor_docno in enumerate(neighbor_docnos):
                n_names = self.entity_store.get_passage_names(neighbor_docno)
                n_mids = self.entity_store.get_passage_mids(neighbor_docno) if self.gamma > 0 else set()

                components = self.score_neighbor(
                    doc_names, doc_mids,
                    n_names, n_mids,
                    float(laff_weights[i]), laff_min, laff_max
                )
                results.append((neighbor_docno, components.combined, components))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def save_caches(self):
        """No-op (all data is in memory, no API caches)."""
        pass

    def get_stats(self) -> Dict:
        """Get statistics."""
        stats = {
            'passages_with_entities': self.entity_store.passage_count(),
            'weights': f"α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}",
            'kg_mode': self.kg_score_mode,
        }
        if self.freebase_graph:
            stats['freebase_entities'] = len(self.freebase_graph.ent2id)
            stats['pair_cache_size'] = len(self.freebase_graph._cache)
        return stats


# ============================================================
# Factory Function
# ============================================================

def create_scorer(
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    kg_score_mode: str = 'log',
    freebase_dir: str = None,
    passage_el_path: str = None,
    full_passage_el_path: str = None,
    query_el_path: str = None,
    passage_el_db: str = None,
    **kwargs
) -> KGScorerUnified:
    """
    Create unified KG scorer.

    If passage_el_db is provided, uses disk-based SQLite store (~50 MB RAM).
    Otherwise loads JSONL into memory (original behavior).
    """
    if passage_el_db:
        entity_store = DiskEntityStore(passage_el_db, query_el_path=query_el_path)
    else:
        entity_store = None

    return KGScorerUnified(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kg_score_mode=kg_score_mode,
        freebase_dir=freebase_dir,
        passage_el_path=passage_el_path if not passage_el_db else None,
        full_passage_el_path=full_passage_el_path if not passage_el_db else None,
        query_el_path=query_el_path if not passage_el_db else None,
        entity_store=entity_store,
    )


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Testing KGScorerUnified")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python kg_scorer_unified.py <passage_el_path> [freebase_dir]")
        print("\nExamples:")
        print("  # Entity overlap only:")
        print("  python kg_scorer_unified.py entity_linking_results/passage_test_with_id_bm25rank1000.jsonl")
        print("\n  # With KG connectivity:")
        print("  python kg_scorer_unified.py entity_linking_results/passage_test_with_id_bm25rank1000.jsonl freebase")
        sys.exit(0)

    passage_el = sys.argv[1]
    freebase_dir = sys.argv[2] if len(sys.argv) > 2 else None
    gamma = 0.3 if freebase_dir else 0.0

    scorer = create_scorer(
        alpha=0.4, beta=0.3, gamma=gamma,
        passage_el_path=passage_el,
        freebase_dir=freebase_dir
    )

    print(f"\nStats: {scorer.get_stats()}")

    store = scorer.entity_store
    sample_ids = list(store._passage_data.keys())[:5]

    if len(sample_ids) >= 2:
        d1, d2 = sample_ids[0], sample_ids[1]
        names1 = store.get_passage_names(d1)
        names2 = store.get_passage_names(d2)

        print(f"\nDoc {d1} entities: {names1}")
        print(f"Doc {d2} entities: {names2}")

        overlap = scorer.compute_entity_overlap(names1, names2)
        print(f"Entity overlap: {overlap:.4f}")

        if freebase_dir:
            mids1 = store.get_passage_mids(d1)
            mids2 = store.get_passage_mids(d2)
            kg_score = scorer.compute_kg_connectivity(mids1, mids2)
            print(f"KG connectivity: {kg_score:.4f}")

    print("\n✓ Test complete!")