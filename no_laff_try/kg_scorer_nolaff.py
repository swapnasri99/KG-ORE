"""
KG Scorer — fully disk-backed, no LAFF dependency.

Everything runs from SQLite:
  - Entity store: passage_entities.db (from entity_store_disk.py)
  - Freebase 1-hop: freebase_1hop.db (from build_freebase_1hop_db.py)

No numpy matrices, no pickle files loaded into RAM at runtime.
"""
import math
import numpy as np
import sqlite3
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from Other_Files.entity_store_disk import DiskEntityStore


@dataclass
class ScoringComponents:
    graph_raw: float
    graph_norm: float
    entity_overlap: float
    kg_connectivity: float
    combined: float


class Freebase1HopDB:
    """
    Disk-backed 1-hop KG connectivity checks via SQLite.

    DB must be built with build_freebase_1hop_db.py which stores
    BIDIRECTIONAL edges (h→t AND t→h), so we only need to query
    one direction.
    """

    def __init__(self, db_path: str, cap_list: int = 50):
        self.db_path = db_path
        self.cap_list = cap_list

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA query_only=ON;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-100000;")  # ~100 MB cache

        total = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        print(f"  Freebase1HopDB: {total:,} edges (SQLite: {db_path})")

    def _cap(self, s: Set[str]) -> List[str]:
        if not s:
            return []
        return list(s)[:self.cap_list]

    def count_hits(self, mids_a: Set[str], mids_b: Set[str], cap_hits: int = 10) -> int:
        """
        Count how many unique mids in A have at least one edge to any mid in B.
        Since DB stores bidirectional edges, one-direction query is sufficient.
        """
        a = self._cap(mids_a)
        b = self._cap(mids_b)
        if not a or not b:
            return 0

        q = f"""
        SELECT COUNT(DISTINCT h)
        FROM edges
        WHERE h IN ({','.join(['?'] * len(a))})
          AND t IN ({','.join(['?'] * len(b))})
        """
        row = self.conn.execute(q, a + b).fetchone()
        hits = int(row[0]) if row and row[0] else 0
        return min(hits, cap_hits)

    def has_any_edge(self, mids_a: Set[str], mids_b: Set[str]) -> bool:
        a = self._cap(mids_a)
        b = self._cap(mids_b)
        if not a or not b:
            return False

        q = f"""
        SELECT 1 FROM edges
        WHERE h IN ({','.join(['?'] * len(a))})
          AND t IN ({','.join(['?'] * len(b))})
        LIMIT 1
        """
        return self.conn.execute(q, a + b).fetchone() is not None


class KGScorerUnified:
    """
    Disk-backed KG scorer. No RAM-heavy structures.

    combined = alpha * graph_norm + beta * entity_overlap + gamma * kg_connectivity

    When alpha=0: purely KG-driven (entity overlap + Freebase connectivity).
    """

    VALID_KG_MODES = {"binary", "count", "log", "coverage", "ratio", "raw", "minmax"}

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        kg_score_mode: str = "log",
        max_connections_cap: int = 10,
        entity_store=None,
        freebase_1hop_db: Optional[Freebase1HopDB] = None,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.kg_score_mode = kg_score_mode
        self.max_connections_cap = int(max_connections_cap)

        # Normalize weights
        total = self.alpha + self.beta + self.gamma
        if total > 0 and abs(total - 1.0) > 0.01:
            print(f"  Warning: weights sum to {total:.2f}, normalizing...")
            self.alpha /= total
            self.beta /= total
            self.gamma /= total

        if self.kg_score_mode not in self.VALID_KG_MODES:
            raise ValueError(f"kg_score_mode must be one of {sorted(self.VALID_KG_MODES)}")

        if entity_store is None:
            raise ValueError("entity_store required")

        self.entity_store = entity_store
        self.freebase_db = freebase_1hop_db if self.gamma > 0 else None

        if self.gamma > 0 and self.freebase_db is None:
            print("  WARNING: gamma > 0 but no freebase_1hop_db. Setting gamma=0.")
            self.beta += self.gamma
            self.gamma = 0.0

        print(f"  KGScorer: α={self.alpha:.2f} (graph) β={self.beta:.2f} (entity) γ={self.gamma:.2f} (KG)")
        print(f"  KG mode: {self.kg_score_mode}")

    def compute_entity_overlap(self, names_a: Set[str], names_b: Set[str]) -> float:
        if not names_a or not names_b:
            return 0.0
        inter = len(names_a & names_b)
        denom = min(len(names_a), len(names_b))
        return float(inter / denom) if denom > 0 else 0.0

    def compute_kg_connectivity(self, mids_a: Set[str], mids_b: Set[str]) -> float:
        if self.gamma <= 0 or self.freebase_db is None:
            return 0.0
        if not mids_a or not mids_b:
            return 0.0

        hits = self.freebase_db.count_hits(mids_a, mids_b, cap_hits=self.max_connections_cap)

        if hits == 0:
            return 0.0

        if self.kg_score_mode == "binary":
            return 1.0
        if self.kg_score_mode == "count":
            return min(float(hits) / self.max_connections_cap, 1.0)
        if self.kg_score_mode == "log":
            return min(math.log2(hits + 1) / 5.0, 1.0)
        if self.kg_score_mode == "coverage":
            denom = min(len(mids_a), self.max_connections_cap)
            return float(hits / denom) if denom > 0 else 0.0
        if self.kg_score_mode == "ratio":
            return float(hits / (len(mids_a) + 1e-9))
        if self.kg_score_mode in ("raw", "minmax"):
            return float(hits)  # minmax normalizes in rescore_neighbors

        return 0.0

    def rescore_neighbors(
        self,
        docno: str,
        neighbor_docnos: List[str],
        graph_weights: np.ndarray,
        topk: Optional[int] = None,
    ) -> List[Tuple[str, float, ScoringComponents]]:
        if len(neighbor_docnos) == 0:
            return []

        gmin = float(np.min(graph_weights))
        gmax = float(np.max(graph_weights))

        doc_names = self.entity_store.get_passage_names(docno)
        doc_mids = self.entity_store.get_passage_mids(docno) if self.gamma > 0 else set()

        if self.kg_score_mode == 'minmax' and self.gamma > 0:
            # Two-pass: compute raw KG, then min-max normalize
            raw_data = []
            for nb, w in zip(neighbor_docnos, graph_weights):
                nb = str(nb)
                nb_names = self.entity_store.get_passage_names(nb)
                nb_mids = self.entity_store.get_passage_mids(nb)

                if self.alpha > 0 and gmax > gmin:
                    g_norm = (float(w) - gmin) / (gmax - gmin)
                    g_norm = max(0.0, min(1.0, g_norm))
                else:
                    g_norm = 0.0

                eo = self.compute_entity_overlap(doc_names, nb_names)
                kg_raw = self.compute_kg_connectivity(doc_mids, nb_mids)

                raw_data.append({
                    'nb': nb, 'w': float(w), 'g_norm': g_norm,
                    'eo': eo, 'kg_raw': kg_raw,
                })

            kg_values = [d['kg_raw'] for d in raw_data]
            kg_min = min(kg_values)
            kg_max = max(kg_values)

            out = []
            for d in raw_data:
                if kg_max > kg_min:
                    kg_norm = (d['kg_raw'] - kg_min) / (kg_max - kg_min)
                else:
                    kg_norm = 0.0 if d['kg_raw'] == 0 else 1.0

                combined = self.alpha * d['g_norm'] + self.beta * d['eo'] + self.gamma * kg_norm

                comp = ScoringComponents(
                    graph_raw=d['w'], graph_norm=d['g_norm'],
                    entity_overlap=d['eo'], kg_connectivity=kg_norm,
                    combined=combined,
                )
                out.append((d['nb'], combined, comp))

        else:
            # Single-pass scoring
            out = []
            for nb, w in zip(neighbor_docnos, graph_weights):
                nb = str(nb)
                nb_names = self.entity_store.get_passage_names(nb)
                nb_mids = self.entity_store.get_passage_mids(nb) if self.gamma > 0 else set()

                if self.alpha > 0 and gmax > gmin:
                    g_norm = (float(w) - gmin) / (gmax - gmin)
                    g_norm = max(0.0, min(1.0, g_norm))
                else:
                    g_norm = 0.0

                eo = self.compute_entity_overlap(doc_names, nb_names)
                kg = self.compute_kg_connectivity(doc_mids, nb_mids)

                combined = self.alpha * g_norm + self.beta * eo + self.gamma * kg

                comp = ScoringComponents(
                    graph_raw=float(w), graph_norm=g_norm,
                    entity_overlap=eo, kg_connectivity=kg,
                    combined=combined,
                )
                out.append((nb, combined, comp))

        out.sort(key=lambda x: x[1], reverse=True)
        if topk is not None:
            out = out[:topk]
        return out

    def passage_count(self):
        return self.entity_store.passage_count()

    def get_stats(self) -> dict:
        return {
            'passages_with_entities': self.passage_count(),
            'weights': f"α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}",
            'kg_mode': self.kg_score_mode,
        }


def create_scorer(
    alpha: float,
    beta: float,
    gamma: float,
    kg_score_mode: str,
    passage_el_db: str,
    freebase_1hop_db: Optional[str] = None,
    max_connections_cap: int = 10,
) -> KGScorerUnified:
    """
    Create fully disk-backed scorer.
    """
    entity_store = DiskEntityStore(passage_el_db)

    fb = None
    if gamma > 0 and freebase_1hop_db:
        fb = Freebase1HopDB(freebase_1hop_db)

    return KGScorerUnified(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        kg_score_mode=kg_score_mode,
        max_connections_cap=max_connections_cap,
        entity_store=entity_store,
        freebase_1hop_db=fb,
    )