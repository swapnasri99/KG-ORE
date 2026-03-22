from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from statistics import mean
import torch
import numpy as np
import heapq
import scipy
import random

random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_adaptive import CorpusGraph
import ir_datasets

#from kg_scorer_unified_patched import KGScorerUnified, create_scorer
# Load MS MARCO docstore (for BM25 scoring text)
from kg_scorer_unified_query_gated import create_scorer, KGScorerUnified

# Load MS MARCO docstore (for BM25 scoring text)
dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

# BM25 scorer
existing_index = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
).indexref
existing_index = pt.IndexFactory.of(existing_index)
ret_scorer = pt.text.scorer(
    takes='docs', body_attr='text', wmodel='BM25',
    background_index=existing_index,
    controls={'termpipelines': 'Stopwords,PorterStemmer'}
)


class OREAdaptiveKGUnified(pt.Transformer):
    def __init__(
        self,
        dual_encoder,
        scorer,
        corpus_index,
        graph: CorpusGraph,
        laff_graph: CorpusGraph,
        kg_scorer: KGScorerUnified,
        *,
        budget: int = 100,
        verbose: bool = False,
        batch_size: int = 16,
        num_bm25_calls: int = 10,
        top_s: int = 25,
        top_s2: int = 15,
        cross_enc_budget: int = 2,
        param_bounds: tuple = (0.25, 0.9),
        kg_neighbor_k: int = 16,
        use_kg_in_cer: bool = False,
        kg_cer_weight_init: float = 0.20,
        neighbor_mode: str = 'kg_laff',  # 'kg_laff' or 'union'
        qrels_map = None,
    ):
        self.scorer = scorer
        self.graph = graph
        self.laff_graph = laff_graph
        self.budget = budget
        self.verbose = verbose
        self.corpus_index = corpus_index
        self.batch_size = batch_size
        self.num_bm25_calls = num_bm25_calls
        self.cross_enc_budget = cross_enc_budget
        self.top_s = top_s
        self.top_s2 = top_s2
        self.param_bounds = param_bounds
        self.dual_encoder = dual_encoder
        self.kg_scorer = kg_scorer
        self.kg_neighbor_k = kg_neighbor_k
        self.use_kg_in_cer = use_kg_in_cer
        self.kg_cer_weight_init = kg_cer_weight_init
        self.neighbor_mode = neighbor_mode
        self.qrels_map = qrels_map or {}

        if self.neighbor_mode not in ('kg_laff', 'union', 'union3', 'kg_only'):
            raise ValueError(f"neighbor_mode must be 'kg_laff', 'union', 'union3', or 'kg_only', got '{neighbor_mode}'")

        self._doc_text_cache = {}
        self._kg_debug_prints = 0

    def _get_doc_text(self, docno: str) -> str:
        if docno not in self._doc_text_cache:
            try:
                self._doc_text_cache[docno] = docstore.get(docno).text
            except Exception:
                self._doc_text_cache[docno] = ''
        return self._doc_text_cache[docno]

    def _get_kg_enhanced_neighbors(
        self,
        qid: str,
        docno: str,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> List[Tuple[str, float, object]]:
        neighbor_docnos = [str(n) for n in neighbors]
        rescored = self.kg_scorer.rescore_neighbors(
            docno, neighbor_docnos, weights, qid=str(qid),query_gate_mode="intersection",
        )

        if self.verbose and self._kg_debug_prints < 3:
            self._kg_debug_prints += 1
            print(
                '\n[KG DEBUG] qid=', qid, 'docno=', docno,
                'top neighbors (laff_norm, overlap, kg_raw, kg_norm, combined)'
            )
            for n_docno, score, comp in rescored[:10]:
                print(
                    f'  n={n_docno}  laff={comp.laff_norm:.3f}  '
                    f'ov={comp.entity_overlap:.3f}  '
                    f'kg_raw={comp.kg_raw:.6f}  kg_norm={comp.kg_connectivity:.6f}  '
                    f'comb={comp.combined:.3f}'
                )

        return rescored

    def _get_union_neighbors(
        self,
        qid: str,
        docno: str,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[List[Tuple[str, float, object]], Dict[str, str]]:
        """
        Union neighbor selection with configurable legs and provenance tracking.

        neighbor_mode='union':    LAFF ∪ D2D
        neighbor_mode='union3':   LAFF ∪ D2D ∪ D2Q
        neighbor_mode='kg_only':  D2D ∪ D2Q (no LAFF leg)

        Returns:
            union_rescored: list of (neighbor_docno, combined_score, comp)
            source_map: dict neighbor_docno -> provenance label
        """
        neighbor_docnos = [str(n) for n in neighbors]

        # --- EL Coverage check (first few queries) ---
        if self.verbose and self._kg_debug_prints < 3:
            covered = sum(1 for n in neighbor_docnos
                          if self.kg_scorer.entity_store.has_passage(n))
            print(f'[EL COVERAGE] qid={qid} doc={docno}: '
                  f'{covered}/{len(neighbor_docnos)} neighbors have entities '
                  f'({100*covered/len(neighbor_docnos):.0f}%)')

        # --- Leg 1: Raw LAFF top-k (skip if kg_only) ---
        if self.neighbor_mode != 'kg_only':
            laff_top = set(neighbor_docnos[:self.kg_neighbor_k])
        else:
            laff_top = set()

        # --- Leg 2: KG-D2D rescored top-k ---
        rescored = self.kg_scorer.rescore_neighbors(
            docno, neighbor_docnos, weights, qid=str(qid)
        )
        kg_d2d_top = set(n for n, _, _ in rescored[:self.kg_neighbor_k])

        # --- Leg 3: KG-D2Q top-k (only for union3 / kg_only) ---
        kg_d2q_top = set()
        if self.neighbor_mode in ('union3', 'kg_only'):
            d2q_scores = self.kg_scorer.score_neighbors_d2q(
                qid, neighbor_docnos, weights
            )
            kg_d2q_top = set(
                n for n, score in d2q_scores[:self.kg_neighbor_k] if score > 0
            )

        # --- Union ---
        union_set = laff_top | kg_d2d_top | kg_d2q_top

        # --- Build fine-grained source map ---
        source_map = {}
        for n in union_set:
            in_laff = n in laff_top
            in_d2d = n in kg_d2d_top
            in_d2q = n in kg_d2q_top

            if in_d2q and not in_d2d and not in_laff:
                source_map[n] = 'd2q_only'
            elif in_d2d and not in_laff and not in_d2q:
                source_map[n] = 'kg_only'       # D2D only (same label as before)
            elif in_laff and not in_d2d and not in_d2q:
                source_map[n] = 'laff_only'
            elif in_d2d and in_d2q and not in_laff:
                source_map[n] = 'd2d+d2q'
            elif in_laff and in_d2d and not in_d2q:
                source_map[n] = 'both'           # LAFF+D2D (same label as before)
            elif in_laff and in_d2q and not in_d2d:
                source_map[n] = 'laff+d2q'
            else:
                source_map[n] = 'all_three'      # in all legs

        if self.verbose and self._kg_debug_prints < 3:
            self._kg_debug_prints += 1
            from collections import Counter as Ctr
            src_counts = Ctr(source_map.values())
            print(f'\n[UNION DEBUG] qid={qid} docno={docno} mode={self.neighbor_mode}')
            print(f'  LAFF: {len(laff_top)}, D2D: {len(kg_d2d_top)}, D2Q: {len(kg_d2q_top)}')
            print(f'  Union: {len(union_set)}')
            for src, cnt in sorted(src_counts.items()):
                print(f'    {src}: {cnt}')

        union_rescored = [(n, s, c) for n, s, c in rescored if n in union_set]
        return union_rescored, source_map

    def _get_selected_neighbors(
        self,
        qid: str,
        docno: str,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[List[Tuple[str, float, object]], Dict[str, str]]:
        """
        Returns:
            selected_neighbors: list of (neighbor_docno, combined_score, comp)
            source_map: dict neighbor_docno -> source label
        """
        if self.neighbor_mode in ('union', 'union3', 'kg_only'):
            return self._get_union_neighbors(qid, docno, neighbors, weights)
        else:
            rescored = self._get_kg_enhanced_neighbors(qid, docno, neighbors, weights)
            selected = rescored[:self.kg_neighbor_k]
            source_map = {n: 'kg_laff' for n, _, _ in selected}
            return selected, source_map

    def _compute_kg_enhanced_cluster_lookup(self, qid: str, cluster_heads):
        """
        Returns three query-conditioned lookups aggregated over cluster heads:
          1) combined_lookup: mean combined KG-filter score for selection/filtering
          2) laff_lookup:    mean laff_norm among retained top-k neighbors
          3) kg_only_lookup: mean kg_connectivity among retained top-k neighbors
        """
        combined_lookup = defaultdict(list)
        laff_lookup = defaultdict(list)
        kg_only_lookup = defaultdict(list)

        for cluster_head in cluster_heads:
            neighbors, weights = self.laff_graph.neighbours(cluster_head, weights=True)
            selected, _ = self._get_selected_neighbors(qid, cluster_head, neighbors, weights)

            for neighbor_docno, combined_score, comp in selected:
                combined_lookup[neighbor_docno].append(float(combined_score))
                laff_lookup[neighbor_docno].append(float(comp.laff_norm))
                kg_only_lookup[neighbor_docno].append(float(comp.kg_connectivity))

        return (
            {k: mean(v) for k, v in combined_lookup.items()},
            {k: mean(v) for k, v in laff_lookup.items()},
            {k: mean(v) for k, v in kg_only_lookup.items()},
        )

    def estimate_bm25_score_batch(self, qids, queries, docids):
        batch = []
        for qid, query, docid in zip(qids, queries, docids):
            batch.append([qid, query, docid, self._get_doc_text(docid)])
        df = pd.DataFrame(batch, columns=['qid', 'query', 'docno', 'text'])
        result_df = ret_scorer(df)
        return list(result_df['docno'].values), list(result_df['score'].values)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))

        lambda_bm25 = 0.65
        lambda_aff = 0.45
        lambda_ce = 0.65
        lambda_kg = self.kg_cer_weight_init

        for _, (query, initial_results) in enumerate(groups):
            qid = initial_results['qid'].iloc[0]
            initial_results = initial_results.sort_values('score', ascending=False)

            arms = [
                ArmKG(docid, name='initial_results_' + docid)
                for docid in initial_results['docno'].tolist()[:self.budget]
            ]

            results = {}
            bm25_scores = dict(zip(initial_results['docno'].values, initial_results['score'].values))
            doc_source = {
                docid: 'initial_bm25'
                for docid in initial_results['docno'].tolist()[:self.budget]
            }

            count = 0
            prev_heads = []

            while len(arms) > 0 and len(results) < self.budget:
                if count == 0:
                    arm = sorted(arms, key=lambda x: x.estimate_utility(), reverse=True)[:self.batch_size]
                else:
                    cluster_heads = [doc for doc, _ in Counter(results).most_common(self.top_s)]
                    combined_lookup, laff_lookup, kg_only_lookup = self._compute_kg_enhanced_cluster_lookup(qid, cluster_heads)

                    filtered_arms = [a for a in arms if a.docnos[-1] not in results]

                    # Score BM25 for newly expanded docs too.
                    missing_docnos = [a.docnos[-1] for a in filtered_arms if a.docnos[-1] not in bm25_scores]
                    if missing_docnos:
                        limit = min(len(missing_docnos), self.num_bm25_calls) if self.num_bm25_calls is not None else len(missing_docnos)
                        if limit > 0:
                            docnos_ret, scores_ret = self.estimate_bm25_score_batch(
                                [qid] * limit,
                                [query] * limit,
                                missing_docnos[:limit],
                            )
                            bm25_scores.update(dict(zip(docnos_ret, scores_ret)))

                    neighbor_criteria_arms = [a for a in filtered_arms if a.docnos[-1] in combined_lookup]
                    criteria_scores = [(a, combined_lookup.get(a.docnos[-1], 0.0)) for a in neighbor_criteria_arms]
                    new_arms = [a for a, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]

                    remaining_arms = [
                        (a, bm25_scores.get(a.docnos[-1], 0.0))
                        for a in filtered_arms
                        if a not in new_arms and a.docnos[-1] in bm25_scores
                    ]
                    bm25_arms = [a for a, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]
                    new_arms.extend(bm25_arms)

                    if len(new_arms) == 0:
                        new_arms = filtered_arms

                    # De-duplicate while preserving order
                    new_arms = list(dict.fromkeys(new_arms))

                    if prev_heads == cluster_heads:
                        arm = sorted(
                            new_arms,
                            key=lambda x: (
                                x.cer_scores[x.docnos[-1]] if x.docnos[-1] in x.cer_scores else x.estimate_cer_score(
                                    qid, query, x.docnos, results,
                                    self.graph, self.laff_graph,
                                    bm25_scores, cluster_heads,
                                    lambda_bm25, lambda_aff, lambda_ce,
                                    laff_lookup, kg_only_lookup,
                                    use_kg_in_cer=self.use_kg_in_cer,
                                    lambda_kg=lambda_kg,
                                )
                            ),
                            reverse=True,
                        )[:self.batch_size]
                    else:
                        cer_scores_list = [
                            x.estimate_cer_score(
                                qid, query, x.docnos, results,
                                self.graph, self.laff_graph,
                                bm25_scores, cluster_heads,
                                lambda_bm25, lambda_aff, lambda_ce,
                                laff_lookup, kg_only_lookup,
                                use_kg_in_cer=self.use_kg_in_cer,
                                lambda_kg=lambda_kg,
                            )
                            for x in new_arms
                        ]
                        arm = [
                            x for x, _ in sorted(
                                zip(new_arms, cer_scores_list), key=lambda x: x[1], reverse=True
                            )[:self.batch_size]
                        ]

                docnos_final = [x.docnos[-1] for x in arm]
                all_docnos = [x.docnos[-1] for x in arms]

                if count > 0:
                    estimated_rank_scores = [x.cer_scores[x.docnos[-1]] for x in arm if x.docnos[-1] in x.cer_scores]

                if len(results) < min(self.batch_size * self.cross_enc_budget, self.budget):
                    with torch.no_grad():
                        query_vecs = self.dual_encoder.encode_queries([query])[0].reshape(1, -1)

                    doc_object = [{'docno': docno} for docno in docnos_final]
                    doc_vecs = np.concatenate([
                        dv.reshape(1, -1)
                        for dv in self.corpus_index.vec_loader()(pd.DataFrame(doc_object))['doc_vec'].values
                    ])

                    dual_score = (query_vecs.dot(doc_vecs.T))[0]

                    batch_df = pd.DataFrame(docnos_final, columns=['docno'])
                    batch_df['qid'] = qid
                    batch_df['query'] = query
                    reranked_scores = list(self.scorer(batch_df)['score'].values)
                    ranked_set_scores = [x + s for x, s in zip(reranked_scores, dual_score)]

                    if count > 0:
                        bm25_features = np.array([x.bm25_scores.get(x.docnos[-1], 0.0) for x in arm]).reshape(-1, 1)
                        affinity_features = np.array([x.estimates[x.docnos[-1]] for x in arm]).reshape(-1, 1)
                        neighbor_score_features = np.array([x.cross_enc_avg[x.docnos[-1]] for x in arm]).reshape(-1, 1)

                        if self.use_kg_in_cer:
                            kg_features = np.array([x.kg_features.get(x.docnos[-1], 0.0) for x in arm]).reshape(-1, 1)
                            features = np.concatenate(
                                (bm25_features, affinity_features, neighbor_score_features, kg_features), axis=1
                            )
                            params = scipy.optimize.lsq_linear(
                                features, ranked_set_scores, lsq_solver='exact', bounds=self.param_bounds
                            )
                            lambda_bm25, lambda_aff, lambda_ce, lambda_kg = params['x']
                        else:
                            features = np.concatenate(
                                (bm25_features, affinity_features, neighbor_score_features), axis=1
                            )
                            params = scipy.optimize.lsq_linear(
                                features, ranked_set_scores, lsq_solver='exact', bounds=self.param_bounds
                            )
                            lambda_bm25, lambda_aff, lambda_ce = params['x']
                else:
                    ranked_set_scores = estimated_rank_scores

                for x, docno, score_value in zip(arm, docnos_final, ranked_set_scores):
                    results[docno] = score_value
                    x.push(score_value)

                if len(results) < self.budget:
                    S_2 = Counter(results).most_common(self.top_s2)
                    S2 = [doc[0] for doc in S_2]
                    neighbor_lookup = set(S2).intersection(set(docnos_final))
                    parent_score_lookup = dict(zip(docnos_final, ranked_set_scores))

                    for docno in neighbor_lookup:
                        parent_score = parent_score_lookup.get(docno, results.get(docno, 0.0))
                        neighbors, weights = self.laff_graph.neighbours(docno, weights=True)
                        selected_neighbors, source_map = self._get_selected_neighbors(qid, docno, neighbors, weights)

                        for neighbor, kg_score, comp in selected_neighbors:
                            if neighbor not in all_docnos:
                                neighbor_arm = ArmKG(neighbor, name=f'neighbors_{docno}')
                                neighbor_arm.push(parent_score)
                                neighbor_arm.kg_scores[neighbor] = float(kg_score)
                                neighbor_arm.kg_features[neighbor] = float(comp.kg_connectivity)
                                arms.append(neighbor_arm)
                                all_docnos.append(neighbor)

                                if neighbor not in doc_source:
                                    doc_source[neighbor] = source_map.get(neighbor, 'unknown')

                prev_heads = cluster_heads if count > 0 else []
                count += 1
                arms = [a for a in arms if not a.is_exhausted()]

            final_ranked_docs = [docno for docno, _ in Counter(results).most_common()]
            top50 = final_ranked_docs[:50]

            relevant_docs = self.qrels_map.get(str(qid), set())

            # Group docs by source
            source_groups = defaultdict(list)
            for rank, d in enumerate(top50, start=1):
                src = doc_source.get(d, 'unknown')
                source_groups[src].append((rank, d))

            # Count relevant per source
            source_rel_counts = {}
            for src, docs in source_groups.items():
                rel = sum(1 for _, d in docs if d in relevant_docs)
                source_rel_counts[src] = (len(docs), rel)

            print('\n' + '=' * 70)
            print(f'[KG TOP50 DEBUG] qid={qid}  mode={self.neighbor_mode}')
            print(f'Final top-50 docs: {len(top50)}')
            print(f'\nProvenance breakdown:')
            for src in sorted(source_rel_counts.keys()):
                total, rel = source_rel_counts[src]
                prec = rel / total if total > 0 else 0
                print(f'  {src:15s}: {total:3d} docs, {rel:3d} relevant (precision={prec:.3f})')

            # Detailed: KG-only docs
            kg_only_docs = source_groups.get('kg_only', [])
            if kg_only_docs:
                print(f'\nKG-only (D2D) docs in top-50:')
                for rank, d in kg_only_docs:
                    rel_flag = 'REL' if d in relevant_docs else 'NOT_REL'
                    print(f'  rank={rank:2d}  docno={d}  {rel_flag}')

            # Detailed: D2Q-only docs
            d2q_only_docs = source_groups.get('d2q_only', [])
            if d2q_only_docs:
                print(f'\nD2Q-only docs in top-50:')
                for rank, d in d2q_only_docs:
                    rel_flag = 'REL' if d in relevant_docs else 'NOT_REL'
                    print(f'  rank={rank:2d}  docno={d}  {rel_flag}')

            # Detailed: D2D+D2Q docs
            d2d_d2q_docs = source_groups.get('d2d+d2q', [])
            if d2d_d2q_docs:
                print(f'\nD2D+D2Q docs in top-50:')
                for rank, d in d2d_d2q_docs:
                    rel_flag = 'REL' if d in relevant_docs else 'NOT_REL'
                    print(f'  rank={rank:2d}  docno={d}  {rel_flag}')

            print(f'\nTop-50 provenance + relevance:')
            for rank, d in enumerate(top50, start=1):
                src = doc_source.get(d, 'unknown')
                rel_flag = 'REL' if d in relevant_docs else 'NOT_REL'
                print(f'  rank={rank:2d}  docno={d}  source={src}  {rel_flag}')

            print('=' * 70)

            for rank, (docno, final_score) in enumerate(Counter(results).most_common()):
                result_builder.extend({
                    'qid': qid,
                    'query': query,
                    'docno': docno,
                    'score': final_score,
                    'rank': rank,
                })

        return result_builder.to_df()


class ArmKG:
    def __init__(self, docnos: str, name: str = ''):
        self.docnos = [docnos]
        self.scores = []
        self.estimated_scores = []
        self.name = name
        self.cer_scores: Dict[str, float] = {}
        self.bm25_scores: Dict[str, float] = {}
        self.laff_scores: Dict[str, float] = {}
        self.estimates: Dict[str, float] = {}
        self.cross_enc_avg: Dict[str, float] = {}
        self.kg_scores: Dict[str, float] = {}
        self.kg_features: Dict[str, float] = {}

    def is_exhausted(self):
        return len(self.docnos) == 0

    def pull(self):
        assert not self.is_exhausted()
        next_docno, self.docnos = self.docnos[0], []
        return next_docno

    def push(self, score: float):
        self.scores.append(score)

    def estimate_utility(self):
        if len(self.scores) == 0:
            return float('-inf')
        return sum(self.scores) / len(self.scores)

    def estimate_cer_score(
        self,
        qid,
        query,
        docnos,
        results,
        neigh_graph,
        graph,
        bm25_score_dict,
        cluster_heads,
        lambda_bm25,
        lambda_aff,
        lambda_ce,
        cluster_laff_lookup,
        cluster_kg_lookup,
        *,
        use_kg_in_cer: bool = False,
        lambda_kg: float = 0.0,
    ):
        doc = docnos[-1]

        bm25_score = float(bm25_score_dict.get(doc, 0.0))
        self.bm25_scores[doc] = bm25_score

        neigh_support = float(cluster_laff_lookup.get(doc, 0.0))
        kg_term = float(cluster_kg_lookup.get(doc, 0.0))
        self.kg_features[doc] = kg_term

        laff_score_dict = dict(zip(*graph.neighbours(doc, weights=True)))
        crss_enc_scores = []
        valid_cluster_heads = [res for res in cluster_heads if res in laff_score_dict]
        crss_enc_scores.extend(results[res] for res in valid_cluster_heads)

        self.estimated_scores = lambda_bm25 * bm25_score + lambda_aff * neigh_support

        if len(crss_enc_scores) > 0:
            score_utility = sum(crss_enc_scores) / len(crss_enc_scores)
        else:
            score_utility = self.estimate_utility()
            if score_utility == float('-inf'):
                score_utility = 0.0

        self.cross_enc_avg[doc] = score_utility
        self.estimates[doc] = self.estimated_scores

        cer = (lambda_aff * self.estimated_scores) + (lambda_ce * score_utility)

        self.cer_scores[doc] = cer
        return cer



def create_ore_kg(
    dual_encoder,
    scorer,
    corpus_index,
    graph,
    laff_graph,
    kg_alpha: float = 0.5,
    kg_beta: float = 0.3,
    kg_gamma: float = 0.2,
    kg_score_mode: str = 'log',
    freebase_dir: str = None,
    passage_el_path: str = None,
    full_passage_el_path: str = None,
    query_el_path: str = None,
    passage_el_db: str = None,
    **kwargs,
):
    kg_scorer = create_scorer(
        alpha=kg_alpha,
        beta=kg_beta,
        gamma=kg_gamma,
        kg_score_mode=kg_score_mode,
        freebase_dir=freebase_dir,
        passage_el_path=passage_el_path,
        full_passage_el_path=full_passage_el_path,
        query_el_path=query_el_path,
        passage_el_db=passage_el_db,
        debug=bool(kwargs.get('verbose', False)),
        debug_print_limit=3,
    )

    return OREAdaptiveKGUnified(
        dual_encoder=dual_encoder,
        scorer=scorer,
        corpus_index=corpus_index,
        graph=graph,
        laff_graph=laff_graph,
        kg_scorer=kg_scorer,
        **kwargs,
    )