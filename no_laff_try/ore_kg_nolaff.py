"""
KG-ORE (No LAFF weights) — Full ORE algorithm with KG scores replacing LAFF everywhere.

Changes from original ore_adaptive.py:
  1. Cluster affinity: KG scores instead of LAFF weights
  2. CER cluster head check: KG neighbor cache instead of LAFF graph
  3. Neighbor expansion: KG-rescored neighbors instead of raw LAFF
  4. Graph: uses bm25.128 corpus graph (not LAFF) for neighbor candidates
  5. All entity/Freebase data read from SQLite (no RAM-heavy structures)

Everything else (dual encoder, MonoT5, lambda learning, BM25 scoring) is identical to original ORE.
"""

from collections import Counter, defaultdict, OrderedDict
from typing import List
from statistics import mean
import torch
import numpy as np
import time
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

from no_laff_try.kg_scorer_nolaff import KGScorerUnified, create_scorer

# MS MARCO docstore for BM25 text scoring
dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

# BM25 scorer
existing_index = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
).indexref
existing_index = pt.IndexFactory.of(existing_index)
ret_scorer = pt.text.scorer(
    takes="docs", body_attr="text", wmodel="BM25",
    background_index=existing_index,
    controls={"termpipelines": "Stopwords,PorterStemmer"}
)


class OREAdaptiveKGUnified(pt.Transformer):
    def __init__(
        self,
        dual_encoder,
        scorer,
        corpus_index,
        graph: CorpusGraph,
        laff_graph: CorpusGraph,  # bm25.128 (NOT laff)
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
        per_query_cache_max: int = 500,
    ):
        self.scorer = scorer
        self.graph = graph
        self.laff_graph = laff_graph  # bm25.128 graph
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
        self._per_query_cache_max = per_query_cache_max

    def _get_doc_text(self, docno: str) -> str:
        try:
            return docstore.get(docno).text
        except:
            return ""

    def _get_kg_neighbor_scores(self, docno: str, cache: OrderedDict) -> dict:
        """Get KG-rescored neighbors from bm25.128 graph. Cached per query."""
        if docno in cache:
            cache.move_to_end(docno)
            return cache[docno]

        neighbors, weights = self.laff_graph.neighbours(docno, weights=True)
        neighbor_docnos = [str(n) for n in neighbors]

        rescored = self.kg_scorer.rescore_neighbors(
            docno, neighbor_docnos, np.asarray(weights),
            topk=self.kg_neighbor_k
        )

        score_dict = {n: sc for n, sc, _ in rescored}

        cache[docno] = score_dict
        cache.move_to_end(docno)
        if len(cache) > self._per_query_cache_max:
            cache.popitem(last=False)

        return score_dict

    def _compute_kg_cluster_lookup(self, cluster_heads, cache):
        """Build cluster neighbor lookup using KG scores (replaces LAFF affinity)."""
        lookup = defaultdict(list)
        for head in cluster_heads:
            kg_scores = self._get_kg_neighbor_scores(str(head), cache)
            for nb, sc in kg_scores.items():
                lookup[nb].append(sc)
        return {k: mean(v) for k, v in lookup.items()}

    def estimate_bm25_score_batch(self, qids, queries, docids):
        batch = []
        for qid, query, docid in zip(qids, queries, docids):
            batch.append([qid, query, docid, self._get_doc_text(docid)])
        df = pd.DataFrame(batch, columns=["qid", "query", "docno", "text"])
        result_df = ret_scorer(df)
        return list(result_df["docno"].values), list(result_df["score"].values)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))

        lambda_param = 0.65
        lambda_param_1 = 0.45
        lambda_param_2 = 0.65

        for i, (query, initial_results) in enumerate(groups):
            qid = initial_results['qid'].iloc[0]
            initial_results = initial_results.sort_values('score', ascending=False)

            # Per-query bounded cache
            kg_cache = OrderedDict()

            arms = [
                ArmKG(docid, name='initial_results_' + str(docid))
                for docid in initial_results['docno'].tolist()[:self.budget]
            ]

            results = {}
            bm25_scores = dict(zip(
                initial_results["docno"].values,
                initial_results["score"].values
            ))

            count = 0
            prev_heads = []

            while len(arms) > 0 and len(results) < self.budget:
                if count == 0:
                    arm = sorted(
                        arms, key=lambda x: x.estimate_utility(), reverse=True
                    )[:self.batch_size]
                else:
                    cluster_heads = [
                        doc for doc, _ in Counter(results).most_common(self.top_s)
                    ]

                    # KG-enhanced cluster lookup (replaces LAFF affinity)
                    cluster_neigh_lookup = self._compute_kg_cluster_lookup(cluster_heads, kg_cache)

                    filtered_arms = [a for a in arms if a.docnos[-1] not in results]

                    # BM25 scoring for missing docs
                    if len(bm25_scores) < (len(initial_results) + self.num_bm25_calls):
                        donos_missing = [
                            x.docnos[-1] for x in filtered_arms
                            if x.docnos[-1] not in bm25_scores
                        ]
                        if len(donos_missing) > 0:
                            qids_list = len(donos_missing) * [qid]
                            queries_list = len(donos_missing) * [query]
                            docnos_ret, scores_ret = self.estimate_bm25_score_batch(
                                qids_list, queries_list, donos_missing
                            )
                            bm25_scores.update(dict(zip(docnos_ret, scores_ret)))

                    # Arm selection using KG cluster scores
                    neighbor_criteria_arms = [
                        a for a in filtered_arms if a.docnos[-1] in cluster_neigh_lookup
                    ]
                    criteria_scores = [
                        (a, cluster_neigh_lookup.get(a.docnos[-1], 0))
                        for a in neighbor_criteria_arms
                    ]
                    new_arms = [
                        a for a, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])
                    ]

                    remaining_arms = [
                        (a, bm25_scores.get(a.docnos[-1], 0))
                        for a in neighbor_criteria_arms
                        if a.docnos[-1] in bm25_scores and a not in new_arms
                    ]
                    bm25_arms = [
                        a for a, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])
                    ]
                    new_arms.extend(bm25_arms)

                    if len(new_arms) == 0:
                        new_arms = filtered_arms

                    # CER estimation with KG-based cluster head checking
                    if prev_heads == cluster_heads:
                        arm = sorted(
                            new_arms,
                            key=lambda x: (
                                x.cer_scores[x.docnos[-1]]
                                if x.docnos[-1] in x.cer_scores
                                else x.estimate_cer_score(
                                    qid, query, x.docnos, results,
                                    self.graph, self.laff_graph, initial_results,
                                    bm25_scores, cluster_heads,
                                    lambda_param, lambda_param_1, lambda_param_2,
                                    cluster_neigh_lookup, kg_cache
                                )
                            ),
                            reverse=True
                        )[:self.batch_size]
                    else:
                        new_arms = list(set(new_arms))
                        cer_scores_list = [
                            x.estimate_cer_score(
                                qid, query, x.docnos, results,
                                self.graph, self.laff_graph, initial_results,
                                bm25_scores, cluster_heads,
                                lambda_param, lambda_param_1, lambda_param_2,
                                cluster_neigh_lookup, kg_cache
                            )
                            for x in new_arms
                        ]
                        arm = sorted(
                            zip(new_arms, cer_scores_list),
                            key=lambda x: x[1], reverse=True
                        )[:self.batch_size]
                        arm = [x for x, _ in arm]

                docnos_final = [x.docnos[-1] for x in arm]
                all_docnos = [x.docnos[-1] for x in arms]

                if count > 0:
                    estimated_rank_scores = [
                        x.cer_scores[x.docnos[-1]]
                        for x in arm if x.docnos[-1] in x.cer_scores
                    ]

                # Cross-encoder scoring (MonoT5 + dual encoder)
                if len(results) < min(self.batch_size * self.cross_enc_budget, self.budget):
                    with torch.no_grad():
                        query_vecs = self.dual_encoder.encode_queries([query])[0].reshape(1, -1)

                    doc_object = [{"docno": docno} for docno in docnos_final]
                    doc_vecs = np.concatenate([
                        dv.reshape(1, -1)
                        for dv in self.corpus_index.vec_loader()(pd.DataFrame(doc_object))["doc_vec"].values
                    ])

                    dual_score = (query_vecs.dot(doc_vecs.T))[0]

                    batch_df = pd.DataFrame(docnos_final, columns=['docno'])
                    batch_df['qid'] = qid
                    batch_df['query'] = query
                    reranked_scores = list(self.scorer(batch_df)["score"].values)

                    ranked_set_scores = [
                        x + s for x, s in zip(reranked_scores, dual_score)
                    ]

                    # Online lambda parameter learning
                    if count > 0:
                        bm25_features = np.array([
                            x.bm25_scores.get(x.docnos[-1], 0) for x in arm
                        ]).reshape(-1, 1)
                        affinity_features = np.array([
                            x.estimates[x.docnos[-1]] for x in arm
                        ]).reshape(-1, 1)
                        neighbor_score_features = np.array([
                            x.cross_enc_avg[x.docnos[-1]] for x in arm
                        ]).reshape(-1, 1)

                        features = np.concatenate(
                            (bm25_features, affinity_features, neighbor_score_features),
                            axis=1
                        )
                        params = scipy.optimize.lsq_linear(
                            features, ranked_set_scores,
                            lsq_solver="exact", bounds=self.param_bounds
                        )
                        lambda_param = params["x"][0]
                        lambda_param_1 = params["x"][1]
                        lambda_param_2 = params["x"][2]
                else:
                    ranked_set_scores = estimated_rank_scores

                # Update results
                for x, docno, score_value in zip(arm, docnos_final, ranked_set_scores):
                    results[docno] = score_value
                    x.push(score_value)

                # KG-Enhanced Neighbor Expansion
                if len(results) < self.budget:
                    S_2 = Counter(results).most_common(self.top_s2)
                    S2 = [doc[0] for doc in S_2]
                    neighbor_lookup = set(S2).intersection(set(docnos_final))

                    for docno in neighbor_lookup:
                        kg_scores = self._get_kg_neighbor_scores(str(docno), kg_cache)
                        sorted_neighbors = sorted(
                            kg_scores.items(), key=lambda x: x[1], reverse=True
                        )

                        for neighbor, kg_score in sorted_neighbors[:self.kg_neighbor_k]:
                            if neighbor not in all_docnos:
                                neighbor_arm = ArmKG(neighbor, name=f'neighbors_{docno}')
                                neighbor_arm.push(score_value)
                                neighbor_arm.kg_scores[neighbor] = kg_score
                                arms.append(neighbor_arm)
                                all_docnos.append(neighbor)

                if count > 0:
                    prev_heads = cluster_heads
                else:
                    prev_heads = []
                count += 1

                arms = [a for a in arms if not a.is_exhausted()]

            # Build final results
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
    """Arm class — identical to original ORE Arm, plus kg_scores tracking."""

    def __init__(self, docnos: str, name: str = ''):
        self.docnos = [str(docnos)]
        self.scores = []
        self.estimated_scores = []
        self.name = name
        self.cer_scores = {}
        self.bm25_scores = {}
        self.laff_scores = {}
        self.estimates = {}
        self.cross_enc_avg = {}
        self.kg_scores = {}

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
        self, qid, query, docnos, results,
        neigh_graph, graph, initial_results,
        bm25_score_dict, cluster_heads,
        lambda_param, lambda_1, lambda_2,
        cluster_neigh_lookup,
        kg_cache=None,
    ):
        """
        CER with KG-based cluster head checking.

        KEY CHANGE: Instead of LAFF graph to find valid cluster heads,
        uses KG neighbor cache. A cluster head is valid if doc has
        a non-zero KG score to that head.
        """
        doc = docnos[-1]

        if doc in bm25_score_dict:
            self.bm25_scores[doc] = bm25_score_dict[doc]
            bm25_score = bm25_score_dict[doc]
        else:
            bm25_score = 0

        crss_enc_scores = []

        # Cluster head check (NO LAFF):
        # Use bm25.128 graph adjacency so CER utility doesn't become sparse.
        try:
            neighs, _w = graph.neighbours(doc, weights=True)   # graph should be bm25.128 here
            neigh_set = set(map(str, neighs))
            valid_cluster_heads = [h for h in cluster_heads if str(h) in neigh_set]
        except Exception:
            valid_cluster_heads = []

        crss_enc_scores.extend(results[h] for h in valid_cluster_heads if h in results)

        # Affinity estimate (cluster_neigh_lookup already KG-enhanced)
        self.estimated_scores = lambda_param * bm25_score + lambda_1 * cluster_neigh_lookup.get(doc, 0)

        if len(crss_enc_scores) > 0:
            score_utility = sum(crss_enc_scores) / len(crss_enc_scores)
        else:
            score_utility = self.estimate_utility()
            if score_utility == float("-inf"):
                score_utility = 0

        self.cross_enc_avg[docnos[-1]] = score_utility
        self.estimates[docnos[-1]] = self.estimated_scores
        cer = self.estimated_scores + lambda_2 * score_utility
        self.cer_scores[docnos[-1]] = cer
        return cer


def create_ore_kg(
    dual_encoder,
    scorer,
    corpus_index,
    graph,
    laff_graph,  # bm25.128 graph (not LAFF)
    kg_alpha: float = 0.0,
    kg_beta: float = 0.5,
    kg_gamma: float = 0.5,
    kg_score_mode: str = 'log',
    freebase_1hop_db: str = None,
    passage_el_db: str = None,
    **kwargs
):
    kg_scorer = create_scorer(
        alpha=kg_alpha,
        beta=kg_beta,
        gamma=kg_gamma,
        kg_score_mode=kg_score_mode,
        passage_el_db=passage_el_db,
        freebase_1hop_db=freebase_1hop_db,
    )

    return OREAdaptiveKGUnified(
        dual_encoder=dual_encoder,
        scorer=scorer,
        corpus_index=corpus_index,
        graph=graph,
        laff_graph=laff_graph,
        kg_scorer=kg_scorer,
        **kwargs
    )