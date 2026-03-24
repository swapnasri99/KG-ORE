from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from statistics import mean
import heapq
import random

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import scipy
import torch
import ir_datasets
from pyterrier_adaptive import CorpusGraph

from kg_scorer_unified_corrected import KGScorerUnified, create_scorer

random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MS MARCO docstore (for BM25 scoring text)
dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

# BM25 scorer
existing_index = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
).indexref
existing_index = pt.IndexFactory.of(existing_index)
ret_scorer = pt.text.scorer(
    takes='docs',
    body_attr='text',
    wmodel='BM25',
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
        neighbor_mode: str = 'kg_laff',
        qrels_map=None,
        kg_bonus_k: int = 15,
        enable_post_ce_bonus: bool = True,
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
        self.kg_bonus_k = kg_bonus_k
        self.enable_post_ce_bonus = enable_post_ce_bonus

        if self.neighbor_mode not in ('kg_laff', 'union'):
            raise ValueError(f"neighbor_mode must be 'kg_laff' or 'union', got '{neighbor_mode}'")

        self._doc_text_cache = {}

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
            docno, neighbor_docnos, weights, qid=str(qid)
        )
        return rescored

    def _get_union_neighbors(
        self,
        qid: str,
        docno: str,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[List[Tuple[str, float, object]], Dict[str, Dict[str, bool]]]:
        neighbor_docnos = [str(n) for n in neighbors]

        laff_top = set(neighbor_docnos[:self.kg_neighbor_k])

        rescored = self.kg_scorer.rescore_neighbors(
            docno, neighbor_docnos, weights, qid=str(qid)
        )
        kg_top = set(n for n, _, _ in rescored[:self.kg_neighbor_k])

        union_set = laff_top | kg_top
        union_rescored = [(n, s, c) for n, s, c in rescored if n in union_set]

        membership = {}
        for n in union_set:
            membership[n] = {
                'in_laff': n in laff_top,
                'in_kg': n in kg_top,
            }

        return union_rescored, membership

    def _get_selected_neighbors(
        self,
        qid: str,
        docno: str,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ):
        if self.neighbor_mode == 'union':
            return self._get_union_neighbors(qid, docno, neighbors, weights)

        rescored = self._get_kg_enhanced_neighbors(qid, docno, neighbors, weights)
        selected = rescored[:self.kg_neighbor_k]
        membership = {
            n: {'in_laff': True, 'in_kg': True}
            for n, _, _ in selected
        }
        return selected, membership

    def _compute_kg_enhanced_cluster_lookup(self, qid: str, cluster_heads):
        combined_lookup = defaultdict(list)
        laff_lookup = defaultdict(list)
        kg_only_lookup = defaultdict(list)

        for cluster_head in cluster_heads:
            neighbors, weights = self.laff_graph.neighbours(cluster_head, weights=True)
            selected, _ = self._get_selected_neighbors(qid, cluster_head, neighbors, weights)

            for neighbor_docno, combined_score, comp in selected:
                combined_lookup[neighbor_docno].append(float(combined_score))
                laff_lookup[neighbor_docno].append(float(comp.laff_raw))
                kg_only_lookup[neighbor_docno].append(float(comp.kg_connectivity))

        return (
            {k: mean(v) for k, v in combined_lookup.items()},
            {k: mean(v) for k, v in laff_lookup.items()},
            {k: mean(v) for k, v in kg_only_lookup.items()},
        )

    def _build_locked_baseline_candidates(self, filtered_arms, laff_lookup, bm25_scores):
        neighbor_criteria_arms = [a for a in filtered_arms if a.docnos[-1] in laff_lookup]

        laff_scores = [(a, laff_lookup.get(a.docnos[-1], 0.0)) for a in neighbor_criteria_arms]
        laff_top = [a for a, _ in heapq.nlargest(35, laff_scores, key=lambda x: x[1])]
        laff_docnos = set(a.docnos[-1] for a in laff_top)

        remaining_arms = [
            (a, bm25_scores.get(a.docnos[-1], 0.0))
            for a in neighbor_criteria_arms
            if a.docnos[-1] in bm25_scores and a.docnos[-1] not in laff_docnos
        ]
        bm25_arms = [a for a, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]

        baseline_candidates = list(dict.fromkeys(laff_top + bm25_arms))
        if len(baseline_candidates) == 0:
            baseline_candidates = filtered_arms

        return baseline_candidates, laff_top

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
        lambda_kg = self.kg_cer_weight_init if self.use_kg_in_cer else 0.0

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
                docid: {
                    'initial_bm25': True,
                    'in_laff': False,
                    'in_kg': False,
                    'kg_ce_post': False,
                }
                for docid in initial_results['docno'].tolist()[:self.budget]
            }
            candidate_pool_docs = {
                docid: {
                    'initial_bm25': True,
                    'in_laff': False,
                    'in_kg': False,
                    'kg_ce_post': False,
                }
                for docid in initial_results['docno'].tolist()[:self.budget]
            }

            kg_candidates_seen = {}
            count = 0
            prev_heads = []

            while len(arms) > 0 and len(results) < self.budget:
                if count == 0:
                    arm = sorted(arms, key=lambda x: x.estimate_utility(), reverse=True)[:self.batch_size]
                    cluster_heads = []
                    combined_lookup = {}
                    laff_lookup = {}
                    kg_only_lookup = {}
                else:
                    cluster_heads = [doc for doc, _ in Counter(results).most_common(self.top_s)]
                    combined_lookup, laff_lookup, kg_only_lookup = self._compute_kg_enhanced_cluster_lookup(
                        qid, cluster_heads
                    )

                    filtered_arms = [a for a in arms if a.docnos[-1] not in results]

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

                    baseline_candidates, _ = self._build_locked_baseline_candidates(
                        filtered_arms, laff_lookup, bm25_scores
                    )

                    baseline_docnos = set(a.docnos[-1] for a in baseline_candidates)
                    kg_bonus_candidates = [
                        (a, combined_lookup.get(a.docnos[-1], 0.0))
                        for a in filtered_arms
                        if a.docnos[-1] in combined_lookup and a.docnos[-1] not in baseline_docnos
                    ]
                    kg_bonus_arms = [
                        a for a, _ in heapq.nlargest(self.kg_bonus_k, kg_bonus_candidates, key=lambda x: x[1])
                    ]

                    if prev_heads == cluster_heads:
                        baseline_batch = sorted(
                            baseline_candidates,
                            key=lambda x: (
                                x.cer_scores[x.docnos[-1]]
                                if x.docnos[-1] in x.cer_scores
                                else x.estimate_cer_score(
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
                        baseline_cer_scores = [
                            x.estimate_cer_score(
                                qid, query, x.docnos, results,
                                self.graph, self.laff_graph,
                                bm25_scores, cluster_heads,
                                lambda_bm25, lambda_aff, lambda_ce,
                                laff_lookup, kg_only_lookup,
                                use_kg_in_cer=self.use_kg_in_cer,
                                lambda_kg=lambda_kg,
                            )
                            for x in baseline_candidates
                        ]
                        baseline_batch = [
                            x for x, _ in sorted(
                                zip(baseline_candidates, baseline_cer_scores),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:self.batch_size]
                        ]

                    arm = baseline_batch

                    for a in kg_bonus_arms:
                        d = a.docnos[-1]
                        if d not in kg_candidates_seen:
                            kg_candidates_seen[d] = combined_lookup.get(d, 0.0)

                docnos_final = [x.docnos[-1] for x in arm]
                all_docnos = [x.docnos[-1] for x in arms]

                if count > 0:
                    estimated_rank_scores = [x.cer_scores[x.docnos[-1]] for x in arm if x.docnos[-1] in x.cer_scores]

                if len(results) < min(self.batch_size * self.cross_enc_budget, self.budget):
                    with torch.no_grad():
                        query_vecs = self.dual_encoder.encode_queries([query])[0].reshape(1, -1)

                    doc_object = [{'docno': docno} for docno in docnos_final]
                    doc_vecs = np.concatenate([
                        doc_vector.reshape(1, -1)
                        for doc_vector in self.corpus_index.vec_loader()(pd.DataFrame(doc_object))['doc_vec'].values
                    ])

                    dual_score = (query_vecs.dot(doc_vecs.T))[0]
                    batch = pd.DataFrame(docnos_final, columns=['docno'])
                    batch['qid'] = qid
                    batch['query'] = query
                    reranked_scores = list(self.scorer(batch)['score'].values)

                    ranked_set_scores = [x + score for x, score in zip(reranked_scores, dual_score)]

                    if count > 0:
                        bm25_features = np.array([
                            x.bm25_scores.get(x.docnos[-1], 0.0) for x in arm
                        ]).reshape(-1, 1)
                        affinity_features = np.array([
                            x.estimates[x.docnos[-1]] for x in arm
                        ]).reshape(-1, 1)
                        neighbor_score_features = np.array([
                            x.cross_enc_avg[x.docnos[-1]] for x in arm
                        ]).reshape(-1, 1)

                        if self.use_kg_in_cer:
                            kg_features = np.array([
                                x.kg_features.get(x.docnos[-1], 0.0) for x in arm
                            ]).reshape(-1, 1)

                            features = np.concatenate(
                                (bm25_features, affinity_features, neighbor_score_features, kg_features),
                                axis=1,
                            )

                            lb = self.param_bounds[0]
                            ub = self.param_bounds[1]
                            params = scipy.optimize.lsq_linear(
                                features,
                                ranked_set_scores,
                                lsq_solver='exact',
                                bounds=([lb, lb, lb, 0.0], [ub, ub, ub, 0.5]),
                            )
                            lambda_bm25, lambda_aff, lambda_ce, lambda_kg = params['x']
                        else:
                            features = np.concatenate(
                                (bm25_features, affinity_features, neighbor_score_features),
                                axis=1,
                            )
                            params = scipy.optimize.lsq_linear(
                                features,
                                ranked_set_scores,
                                lsq_solver='exact',
                                bounds=self.param_bounds,
                            )
                            lambda_bm25, lambda_aff, lambda_ce = params['x']
                else:
                    ranked_set_scores = estimated_rank_scores

                for x, docno, score_value in zip(arm, docnos_final, ranked_set_scores):
                    results[docno] = score_value
                    x.push(score_value)

                if len(results) < self.budget:
                    s_2 = Counter(results).most_common(self.top_s2)
                    s2 = [doc[0] for doc in s_2]
                    neighbor_lookup = set(s2).intersection(set(docnos_final))
                    parent_score_lookup = dict(zip(docnos_final, ranked_set_scores))

                    for docno in neighbor_lookup:
                        parent_score = parent_score_lookup.get(docno, results.get(docno, 0.0))
                        neighbors, weights = self.laff_graph.neighbours(docno, weights=True)
                        selected_neighbors, membership = self._get_selected_neighbors(qid, docno, neighbors, weights)

                        for neighbor, kg_score, comp in selected_neighbors:
                            if neighbor not in all_docnos:
                                neighbor_arm = ArmKG(neighbor, name=f'neighbors_{docno}')
                                neighbor_arm.push(parent_score)
                                neighbor_arm.kg_scores[neighbor] = float(kg_score)
                                neighbor_arm.kg_features[neighbor] = float(comp.kg_connectivity)
                                arms.append(neighbor_arm)
                                all_docnos.append(neighbor)

                                flags = membership.get(neighbor, {'in_laff': False, 'in_kg': False})

                                if neighbor not in candidate_pool_docs:
                                    candidate_pool_docs[neighbor] = {
                                        'initial_bm25': False,
                                        'in_laff': flags['in_laff'],
                                        'in_kg': flags['in_kg'],
                                        'kg_ce_post': False,
                                    }
                                else:
                                    candidate_pool_docs[neighbor]['in_laff'] = candidate_pool_docs[neighbor]['in_laff'] or flags['in_laff']
                                    candidate_pool_docs[neighbor]['in_kg'] = candidate_pool_docs[neighbor]['in_kg'] or flags['in_kg']

                                if neighbor not in doc_source:
                                    doc_source[neighbor] = {
                                        'initial_bm25': False,
                                        'in_laff': flags['in_laff'],
                                        'in_kg': flags['in_kg'],
                                        'kg_ce_post': False,
                                    }
                                else:
                                    doc_source[neighbor]['in_laff'] = doc_source[neighbor]['in_laff'] or flags['in_laff']
                                    doc_source[neighbor]['in_kg'] = doc_source[neighbor]['in_kg'] or flags['in_kg']

                                if flags['in_kg'] and not flags['in_laff']:
                                    if neighbor not in kg_candidates_seen:
                                        kg_candidates_seen[neighbor] = parent_score

                prev_heads = cluster_heads if count > 0 else []
                count += 1
                arms = [a for a in arms if not a.is_exhausted()]

            if self.enable_post_ce_bonus:
                kg_unseen = {d: s for d, s in kg_candidates_seen.items() if d not in results}
                if kg_unseen:
                    kg_top = sorted(kg_unseen.items(), key=lambda x: x[1], reverse=True)[:self.kg_bonus_k]
                    kg_docnos = [d for d, _ in kg_top]

                    with torch.no_grad():
                        query_vecs = self.dual_encoder.encode_queries([query])[0].reshape(1, -1)

                    doc_object = [{'docno': d} for d in kg_docnos]
                    doc_vecs = np.concatenate([
                        dv.reshape(1, -1)
                        for dv in self.corpus_index.vec_loader()(pd.DataFrame(doc_object))['doc_vec'].values
                    ])
                    dual_score = (query_vecs.dot(doc_vecs.T))[0]

                    batch_df = pd.DataFrame(kg_docnos, columns=['docno'])
                    batch_df['qid'] = qid
                    batch_df['query'] = query
                    kg_ce_scores = list(self.scorer(batch_df)['score'].values)
                    kg_final_scores = [ce + ds for ce, ds in zip(kg_ce_scores, dual_score)]

                    sorted_results = sorted(results.items(), key=lambda x: x[1])
                    kg_merged = 0

                    for kg_doc, kg_score in sorted(zip(kg_docnos, kg_final_scores), key=lambda x: x[1], reverse=True):
                        weakest_doc, weakest_score = sorted_results[0]
                        if kg_score > weakest_score:
                            del results[weakest_doc]
                            results[kg_doc] = kg_score

                            if kg_doc in doc_source:
                                doc_source[kg_doc]['kg_ce_post'] = True
                                doc_source[kg_doc]['in_kg'] = True
                            else:
                                doc_source[kg_doc] = {
                                    'initial_bm25': False,
                                    'in_laff': False,
                                    'in_kg': True,
                                    'kg_ce_post': True,
                                }

                            sorted_results.pop(0)
                            kg_merged += 1
                        else:
                            break

                    if self.verbose:
                        print(
                            f'[KG-POST-CE] qid={qid}: '
                            f'CE-scored {len(kg_docnos)} KG docs, merged {kg_merged}'
                        )


            if self.verbose:
                final_ranked_docs = [docno for docno, _ in Counter(results).most_common()]
                top50 = final_ranked_docs[:50]
                relevant_docs = self.qrels_map.get(str(qid), set())
                
                def count_category(flags_dict, mode):
                    docs = []
                    for d, flags in flags_dict.items():
                        if mode == "laff_only":
                            cond = flags.get("in_laff", False) and not flags.get("in_kg", False)
                        elif mode == "kg_only":
                            cond = flags.get("in_kg", False) and not flags.get("in_laff", False)
                        elif mode == "both":
                            cond = flags.get("in_laff", False) and flags.get("in_kg", False)
                        elif mode == "initial_bm25":
                            cond = flags.get("initial_bm25", False)
                        elif mode == "kg_ce_post":
                            cond = flags.get("kg_ce_post", False)
                        else:
                            cond = False

                        if cond:
                            docs.append(d)

                    rel_docs = [d for d in docs if d in relevant_docs]
                    return len(docs), len(rel_docs)

                def count_top50_category(mode):
                    docs = []
                    for d in top50:
                        flags = doc_source.get(d, {})
                        if mode == "laff_only":
                            cond = flags.get("in_laff", False) and not flags.get("in_kg", False)
                        elif mode == "kg_only":
                            cond = flags.get("in_kg", False) and not flags.get("in_laff", False)
                        elif mode == "both":
                            cond = flags.get("in_laff", False) and flags.get("in_kg", False)
                        elif mode == "initial_bm25":
                            cond = flags.get("initial_bm25", False)
                        elif mode == "kg_ce_post":
                            cond = flags.get("kg_ce_post", False)
                        else:
                            cond = False

                        if cond:
                            docs.append(d)

                    rel_docs = [d for d in docs if d in relevant_docs]
                    return len(docs), len(rel_docs)

                print("\n" + "=" * 74)
                print(f"[POOL/TOP50] qid={qid}  mode={self.neighbor_mode}")
                print(f'{"source":14s} {"pool":>6s} {"pool_rel":>9s} {"top50":>7s} {"top50_rel":>10s}')
                print("-" * 74)

                labels = [
                    "initial_bm25",
                    "laff_only",
                    "kg_only",
                    "both",
                    "kg_ce_post",
                ]

                for label in labels:
                    pool_n, pool_rel = count_category(candidate_pool_docs, label)
                    top_n, top_rel = count_top50_category(label)
                    print(f'{label:14s} {pool_n:6d} {pool_rel:9d} {top_n:7d} {top_rel:10d}')

                print("=" * 74)

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

        self.estimated_scores = (lambda_bm25 * bm25_score) + (lambda_aff * neigh_support)

        if len(crss_enc_scores) > 0:
            score_utility = sum(crss_enc_scores) / len(crss_enc_scores)
        else:
            score_utility = self.estimate_utility()
            if score_utility == float('-inf'):
                score_utility = 0.0

        self.cross_enc_avg[doc] = score_utility
        self.estimates[doc] = self.estimated_scores

        cer = (lambda_aff * self.estimated_scores) + (lambda_ce * score_utility)
        if use_kg_in_cer:
            cer += lambda_kg * kg_term

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
        debug=False,
        debug_print_limit=0,
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