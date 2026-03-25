from collections import Counter, defaultdict
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

dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

_bm25_ref = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
).indexref
existing_index = pt.IndexFactory.of(_bm25_ref)
ret_scorer = pt.text.scorer(
    takes='docs',
    body_attr='text',
    wmodel='BM25',
    background_index=existing_index,
    controls={'termpipelines': 'Stopwords,PorterStemmer'}
)


class OREAdaptiveKGUnionOnBaseline(pt.Transformer):
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
        num_bm25_calls: int = 0,
        top_s: int = 25,
        top_s2: int = 15,
        cross_enc_budget: int = 2,
        param_bounds: tuple = (0.25, 0.9),
        kg_neighbor_k: int = 16,
        qrels_map=None,
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
        self.qrels_map = qrels_map or {}
        self._doc_text_cache = {}

    def _get_doc_text(self, docno: str) -> str:
        if docno not in self._doc_text_cache:
            try:
                self._doc_text_cache[docno] = docstore.get(docno).text
            except Exception:
                self._doc_text_cache[docno] = ''
        return self._doc_text_cache[docno]

    def estimate_bm25_score_batch(self, qids, queries, docids, initial_results):
        batch = []
        for qid, query, docid in zip(qids, queries, docids):
            batch.append([qid, query, docid, self._get_doc_text(docid)])
        df = pd.DataFrame(batch, columns=['qid', 'query', 'docno', 'text'])
        result_df = ret_scorer(df)
        return list(result_df['docno'].values), list(result_df['score'].values)

    _expansion_debug_count = 0

    def _get_union_expansion_neighbors(self, qid: str, docno: str, neighbors: np.ndarray, weights: np.ndarray):
        neighbor_docnos = [str(n) for n in neighbors]
        laff_top = neighbor_docnos[:self.kg_neighbor_k]
        laff_top_set = set(laff_top)
        rescored = self.kg_scorer.rescore_neighbors(docno, neighbor_docnos, weights, qid=str(qid))
        kg_top = [n for n, _score, _comp in rescored[:self.kg_neighbor_k]]
        kg_only_top = [n for n in kg_top if n not in laff_top_set]

        if self.verbose and OREAdaptiveKGUnionOnBaseline._expansion_debug_count < 3:
            OREAdaptiveKGUnionOnBaseline._expansion_debug_count += 1
            top5 = rescored[:5]
            print(f'  [UNION-DEBUG] expanding doc={docno}, total_neighbors={len(neighbor_docnos)}')
            for n, s, c in top5:
                in_laff = 'LAFF' if n in laff_top_set else 'KG-ONLY'
                print(f'    {n}: combined={s:.4f} eo={c.entity_overlap:.4f} kg_conn={c.kg_connectivity:.4f} laff_raw={c.laff_raw:.4f} [{in_laff}]')
            print(f'    → laff_top=16, kg_only={len(kg_only_top)}')

        return laff_top, kg_only_top

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))

        lambda_param = 0.65
        lambda_param_1 = 0.45
        lambda_param_2 = 0.65

        for _, (query, initial_results) in enumerate(groups):
            qid = initial_results['qid'].iloc[0]
            initial_results = initial_results.sort_values('score', ascending=False)

            arms = [ArmBaselineUnion(docid, name='initial_results_' + docid) for docid in initial_results['docno'].tolist()[:self.budget]]
            results = {}
            bm25_scores = dict(zip(initial_results['docno'].values, initial_results['score'].values))

            doc_source = {}
            for docid in initial_results['docno'].tolist()[:self.budget]:
                doc_source[str(docid)] = 'bm25'

            count = 0
            while len(arms) > 0 and len(results) < self.budget:
                if count == 0:
                    arm = sorted(arms, key=lambda x: x.estimate_utility(), reverse=True)[:self.batch_size]
                else:
                    cluster_heads = [doc for doc, _ in Counter(results).most_common(self.top_s)]

                    cluster_neigh_lookup = defaultdict(list)
                    for cluster_head in cluster_heads:
                        neighbors, scores = self.laff_graph.neighbours(cluster_head, weights=True)
                        for neighbor, score in zip(neighbors, scores):
                            cluster_neigh_lookup[neighbor].append(score)
                    cluster_neigh_lookup = {key: mean(scores) for key, scores in cluster_neigh_lookup.items()}

                    filtered_arms = [arm for arm in arms if arm.docnos[-1] not in results]

                    if len(bm25_scores) < (len(initial_results) + self.num_bm25_calls):
                        donos_missing = [x.docnos[-1] for x in filtered_arms if x.docnos[-1] not in bm25_scores]
                        if len(donos_missing) > 0:
                            qids = len(donos_missing) * [qid]
                            queries = len(donos_missing) * [query]
                            docnos, scores = self.estimate_bm25_score_batch(qids, queries, donos_missing, initial_results)
                            bm25_scores = dict(zip(docnos, scores))

                    if prev_heads == cluster_heads:
                        neighbor_criteria_arms = [arm for arm in filtered_arms if arm.docnos[-1] in cluster_neigh_lookup]
                        criteria_scores = [(arm, cluster_neigh_lookup.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms]
                        new_arms = [arm for arm, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]
                        remaining_arms = [
                            (arm, bm25_scores.get(arm.docnos[-1], 0))
                            for arm in neighbor_criteria_arms
                            if arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in new_arms
                        ]
                        bm25_arms = [arm for arm, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]
                        new_arms.extend(bm25_arms)
                        if len(new_arms) == 0:
                            new_arms = filtered_arms

                        arm = sorted(
                            new_arms,
                            key=lambda x: x.cer_scores[x.docnos[-1]]
                            if x.docnos[-1] in x.cer_scores
                            else x.estimate_cer_score(
                                qid,
                                query,
                                x.docnos,
                                results,
                                self.graph,
                                self.laff_graph,
                                initial_results,
                                bm25_scores,
                                cluster_heads,
                                lambda_param,
                                lambda_param_1,
                                lambda_param_2,
                                cluster_neigh_lookup,
                            ),
                            reverse=True,
                        )[:self.batch_size]
                    else:
                        neighbor_criteria_arms = [arm for arm in filtered_arms if arm.docnos[-1] in cluster_neigh_lookup]
                        criteria_scores = [(arm, cluster_neigh_lookup.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms]
                        new_arms = [arm for arm, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]

                        selected_docnos = {a.docnos[-1] for a in new_arms}

                        remaining_arms = [
                            (arm, bm25_scores.get(arm.docnos[-1], 0))
                            for arm in neighbor_criteria_arms
                            if arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in selected_docnos
                        ]
                        bm25_arms = [arm for arm, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]
                        new_arms.extend(bm25_arms)

                        selected_docnos = {a.docnos[-1] for a in new_arms}

                        kg_gate_arms = [
                            arm for arm in neighbor_criteria_arms
                            if doc_source.get(str(arm.docnos[-1])) == 'kg'
                            and arm.docnos[-1] not in selected_docnos
                        ]

                        kg_gate_scored = [
                            (arm, cluster_neigh_lookup.get(arm.docnos[-1], 0), bm25_scores.get(arm.docnos[-1], 0))
                            for arm in kg_gate_arms
                        ]

                        kg_arms = [
                            arm for arm, _, _ in sorted(
                                kg_gate_scored,
                                key=lambda x: (x[1], x[2]),
                                reverse=True
                            )[:5]
                        ]
                        new_arms.extend(kg_arms)

                        new_arms = set(new_arms)
                        if len(new_arms) == 0:
                            new_arms = filtered_arms

                        cer_scores = [
                            x.estimate_cer_score(
                                qid,
                                query,
                                x.docnos,
                                results,
                                self.graph,
                                self.laff_graph,
                                initial_results,
                                bm25_scores,
                                cluster_heads,
                                lambda_param,
                                lambda_param_1,
                                lambda_param_2,
                                cluster_neigh_lookup,
                            )
                            for x in new_arms
                        ]
                        arm = sorted(zip(new_arms, cer_scores), key=lambda x: x[1], reverse=True)[:self.batch_size]
                        arm = [x for x, _ in arm]

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
                            x.bm25_scores[x.docnos[-1]] if x.docnos[-1] in x.bm25_scores else 0.0
                            for x in arm
                        ]).reshape(-1, 1)
                        affinity_features = np.array([x.estimates[x.docnos[-1]] for x in arm]).reshape(-1, 1)
                        neighbor_score_features = np.array([x.cross_enc_avg[x.docnos[-1]] for x in arm]).reshape(-1, 1)
                        features = np.concatenate((bm25_features, affinity_features, neighbor_score_features), axis=1)
                        params = scipy.optimize.lsq_linear(
                            features,
                            ranked_set_scores,
                            lsq_solver='exact',
                            bounds=(self.param_bounds),
                        )
                        lambda_param = params['x'][0]
                        lambda_param_1 = params['x'][1]
                        lambda_param_2 = params['x'][2]
                else:
                    ranked_set_scores = estimated_rank_scores

                for x, docno, score_value in zip(arm, docnos_final, ranked_set_scores):
                    results[docno] = score_value
                    x.push(score_value)

                if len(results) < self.budget:
                    s_2 = Counter(results).most_common(self.top_s2)
                    s2 = [doc[0] for doc in s_2]
                    neighbor_lookup = set(s2).intersection(set(docnos_final))

                    for docno in neighbor_lookup:
                        neighbors, weights = self.laff_graph.neighbours(docno, weights=True)
                        laff_neighbors, kg_only_neighbors = self._get_union_expansion_neighbors(qid, docno, neighbors, weights)

                        for neighbor in laff_neighbors:
                            if neighbor not in all_docnos:
                                neighbor_arm = ArmBaselineUnion(neighbor, name=f'neighbors_{docno}')
                                neighbor_arm.push(score_value)
                                arms.append(neighbor_arm)
                                all_docnos.append(neighbor)
                                if str(neighbor) not in doc_source:
                                    doc_source[str(neighbor)] = 'laff'

                        for neighbor in kg_only_neighbors:
                            if neighbor not in all_docnos:
                                neighbor_arm = ArmBaselineUnion(neighbor, name=f'kg_neighbors_{docno}')
                                neighbor_arm.push(score_value)
                                arms.append(neighbor_arm)
                                all_docnos.append(neighbor)
                                if str(neighbor) not in doc_source:
                                    doc_source[str(neighbor)] = 'kg'

                if count > 0:
                    prev_heads = cluster_heads
                else:
                    prev_heads = []
                count += 1
                arms = [a for a in arms if not a.is_exhausted()]

            if self.verbose:
                final_ranked_docs = [docno for docno, _ in Counter(results).most_common()]
                top50 = final_ranked_docs[:50]
                relevant_docs = self.qrels_map.get(str(qid), set())

                pool_bm25 = sum(1 for d, s in doc_source.items() if s == 'bm25')
                pool_laff = sum(1 for d, s in doc_source.items() if s == 'laff')
                pool_kg = sum(1 for d, s in doc_source.items() if s == 'kg')
                pool_kg_rel = sum(1 for d, s in doc_source.items() if s == 'kg' and d in relevant_docs)

                top50_bm25 = sum(1 for d in top50 if doc_source.get(str(d)) == 'bm25')
                top50_laff = sum(1 for d in top50 if doc_source.get(str(d)) == 'laff')
                top50_kg = sum(1 for d in top50 if doc_source.get(str(d)) == 'kg')

                top50_bm25_rel = sum(1 for d in top50 if doc_source.get(str(d)) == 'bm25' and d in relevant_docs)
                top50_laff_rel = sum(1 for d in top50 if doc_source.get(str(d)) == 'laff' and d in relevant_docs)
                top50_kg_rel = sum(1 for d in top50 if doc_source.get(str(d)) == 'kg' and d in relevant_docs)
                top50_rel = top50_bm25_rel + top50_laff_rel + top50_kg_rel

                print(
                    f'[KG-ORE] qid={qid}  '
                    f'pool: bm25={pool_bm25} laff={pool_laff} kg={pool_kg}({pool_kg_rel}rel)  |  '
                    f'top50: bm25={top50_bm25}({top50_bm25_rel}rel) laff={top50_laff}({top50_laff_rel}rel) kg={top50_kg}({top50_kg_rel}rel)  |  '
                    f'total_rel={top50_rel}'
                )

            for rank, (docno, final_score) in enumerate(Counter(results).most_common()):
                result_builder.extend({
                    'qid': qid,
                    'query': query,
                    'docno': docno,
                    'score': final_score,
                    'rank': rank,
                })

        return result_builder.to_df()


class ArmBaselineUnion:
    def __init__(self, docnos: str, name: str = ''):
        self.docnos = [docnos]
        self.scores = []
        self.estimated_scores = []
        self.name = name
        self.cer_scores = {}
        self.bm25_scores = {}
        self.laff_scores = {}
        self.estimates = {}
        self.cross_enc_avg = {}

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

    def estimate_cer_score(self, qid, query, docnos, results, neigh_graph, graph, initial_results, bm25_score_dict, cluster_heads, lambda_param, lambda_1, lambda_2, cluster_neigh_lookup):
        doc = docnos[-1]
        if doc in bm25_score_dict:
            self.bm25_scores[doc] = bm25_score_dict[doc]
            bm25_score = bm25_score_dict[doc]
        else:
            bm25_score = 0

        laff_score_dict = dict(zip(*graph.neighbours(doc, weights=True)))
        crss_enc_scores = []
        valid_cluster_heads = [res for res in cluster_heads if res in laff_score_dict]
        crss_enc_scores.extend(results[res] for res in valid_cluster_heads)

        self.estimated_scores = lambda_param * bm25_score + lambda_1 * (cluster_neigh_lookup.get(doc, 0))

        if len(crss_enc_scores) > 0:
            score_utility = sum(crss_enc_scores) / len(crss_enc_scores)
        else:
            score_utility = self.estimate_utility()
            if score_utility == float('-inf'):
                score_utility = 0
        self.cross_enc_avg[docnos[-1]] = score_utility
        self.estimates[docnos[-1]] = self.estimated_scores
        self.cer_scores[docnos[-1]] = (lambda_1 * self.estimated_scores) + (lambda_2 * score_utility)
        return (lambda_1 * self.estimated_scores) + (lambda_2 * score_utility)


def create_ore_kg_union_on_baseline(
    dual_encoder,
    scorer,
    corpus_index,
    graph,
    laff_graph,
    kg_alpha: float = 0.0,
    kg_beta: float = 0.3,
    kg_gamma: float = 0.7,
    kg_score_mode: str = 'minmax',
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

    return OREAdaptiveKGUnionOnBaseline(
        dual_encoder=dual_encoder,
        scorer=scorer,
        corpus_index=corpus_index,
        graph=graph,
        laff_graph=laff_graph,
        kg_scorer=kg_scorer,
        **kwargs,
    )