from collections import Counter, defaultdict
from statistics import mean
import heapq
import random
import time
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
        laff_graph_full: CorpusGraph,
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
        laff_pre_k: int = 128,
        prf_min_support: int = 1,
        prf_kg_candidates_per_parent: int = 8,
        prf_kg_max_per_iter: int = 5,
        use_laff_norm_filter: bool = False,
        laff_norm_threshold: float = 0.6,
        qrels_map=None,
        debug_qid=None,
    ):
        self.scorer = scorer
        self.graph = graph
        self.laff_graph = laff_graph
        self.laff_graph_full = laff_graph_full
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
        self.laff_pre_k = laff_pre_k
        self.prf_min_support = prf_min_support
        self.prf_kg_candidates_per_parent = prf_kg_candidates_per_parent
        self.prf_kg_max_per_iter = prf_kg_max_per_iter
        self.use_laff_norm_filter = use_laff_norm_filter
        self.laff_norm_threshold = laff_norm_threshold
        self.qrels_map = qrels_map or {}
        self.debug_qid = str(debug_qid) if debug_qid is not None else None
        self.track_docno = None

    def estimate_bm25_score_batch(self, qids, queries, docids, initial_results):
        batch = []
        for qid, query, docid in zip(qids, queries, docids):
            batch.append([qid, query, docid, docstore.get(docid).text])
        df = pd.DataFrame(batch, columns=['qid', 'query', 'docno', 'text'])
        result_df = ret_scorer(df)
        return list(result_df['docno'].values), list(result_df['score'].values)

    def _get_qrel_label(self, qid, docno):
        qrels_for_q = self.qrels_map.get(str(qid), self.qrels_map.get(qid, {}))

        if not qrels_for_q:
            return 0

        if isinstance(qrels_for_q, dict):
            return qrels_for_q.get(str(docno), qrels_for_q.get(docno, 0))

        if isinstance(qrels_for_q, set):
            return 2 if (str(docno) in qrels_for_q or docno in qrels_for_q) else 0

        if isinstance(qrels_for_q, (list, tuple)):
            return 2 if (str(docno) in qrels_for_q or docno in qrels_for_q) else 0

        return 0

    def _get_union_expansion_neighbors(self, qid: str, query: str, docno: str, neighbors: np.ndarray, weights: np.ndarray):
        neighbor_docnos = [str(n) for n in neighbors]

        laff_top = neighbor_docnos[:self.kg_neighbor_k]
        laff_top_set = set(laff_top)

        pre_k = min(len(neighbor_docnos), self.laff_pre_k)
        pre_docnos = neighbor_docnos[:pre_k]
        pre_weights = weights[:pre_k]

        # KG rescore to get entity overlap + KG connectivity
        rescored = self.kg_scorer.rescore_neighbors(docno, pre_docnos, pre_weights, qid=str(qid))

        # Filter: NOT in LAFF top-k, must have at least some entity overlap
        candidates = [
            (n, score, comp)
            for n, score, comp in rescored
            if n not in laff_top_set
            and comp.entity_overlap >0.0
        ]

        if len(candidates) == 0:
            return laff_top, []

        # ── BM25 query relevance for each candidate ──
        cand_docnos = [n for n, _, _ in candidates]
        bm25_batch = pd.DataFrame([
            {'qid': qid, 'query': query, 'docno': n, 'text': docstore.get(n).text}
            for n in cand_docnos
        ])
        bm25_result = ret_scorer(bm25_batch)
        bm25_map = dict(zip(bm25_result['docno'], bm25_result['score']))

        # Drop candidates with zero BM25 query relevance
        candidates = [
            (n, score, comp)
            for n, score, comp in candidates
            if bm25_map.get(n, 0.0) > 0.0
        ]

        if len(candidates) == 0:
            return laff_top, []

        # ── Rank by each signal independently ──
        # Signal 1: BM25(neighbor, query) — query relevance
        by_bm25 = sorted(candidates, key=lambda x: bm25_map.get(x[0], 0.0), reverse=True)
        bm25_rank = {n: rank for rank, (n, _, _) in enumerate(by_bm25)}

        # Signal 2: Entity overlap (higher = better)
        by_entity = sorted(candidates, key=lambda x: x[2].entity_overlap, reverse=True)
        entity_rank = {n: rank for rank, (n, _, _) in enumerate(by_entity)}

        # Signal 3: KG connectivity (higher = better)
        by_kg = sorted(candidates, key=lambda x: x[2].kg_connectivity, reverse=True)
        kg_rank = {n: rank for rank, (n, _, _) in enumerate(by_kg)}

        # ── RRF fusion (k=60 is standard) ──
        rrf_k = 60
        rrf_scores = []
        for n, score, comp in candidates:
            rrf = (
                1.0 / (rrf_k + bm25_rank[n]) +
                1.0 / (rrf_k + entity_rank[n]) +
                1.0 / (rrf_k + kg_rank[n])
            )
            rrf_scores.append((n, rrf, comp, bm25_map.get(n, 0.0)))

        # Sort by RRF score descending, take top candidates
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        kg_candidates = rrf_scores[:self.prf_kg_candidates_per_parent]

        if self.verbose and len(kg_candidates) > 0:
            print(
                f"  [RRF] parent={docno}  candidates={len(candidates)}  selected={len(kg_candidates)}"
            )
            for n, rrf, comp, bm25_q in kg_candidates[:3]:
                print(
                    f"    {n}  rrf={rrf:.4f}  "
                    f"bm25q={bm25_q:.2f}(r{bm25_rank[n]})  "
                    f"ent={comp.entity_overlap:.2f}(r{entity_rank[n]})  "
                    f"kg={comp.kg_connectivity:.2f}(r{kg_rank[n]})"
                )

        # Return format: (docno, rrf_score, comp, laff_norm_placeholder)
        return laff_top, [(n, rrf, comp, 0.0) for n, rrf, comp, _ in kg_candidates]

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))

        lambda_param = 0.65
        lambda_param_1 = 0.45
        lambda_param_2 = 0.65

        for _, (query, initial_results) in enumerate(groups):
            qid = initial_results['qid'].iloc[0]

            initial_results = initial_results.sort_values('score', ascending=False)

            arms = [
                ArmBaselineUnion(docid, name='initial_results_' + docid)
                for docid in initial_results['docno'].tolist()[:self.budget]
            ]

            results = {}
            bm25_scores = dict(zip(initial_results['docno'].values, initial_results['score'].values))

            doc_source = {}
            for docid in initial_results['docno'].tolist()[:self.budget]:
                doc_source[str(docid)] = 'bm25'

            count = 0
            prev_heads = []
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

                        remaining_arms = [(arm, bm25_scores.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms if arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in new_arms]
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

                        remaining_arms = [(arm, bm25_scores.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms if arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in new_arms]
                        bm25_arms = [arm for arm, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]

                        new_arms.extend(bm25_arms)
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

                    kg_support = defaultdict(int)
                    kg_best_score = {}
                    kg_best_parent_score = {}
                    kg_parent_docs = defaultdict(list)
                    laff_to_add = []
                    kg_relevant_found = {}

                    parent_score_map = {docno: results.get(docno, 0.0) for docno in neighbor_lookup}

                    for docno in neighbor_lookup:
                        neighbors, weights = self.laff_graph_full.neighbours(docno, weights=True)
                        laff_neighbors, kg_candidates = self._get_union_expansion_neighbors(qid, query, docno, neighbors, weights)
                        parent_score = results.get(docno, 0.0)

                        for neighbor in laff_neighbors:
                            laff_to_add.append((neighbor, docno, parent_score))

                        for neighbor, kg_score, _comp, _laff_norm in kg_candidates:
                            label = self._get_qrel_label(qid, neighbor)
                            if label >= 2:
                                kg_relevant_found[neighbor] = label

                            kg_support[neighbor] += 1
                            kg_parent_docs[neighbor].append(docno)

                            current_score = float(kg_score)
                            if neighbor not in kg_best_score or current_score > kg_best_score[neighbor]:
                                kg_best_score[neighbor] = current_score
                                kg_best_parent_score[neighbor] = parent_score

                    for neighbor, parent_doc, parent_score in laff_to_add:
                        if neighbor not in all_docnos:
                            neighbor_arm = ArmBaselineUnion(neighbor, name=f'neighbors_{parent_doc}')
                            neighbor_arm.push(parent_score)
                            arms.append(neighbor_arm)
                            all_docnos.append(neighbor)
                            if str(neighbor) not in doc_source:
                                doc_source[str(neighbor)] = 'laff'

                    filtered_kg = [
                        n for n, support in kg_support.items()
                        if support >= self.prf_min_support
                    ]
                    filtered_kg = sorted(
                        filtered_kg,
                        key=lambda n: (kg_support[n], kg_best_score.get(n, 0.0)),
                        reverse=True,
                    )[:self.prf_kg_max_per_iter]

                    if self.verbose:
                        kg_found_rel = sum(1 for d in kg_relevant_found)
                        kg_new = [n for n in filtered_kg if n not in all_docnos]
                        kg_new_rel = sum(1 for n in kg_new if self._get_qrel_label(qid, n) >= 2)
                        kg_skip = [n for n in filtered_kg if n in all_docnos]
                        missed = [d for d in kg_relevant_found if d not in filtered_kg]
                        print(
                            f"[PRF] qid={qid} iter={count}  "
                            f"parents={len(neighbor_lookup)}  "
                            f"kg_cands={len(kg_best_score)}  "
                            f"passed={len(filtered_kg)}  "
                            f"new={len(kg_new)}(rel={kg_new_rel})  "
                            f"skip={len(kg_skip)}  "
                            f"missed_rel={len(missed)}"
                        )
                        if missed:
                            for d in missed:
                                print(f"  missed: {d} qrel={kg_relevant_found[d]} support={kg_support.get(d,0)} in_pool={d in all_docnos}({doc_source.get(str(d),'?')})")

                    for neighbor in filtered_kg:
                        if neighbor not in all_docnos:
                            parent_doc = kg_parent_docs[neighbor][0]
                            init_score = kg_best_parent_score.get(neighbor, 0.0)
                            neighbor_arm = ArmBaselineUnion(neighbor, name=f'kg_neighbors_{parent_doc}')
                            neighbor_arm.push(init_score)
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
                pool_by_source = defaultdict(int)
                for src in doc_source.values():
                    pool_by_source[src] += 1

                final_top50 = [doc for doc, _score in Counter(results).most_common(50)]
                final_rel = sum(1 for d in final_top50 if self._get_qrel_label(qid, d) >= 2)
                total_rel = len(self.qrels_map.get(str(qid), set()))
                kg_added = pool_by_source.get('kg', 0)
                kg_rel = sum(1 for d, src in doc_source.items() if src == 'kg' and self._get_qrel_label(qid, d) >= 2)
                recall = (final_rel / total_rel) if total_rel > 0 else 0.0

                print(
                    f"[QUERY] qid={qid}  pool: bm25={pool_by_source.get('bm25',0)} laff={pool_by_source.get('laff',0)} kg={kg_added}(rel={kg_rel})  "
                    f"top50_rel={final_rel}/{total_rel}  recall@50={recall:.4f}"
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
    laff_graph_full,
    kg_alpha: float = 0.0,
    kg_beta: float = 0.6,
    kg_gamma: float = 0.4,
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
        laff_graph_full=laff_graph_full,
        kg_scorer=kg_scorer,
        **kwargs,
    )