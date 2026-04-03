
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from statistics import mean
import heapq

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import scipy
import torch
import ir_datasets
from pyterrier_adaptive import CorpusGraph

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Deep per-query KG-ORE diagnostic')
parser.add_argument('--qid', type=str, required=True)
parser.add_argument('--dl', type=int, default=20)
parser.add_argument('--budget', type=int, default=50)
parser.add_argument('--ce', type=int, default=4)
parser.add_argument('--s1', type=int, default=10)
parser.add_argument('--s2', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--kg_neighbor_k', type=int, default=16)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--kg_mode', type=str, default='minmax')
parser.add_argument('--passage_el', type=str, default=None)
parser.add_argument('--passage_el_db', type=str, default=None)
parser.add_argument('--freebase_dir', type=str, default=None)
parser.add_argument('--query_el', type=str, default=None)
args = parser.parse_args()

PARAM_BOUNDS = (0.25, 0.95)
PRF_KG_PER_PARENT = 8
PRF_KG_MAX_PER_ITER = 10
PRF_MIN_SUPPORT = 1

# ── Init ──────────────────────────────────────────────────────────────────────
if not pt.java.started():
    pt.java.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

dataset = pt.get_dataset('irds:msmarco-passage')
dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

from pyterrier_dr import FlexIndex, TasB
from pyterrier_t5 import MonoT5ReRanker
from kg_scorer_unified_corrected import KGScorerUnified, create_scorer

model = TasB.dot(batch_size=1, device=device)
idx_art = pta.Artifact.from_hf('macavaney/msmarco-passage.tasb.flex')
idx = FlexIndex(idx_art.path)

_bm25_ref = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25'
).indexref
existing_index = pt.IndexFactory.of(_bm25_ref)
ret_scorer = pt.text.scorer(
    takes='docs', body_attr='text', wmodel='BM25',
    background_index=existing_index,
    controls={'termpipelines': 'Stopwords,PorterStemmer'}
)

bm25 = pt.terrier.Retriever.from_dataset(
    'msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=args.budget
)
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch_size)

graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
laff_graph_full = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff')
laff_graph = laff_graph_full.to_limit_k(16)

kg_scorer = create_scorer(
    alpha=args.alpha, beta=args.beta, gamma=args.gamma,
    kg_score_mode=args.kg_mode,
    freebase_dir=args.freebase_dir,
    passage_el_path=args.passage_el,
    query_el_path=args.query_el,
    passage_el_db=args.passage_el_db,
    debug=False, debug_print_limit=0,
)

# ── Load query ────────────────────────────────────────────────────────────────
eval_dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')
topics_df = eval_dataset.get_topics()
qrels_df = eval_dataset.get_qrels()

topics_1 = topics_df[topics_df['qid'] == args.qid].copy()
qrels_1 = qrels_df[qrels_df['qid'] == args.qid].copy()

if len(topics_1) == 0:
    print(f'ERROR: qid={args.qid} not found. Available: {sorted(topics_df["qid"].tolist())}')
    sys.exit(1)

query_text = topics_1['query'].iloc[0]
# Remove colons for BM25
query_clean = query_text.replace(":", " ")

rel_docs = {d: l for d, l in zip(qrels_1['docno'].values, qrels_1['label'].values) if l >= 2}
all_judged = dict(zip(qrels_1['docno'].values, qrels_1['label'].values))


def is_rel(docno):
    return str(docno) in rel_docs


def rel_label(docno):
    return rel_docs.get(str(docno), all_judged.get(str(docno), -1))


def rel_tag(docno):
    l = rel_label(docno)
    if l >= 2:
        return f' ★REL(l={l})'
    elif l >= 1:
        return f' ·marginal(l={l})'
    return ''


print(f'\n{"="*80}')
print(f'DEEP QUERY DIAGNOSTIC: qid={args.qid}')
print(f'Query: "{query_text}"')
print(f'Settings: budget={args.budget} ce={args.ce} s1={args.s1} s2={args.s2} batch={args.batch_size}')
print(f'MonoT5 gate: results < min({args.batch_size}*{args.ce}, {args.budget}) = {min(args.batch_size*args.ce, args.budget)}')
print(f'Relevant docs (label>=2): {len(rel_docs)}')
print(f'{"="*80}')

# ── BM25 retrieval ────────────────────────────────────────────────────────────
topics_clean = topics_1.copy()
topics_clean['query'] = topics_clean['query'].str.replace(':', ' ', regex=False)
bm25_results = bm25(topics_clean)
bm25_results = bm25_results.sort_values('score', ascending=False)

bm25_pool = bm25_results['docno'].tolist()[:args.budget]
bm25_score_map = dict(zip(bm25_results['docno'].values, bm25_results['score'].values))

bm25_rel = [d for d in bm25_pool if is_rel(d)]
print(f'\n[BM25 POOL] {len(bm25_pool)} docs, {len(bm25_rel)} relevant')
for d in bm25_rel:
    r = bm25_pool.index(d)
    print(f'  bm25_rank={r:3d}  {d}  bm25={bm25_score_map[d]:.2f}  label={rel_docs[d]}')

missing_rel = {d: l for d, l in rel_docs.items() if d not in bm25_pool}
print(f'  Missing relevant: {len(missing_rel)}')

# ── Reachability quick check ──────────────────────────────────────────────────
reachable_from = defaultdict(list)
for bm25_doc in bm25_pool:
    neighbors, weights = laff_graph_full.neighbours(bm25_doc, weights=True)
    for pos, (n, w) in enumerate(zip([str(x) for x in neighbors], weights)):
        if n in missing_rel:
            zone = 'LAFF16' if pos < args.kg_neighbor_k else 'KG'
            reachable_from[n].append((bm25_doc, pos, float(w), zone))

reachable = set(reachable_from.keys())
unreachable = set(missing_rel.keys()) - reachable
laff16_reachable = sum(1 for d, ps in reachable_from.items() if any(z == 'LAFF16' for _, _, _, z in ps))
kg_only_reachable = sum(1 for d, ps in reachable_from.items() if all(z == 'KG' for _, _, _, z in ps))

print(f'\n[REACHABILITY] {len(reachable)}/{len(missing_rel)} missing rel docs reachable')
print(f'  via LAFF16: {laff16_reachable},  KG-only(16-127): {kg_only_reachable},  unreachable: {len(unreachable)}')
if unreachable:
    for d in sorted(unreachable):
        print(f'  UNREACHABLE: {d} label={rel_docs[d]}')


# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENTED PIPELINE LOOP
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_doc_as_query(text, max_terms=40):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[#:'\"()\[\]{}+\-!?\^~*/\|&=<>,.;%$@`]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    terms = [t for t in text.split() if len(t) > 2]
    return " ".join(terms[:max_terms])


class Arm:
    def __init__(self, docno, name=''):
        self.docnos = [docno]
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

    def push(self, score):
        self.scores.append(score)

    def estimate_utility(self):
        if len(self.scores) == 0:
            return float('-inf')
        return sum(self.scores) / len(self.scores)

    def estimate_cer_score(self, doc, feature_maps, lp, l1, l2):
        bm25_raw = feature_maps['bm25_raw'].get(doc, 0.0)
        neigh_raw = feature_maps['neigh_raw'].get(doc, 0.0)
        utility_raw = feature_maps['utility_raw'].get(doc, 0.0)
        bm25_norm = feature_maps['bm25_norm'].get(doc, 0.0)
        neigh_norm = feature_maps['neigh_norm'].get(doc, 0.0)
        utility_mix = feature_maps['util_mix'].get(doc, 0.0)

        self.bm25_scores[doc] = bm25_raw
        self.cross_enc_avg[doc] = utility_raw
        self.estimates[doc] = {
            'bm25_norm': bm25_norm,
            'neigh_norm': neigh_norm,
            'utility_mix': utility_mix,
        }

        cer_score = (lp * bm25_norm) + (l1 * neigh_norm) + (l2 * utility_mix)
        self.cer_scores[doc] = cer_score
        return cer_score


def get_kg_expansion(qid, query, parent_docno, neighbors, weights):
    """Same logic as _get_union_expansion_neighbors."""
    neighbor_docnos = [str(n) for n in neighbors]
    laff_top = neighbor_docnos[:args.kg_neighbor_k]
    laff_top_set = set(laff_top)

    pre_k = min(len(neighbor_docnos), 128)
    pre_docnos = neighbor_docnos[:pre_k]
    pre_weights = weights[:pre_k]

    rescored = kg_scorer.rescore_neighbors(parent_docno, pre_docnos, pre_weights, qid=str(qid))

    candidates = [
        (n, score, comp) for n, score, comp in rescored
        if n not in laff_top_set and comp.entity_overlap > 0.0
    ]
    if not candidates:
        return laff_top, [], {'total_kg_range': len(neighbor_docnos) - len(laff_top),
                              'after_entity': 0, 'after_bm25': 0}

    cand_docnos = [n for n, _, _ in candidates]
    bm25_batch = pd.DataFrame([
        {'qid': qid, 'query': query, 'docno': n, 'text': docstore.get(n).text}
        for n in cand_docnos
    ])
    bm25_result = ret_scorer(bm25_batch)
    bm25_map = dict(zip(bm25_result['docno'], bm25_result['score']))

    after_ent = len(candidates)
    candidates = [(n, s, c) for n, s, c in candidates if bm25_map.get(n, 0.0) > 0.0]
    after_bm25 = len(candidates)

    if not candidates:
        return laff_top, [], {'total_kg_range': len(neighbor_docnos) - len(laff_top),
                              'after_entity': after_ent, 'after_bm25': 0}

    by_bm25 = sorted(candidates, key=lambda x: bm25_map.get(x[0], 0.0), reverse=True)
    bm25_rank = {n: r for r, (n, _, _) in enumerate(by_bm25)}
    by_entity = sorted(candidates, key=lambda x: x[2].entity_overlap, reverse=True)
    entity_rank = {n: r for r, (n, _, _) in enumerate(by_entity)}
    by_kg = sorted(candidates, key=lambda x: x[2].kg_connectivity, reverse=True)
    kg_rank_map = {n: r for r, (n, _, _) in enumerate(by_kg)}

    rrf_k = 60
    rrf_scores = []
    for n, score, comp in candidates:
        rrf = (1.0 / (rrf_k + bm25_rank[n]) +
               1.0 / (rrf_k + entity_rank[n]) +
               1.0 / (rrf_k + kg_rank_map[n]))
        rrf_scores.append((n, rrf, comp, bm25_map.get(n, 0.0)))
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    selected = rrf_scores[:PRF_KG_PER_PARENT]

    stats = {
        'total_kg_range': len(neighbor_docnos) - len(laff_top),
        'after_entity': after_ent,
        'after_bm25': after_bm25,
    }
    return laff_top, [(n, rrf, comp, 0.0) for n, rrf, comp, _ in selected], stats


def estimate_bm25_batch(qid, query, docids):
    batch = [{'qid': qid, 'query': query, 'docno': d, 'text': docstore.get(d).text} for d in docids]
    df = pd.DataFrame(batch)
    result_df = ret_scorer(df)
    return dict(zip(result_df['docno'].values, result_df['score'].values))


def minmax_normalize(score_map):
    if not score_map:
        return {}
    vals = np.array(list(score_map.values()), dtype=float)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo < 1e-12:
        return {k: (1.0 if v > 0 else 0.0) for k, v in score_map.items()}
    return {k: float((v - lo) / (hi - lo)) for k, v in score_map.items()}


def compute_candidate_feature_maps(candidate_pool, cluster_heads, cluster_neigh_lookup, results, bm25_scores):
    bm25_raw = {}
    neigh_raw = {}
    utility_raw = {}
    for arm in candidate_pool:
        doc = arm.docnos[-1]
        bm25_raw[doc] = float(bm25_scores.get(doc, 0.0))
        neigh_raw[doc] = float(cluster_neigh_lookup.get(doc, 0.0))

        laff_neigh = dict(zip(*laff_graph.neighbours(doc, weights=True)))
        valid_cluster_heads = [res for res in cluster_heads if res in laff_neigh and res in results]
        if valid_cluster_heads:
            utility = float(mean(results[res] for res in valid_cluster_heads))
        else:
            utility = arm.estimate_utility()
            if utility == float('-inf'):
                utility = 0.0
        utility_raw[doc] = max(float(utility), 0.0)

    bm25_norm = minmax_normalize(bm25_raw)
    neigh_norm = minmax_normalize(neigh_raw)
    util_norm = minmax_normalize(utility_raw)

    frontier_ratio = len([a for a in candidate_pool if a.docnos[-1] in cluster_neigh_lookup]) / max(len(candidate_pool), 1)
    bm25_nonzero_ratio = sum(1 for d in bm25_raw if bm25_raw[d] > 0.0) / max(len(candidate_pool), 1)
    exploit_strength = np.clip(0.55 * frontier_ratio + 0.45 * bm25_nonzero_ratio, 0.20, 0.85)

    util_soft = {d: min(v, 0.75) for d, v in util_norm.items()}
    util_mix = {
        d: float(exploit_strength * util_norm[d] + (1.0 - exploit_strength) * util_soft[d])
        for d in util_norm
    }

    return {
        'bm25_raw': bm25_raw,
        'neigh_raw': neigh_raw,
        'utility_raw': utility_raw,
        'bm25_norm': bm25_norm,
        'neigh_norm': neigh_norm,
        'util_norm': util_norm,
        'util_soft': util_soft,
        'util_mix': util_mix,
        'exploit_strength': float(exploit_strength),
        'frontier_ratio': float(frontier_ratio),
        'bm25_nonzero_ratio': float(bm25_nonzero_ratio),
    }


# ── RUN ───────────────────────────────────────────────────────────────────────
qid = args.qid
query = query_clean

arms = [Arm(str(d), name=f'bm25_{d}') for d in bm25_pool]
for a in arms:
    a.push(bm25_score_map.get(a.docnos[-1], 0.0))

results = {}
bm25_scores = dict(bm25_score_map)
doc_source = {str(d): 'bm25' for d in bm25_pool}

lambda_param, lambda_param_1, lambda_param_2 = 0.65, 0.45, 0.65
count = 0
prev_heads = []
monot5_gate = min(args.batch_size * args.ce, args.budget)

print(f'\n{"═"*80}')
print(f'PIPELINE EXECUTION (MonoT5 gate at results < {monot5_gate})')
print(f'{"═"*80}')

while len(arms) > 0 and len(results) < args.budget:
    print(f'\n{"─"*80}')
    print(f'ITERATION {count}  |  results={len(results)}/{args.budget}  |  arms={len(arms)}')
    print(f'{"─"*80}')

    # Count relevant docs in current arms
    rel_in_arms = [(a.docnos[-1], a.name) for a in arms if is_rel(a.docnos[-1]) and a.docnos[-1] not in results]
    rel_scored = [d for d in results if is_rel(d)]
    print(f'  Relevant docs: {len(rel_scored)} scored, {len(rel_in_arms)} waiting in arms, '
          f'{len(rel_docs) - len(rel_scored) - len(rel_in_arms)} not in pool')
    if rel_in_arms:
        for d, name in rel_in_arms:
            print(f'    WAITING: {d} label={rel_docs[d]} source={name}')

    # ── Batch selection ───────────────────────────────────────────────────
    if count == 0:
        arm = sorted(arms, key=lambda x: x.estimate_utility(), reverse=True)[:args.batch_size]
        print(f'  Selection: utility sort (iter 0) -> top {args.batch_size}')
    else:
        cluster_heads = [doc for doc, _ in Counter(results).most_common(args.s1)]
        heads_changed = (prev_heads != cluster_heads)
        print(f'  Cluster heads (top-{args.s1}): {cluster_heads[:5]}...')
        print(f'  Heads changed: {heads_changed}')

        cluster_neigh_lookup = defaultdict(list)
        for ch in cluster_heads:
            neighbors, scores = laff_graph.neighbours(ch, weights=True)
            for neighbor, score in zip(neighbors, scores):
                cluster_neigh_lookup[neighbor].append(score)
        cluster_neigh_lookup = {k: mean(v) for k, v in cluster_neigh_lookup.items()}

        filtered_arms = [a for a in arms if a.docnos[-1] not in results]
        print(f'  Filtered arms (not yet scored): {len(filtered_arms)}')

        # Check if relevant docs are in LAFF neighborhood of cluster heads
        rel_in_filtered = [a for a in filtered_arms if is_rel(a.docnos[-1])]
        for a in rel_in_filtered:
            d = a.docnos[-1]
            in_neigh = d in cluster_neigh_lookup
            neigh_score = cluster_neigh_lookup.get(d, 0)
            bm25_s = bm25_scores.get(d, 0)
            print(f'    REL in filtered: {d} label={rel_docs[d]} in_cluster_neigh={in_neigh} '
                  f'neigh_score={neigh_score:.4f} bm25={bm25_s:.2f} source={a.name}')

        # CER selection logic
        neighbor_criteria_arms = [a for a in filtered_arms if a.docnos[-1] in cluster_neigh_lookup]
        print(f'  Neighbor criteria arms: {len(neighbor_criteria_arms)}/{len(filtered_arms)}')

        criteria_scores = [(a, cluster_neigh_lookup.get(a.docnos[-1], 0)) for a in neighbor_criteria_arms]
        new_arms_laff = [a for a, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]

        selected_docnos = {a.docnos[-1] for a in new_arms_laff}
        remaining = [
            (a, bm25_scores.get(a.docnos[-1], 0.0))
            for a in neighbor_criteria_arms
            if a.docnos[-1] in bm25_scores and a.docnos[-1] not in selected_docnos
        ]
        bm25_arms_extra = [a for a, _ in heapq.nlargest(25, remaining, key=lambda x: x[1])]
        candidate_pool = new_arms_laff + bm25_arms_extra
        candidate_pool = list({a.docnos[-1]: a for a in candidate_pool}.values())
        if not candidate_pool:
            candidate_pool = filtered_arms

        feature_maps = compute_candidate_feature_maps(
            candidate_pool, cluster_heads, cluster_neigh_lookup, results, bm25_scores
        )
        print(
            f"  CER mode: normalized features | frontier_ratio={feature_maps['frontier_ratio']:.3f} "
            f"bm25_ratio={feature_maps['bm25_nonzero_ratio']:.3f} "
            f"exploit_strength={feature_maps['exploit_strength']:.3f}"
        )

        for cand in candidate_pool:
            d = cand.docnos[-1]
            cand.estimate_cer_score(d, feature_maps, lambda_param, lambda_param_1, lambda_param_2)

        arm = sorted(
            candidate_pool,
            key=lambda x: x.cer_scores.get(x.docnos[-1], float('-inf')),
            reverse=True,
        )[:args.batch_size]

        print(f'  CER candidates: {len(candidate_pool)} -> selected {len(arm)} for MonoT5')

        # Show CER scores for ALL relevant docs in candidate pool
        rel_in_candidates = [a for a in candidate_pool if is_rel(a.docnos[-1])]
        selected_set = {a.docnos[-1] for a in arm}
        for a in rel_in_candidates:
            d = a.docnos[-1]
            cer = a.cer_scores.get(d, '?')
            picked = '→ SELECTED' if d in selected_set else '→ SKIPPED'
            print(f'    REL in CER pool: {d} cer={cer} {picked}')

    # ── What gets scored ──────────────────────────────────────────────────
    docnos_final = [x.docnos[-1] for x in arm]
    all_docnos = [x.docnos[-1] for x in arms]

    if count > 0:
        estimated_rank_scores = [x.cer_scores[x.docnos[-1]] for x in arm if x.docnos[-1] in x.cer_scores]

    uses_monot5 = len(results) < monot5_gate
    print(f'\n  MonoT5 scoring: {"YES" if uses_monot5 else "NO (using CER estimates)"}')
    print(f'  Batch ({len(docnos_final)} docs):')

    if uses_monot5:
        with torch.no_grad():
            query_vecs = model.encode_queries([query])[0].reshape(1, -1)
        doc_object = [{'docno': d} for d in docnos_final]
        doc_vecs = np.concatenate([
            dv.reshape(1, -1)
            for dv in idx.vec_loader()(pd.DataFrame(doc_object))['doc_vec'].values
        ])
        dual_score = (query_vecs.dot(doc_vecs.T))[0]

        batch_df = pd.DataFrame(docnos_final, columns=['docno'])
        batch_df['qid'] = qid
        batch_df['query'] = query
        reranked_scores = list(scorer(batch_df)['score'].values)
        ranked_set_scores = [x + s for x, s in zip(reranked_scores, dual_score)]

        if count > 0:
            bm25_features = np.array([
                x.estimates[x.docnos[-1]]['bm25_norm'] for x in arm
            ]).reshape(-1, 1)
            neigh_features = np.array([
                x.estimates[x.docnos[-1]]['neigh_norm'] for x in arm
            ]).reshape(-1, 1)
            utility_features = np.array([
                x.estimates[x.docnos[-1]]['utility_mix'] for x in arm
            ]).reshape(-1, 1)
            features = np.concatenate((bm25_features, neigh_features, utility_features), axis=1)
            params = scipy.optimize.lsq_linear(features, ranked_set_scores, lsq_solver='exact', bounds=PARAM_BOUNDS)
            lambda_param, lambda_param_1, lambda_param_2 = params['x']
            #print(f'  Updated lambdas: λ_bm25={lambda_param:.3f} λ_neigh={lambda_param_1:.3f} λ_util={lambda_param_2:.3f}')
    else:
        ranked_set_scores = estimated_rank_scores

    # Print batch with scores
    for i, (d, s) in enumerate(sorted(zip(docnos_final, ranked_set_scores), key=lambda x: -x[1])):
        tag = rel_tag(d)
        src = doc_source.get(str(d), '?')
        if uses_monot5:
            mt5_idx = docnos_final.index(d)
            print(f'    [{i:2d}] {d}  final={s:.4f}  monot5={reranked_scores[mt5_idx]:.4f}  '
                  f'tasb={dual_score[mt5_idx]:.4f}  src={src}{tag}')
        else:
            print(f'    [{i:2d}] {d}  cer_est={s:.4f}  src={src}{tag}')

    # ── Store results ─────────────────────────────────────────────────────
    for x, docno, score_value in zip(arm, docnos_final, ranked_set_scores):
        results[docno] = score_value
        x.push(score_value)

    # ── Expansion ─────────────────────────────────────────────────────────
    if len(results) < args.budget:
        s_2 = Counter(results).most_common(args.s2)
        s2 = [doc for doc, _ in s_2]
        neighbor_lookup = set(s2).intersection(set(docnos_final))

        print(f'\n  Expansion: {len(neighbor_lookup)} parents (top-{args.s2} ∩ just-scored)')

        kg_support = defaultdict(int)
        kg_best_score = {}
        kg_best_parent_score = {}
        kg_parent_docs = defaultdict(list)
        laff_to_add = []
        kg_all_candidates = []

        for parent_doc in neighbor_lookup:
            neighbors, weights = laff_graph_full.neighbours(parent_doc, weights=True)
            laff_neighbors, kg_candidates, stats = get_kg_expansion(qid, query, parent_doc, neighbors, weights)
            parent_score = results.get(parent_doc, 0.0)

            # LAFF top-16
            laff_new = [n for n in laff_neighbors if n not in all_docnos]
            laff_rel_new = [n for n in laff_new if is_rel(n)]
            for n in laff_neighbors:
                laff_to_add.append((n, parent_doc, parent_score))

            # KG candidates
            kg_new = [n for n, _, _, _ in kg_candidates if n not in all_docnos]
            kg_rel_new = [n for n in kg_new if is_rel(n)]

            if laff_rel_new or kg_rel_new or stats['after_entity'] > 0:
                print(f'    parent={parent_doc} score={parent_score:.4f}')
                print(f'      LAFF16: {len(laff_new)} new ({len(laff_rel_new)} rel)')
                print(f'      KG range: {stats["total_kg_range"]} total -> '
                      f'{stats["after_entity"]} after entity -> '
                      f'{stats["after_bm25"]} after bm25 -> '
                      f'{len(kg_candidates)} selected')
                if kg_rel_new:
                    for n in kg_rel_new:
                        print(f'        ★ KG brings relevant: {n} label={rel_docs[n]}')

            for n, kg_score, _comp, _ln in kg_candidates:
                kg_support[n] += 1
                kg_parent_docs[n].append(parent_doc)
                s = float(kg_score)
                if n not in kg_best_score or s > kg_best_score[n]:
                    kg_best_score[n] = s
                    kg_best_parent_score[n] = parent_score

        # Add LAFF neighbors to arms
        laff_added = 0
        laff_rel_added = 0
        for n, parent_doc, parent_score in laff_to_add:
            if n not in all_docnos:
                a = Arm(n, name=f'laff_{parent_doc}')
                a.push(parent_score)
                arms.append(a)
                all_docnos.append(n)
                doc_source[str(n)] = 'laff'
                laff_added += 1
                if is_rel(n):
                    laff_rel_added += 1
                    print(f'    ★ LAFF adds relevant: {n} label={rel_docs[n]} via parent={parent_doc}')

        # Filter and add KG candidates
        filtered_kg = [n for n, s in kg_support.items() if s >= PRF_MIN_SUPPORT]
        filtered_kg = sorted(filtered_kg, key=lambda n: (kg_support[n], kg_best_score.get(n, 0.0)), reverse=True)
        filtered_kg = filtered_kg[:PRF_KG_MAX_PER_ITER]

        kg_added = 0
        kg_rel_added = 0
        for n in filtered_kg:
            if n not in all_docnos:
                parent_doc = kg_parent_docs[n][0]
                init_score = kg_best_parent_score.get(n, 0.0)
                a = Arm(n, name=f'kg_{parent_doc}')
                a.push(init_score)
                arms.append(a)
                all_docnos.append(n)
                doc_source[str(n)] = 'kg'
                kg_added += 1
                if is_rel(n):
                    kg_rel_added += 1
                    print(f'    ★ KG adds relevant: {n} label={rel_docs[n]} via parent={parent_doc}')

        # Check for relevant docs that were KG candidates but got cut by max_per_iter
        kg_rel_missed = [n for n in kg_best_score if is_rel(n) and n not in filtered_kg and n not in all_docnos]
        if kg_rel_missed:
            print(f'    !! KG found but CUT by prf_max_per_iter={PRF_KG_MAX_PER_ITER}:')
            for n in kg_rel_missed:
                print(f'       {n} label={rel_docs[n]} support={kg_support[n]} score={kg_best_score[n]:.4f}')

        print(f'  Expansion totals: +{laff_added} LAFF ({laff_rel_added} rel), +{kg_added} KG ({kg_rel_added} rel)')

    if count > 0:
        prev_heads = cluster_heads
    else:
        prev_heads = []

    count += 1
    arms = [a for a in arms if not a.is_exhausted()]

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"═"*80}')
print(f'FINAL RESULTS')
print(f'{"═"*80}')

final_ranked = [d for d, _ in Counter(results).most_common()]
top_budget = final_ranked[:args.budget]

rel_in_final = [(d, rel_docs[d]) for d in top_budget if is_rel(d)]
print(f'\nRelevant in top-{args.budget}: {len(rel_in_final)}/{len(rel_docs)}')
recall = len(rel_in_final) / len(rel_docs) if len(rel_docs) > 0 else 0.0
print(f'Recall@{args.budget}: {recall:.4f}')

print(f'\nRelevant docs in top-{args.budget}:')
for d, l in rel_in_final:
    rank = top_budget.index(d)
    score = results[d]
    src = doc_source.get(str(d), '?')
    print(f'  rank={rank:3d}  {d}  score={score:.4f}  label={l}  source={src}')



# Source breakdown
print(f'\nPool composition:')
src_counts = Counter(doc_source.values())
for src, cnt in src_counts.most_common():
    rel_from_src = sum(1 for d, s in doc_source.items() if s == src and is_rel(d))
    print(f'  {src}: {cnt} docs ({rel_from_src} relevant)')

print(f'\n{"═"*80}')
print(f'DIAGNOSIS SUMMARY for qid={args.qid}')
print(f'{"═"*80}')
print(f'Query: "{query_text}"')
print(f'Relevant docs: {len(rel_docs)} total')
print(f'  In BM25 pool:          {len(bm25_rel)}')
print(f'  Reachable via LAFF16:  {laff16_reachable}')
print(f'  Reachable via KG only: {kg_only_reachable}')
print(f'  Unreachable:           {len(unreachable)}')
print(f'Final Recall@{args.budget}: {recall:.4f} ({len(rel_in_final)}/{len(rel_docs)})')
print(f'{"═"*80}')