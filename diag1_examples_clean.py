import argparse
import json
import sqlite3
from collections import defaultdict

import ir_datasets
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from kg_scorer_unified_patched import create_scorer


def load_query_entities(query_el_path):
    qid_to_ents = {}
    if query_el_path is None:
        return qid_to_ents
    with open(query_el_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("id") or row.get("qid"))
            ents = row.get("entities", [])
            cleaned = []
            for e in ents:
                if isinstance(e, dict):
                    eid = str(e.get("entity_id") or e.get("id") or "")
                    name = str(e.get("entity_name") or e.get("name") or "")
                    if eid:
                        cleaned.append((eid, name))
            qid_to_ents[qid] = cleaned
    return qid_to_ents


def get_passage_entities_from_db(conn, docno):
    if conn is None:
        return []
    cur = conn.cursor()
    possible_queries = [
        ("SELECT entity_id, entity_name FROM passage_entities WHERE docno = ?", (str(docno),)),
        ("SELECT entity_id, entity_name FROM entities WHERE docno = ?", (str(docno),)),
        ("SELECT entity_id, entity_name FROM passage_entities WHERE passage_id = ?", (str(docno),)),
    ]
    for sql, params in possible_queries:
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            if rows:
                return [(str(r[0]), str(r[1])) for r in rows]
        except sqlite3.Error:
            continue
    return []


def entity_names(ents):
    return [name for _, name in ents if name]


def get_passage_text(docstore, docno, max_chars=900):
    try:
        doc = docstore.get(str(docno))
        if doc is None:
            return ""
        text = getattr(doc, "text", "") or ""
        text = str(text).replace("\n", " ").strip()
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except Exception:
        return ""


def summarize_seed_hits(seed_hits):
    if not seed_hits:
        return "not seen under stored seeds"

    parts = []
    for hit in seed_hits:
        laff_part = f"LAFF top-{hit['neighbor_k']}" if hit["laff_in_topk"] else "LAFF miss"
        kg_part = f"KG top-{hit['neighbor_k']}" if hit["kg_in_topk"] else "KG miss"
        parts.append(
            f"seed={hit['seed']} | {laff_part} | {kg_part} | one_hop_relations={hit['kg_raw']}"
        )
    return " ; ".join(parts)


def print_query_summary(qid, query, relset, laff_rel, kg_rel, kg_extra_rel, laff_only_rel):
    print("\n" + "=" * 120)
    print(f"QID   : {qid}")
    print(f"Query : {query}")
    print(f"All relevant docs ({len(relset)}): {sorted(relset)}")
    print(f"LAFF captured ({len(laff_rel)}): {sorted(laff_rel)}")
    print(f"KG captured   ({len(kg_rel)}): {sorted(kg_rel)}")
    print(f"KG-only docs  ({len(kg_extra_rel)}): {sorted(kg_extra_rel)}")
    print(f"LAFF-only docs({len(laff_only_rel)}): {sorted(laff_only_rel)}")



def print_doc_compact(label, qid, query, docno, qrel_label, ents, text, seed_hits):
    relation_counts = [int(hit["kg_raw"]) for hit in seed_hits if hit.get("kg_raw") is not None]
    max_rel = max(relation_counts) if relation_counts else 0
    print("\n" + "-" * 120)
    print(label)
    print(f"QID               : {qid}")
    print(f"Query             : {query}")
    print(f"Docno             : {docno}")
    print(f"Qrel label        : {qrel_label}")
    print(f"One-hop relations : {max_rel}")
    print(f"Passage entities  : {entity_names(ents)[:20]}")
    print(f"Seen via          : {summarize_seed_hits(seed_hits)}")
    print(f"Text              : {text}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", type=int, default=19)
    ap.add_argument("--seed_k", type=int, default=5)
    ap.add_argument("--neighbor_k", type=int, default=16)
    ap.add_argument("--lk", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kg_mode", type=str, default="minmax")
    ap.add_argument("--passage_el", type=str, default=None)
    ap.add_argument("--passage_el_db", type=str, default=None)
    ap.add_argument("--query_el", type=str, default=None)
    ap.add_argument("--freebase_dir", type=str, default=None)
    ap.add_argument("--inspect_qid", type=str, default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--print_limit", type=int, default=2)
    ap.add_argument("--out_csv", type=str, default="diag1_examples_clean.csv")
    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    eval_dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics = eval_dataset.get_topics()
    qrels = eval_dataset.get_qrels()
    qrels_map = (
        qrels[qrels["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(lambda s: set(map(str, s)))
        .to_dict()
    )
    qrel_label_map = {(str(r.qid), str(r.docno)): int(r.label) for r in qrels.itertuples(index=False)}
    topic_map = {str(r.qid): r.query for r in topics.itertuples(index=False)}

    query_entity_map = load_query_entities(args.query_el)
    conn = sqlite3.connect(args.passage_el_db) if args.passage_el_db else None
    docstore = ir_datasets.load("msmarco-passage").docs_store()

    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", num_results=max(100, args.seed_k)
    )
    bm25_res = bm25(topics)
    if args.inspect_qid is not None:
        bm25_res = bm25_res[bm25_res["qid"].astype(str) == str(args.inspect_qid)]

    laff_graph = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    kg_scorer = create_scorer(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        kg_score_mode=args.kg_mode,
        freebase_dir=args.freebase_dir,
        passage_el_path=args.passage_el,
        query_el_path=args.query_el,
        passage_el_db=args.passage_el_db,
    )

    rows = []
    per_query_debug = {}
    processed = 0

    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        relset = qrels_map.get(qid, set())
        if not relset:
            continue
        processed += 1
        if args.max_queries is not None and processed > args.max_queries:
            break

        seeds = (
            group.sort_values("score", ascending=False)["docno"].astype(str).tolist()[: args.seed_k]
        )

        laff_union = set()
        kg_union = set()
        seed_details = []

        for seed in seeds:
            neighbors, weights = laff_graph.neighbours(seed, weights=True)
            neighbor_docnos = [str(n) for n in neighbors]
            weights = np.array(list(weights), dtype=float)

            laff_pairs = sorted(zip(neighbor_docnos, weights.tolist()), key=lambda x: x[1], reverse=True)
            laff_topk = [docno for docno, _ in laff_pairs[: args.neighbor_k]]
            laff_union.update(laff_topk)

            rescored = kg_scorer.rescore_neighbors(
                docno=seed,
                neighbor_docnos=neighbor_docnos,
                laff_weights=weights,
                qid=qid,
            )
            kg_topk = [docno for docno, _, _ in rescored[: args.neighbor_k]]
            kg_union.update(kg_topk)

            laff_topk_set = set(laff_topk)
            kg_topk_set = set(kg_topk)
            kg_comp_map_all = {docno: comp for docno, _, comp in rescored}

            seed_details.append(
                {
                    "seed": seed,
                    "laff_topk_set": laff_topk_set,
                    "kg_topk_set": kg_topk_set,
                    "kg_comp_map_all": kg_comp_map_all,
                }
            )

        laff_rel = laff_union & relset
        kg_rel = kg_union & relset
        kg_extra_rel = kg_rel - laff_rel
        laff_only_rel = laff_rel - kg_rel

        rows.append(
            {
                "qid": qid,
                "query": topic_map.get(qid, ""),
                "total_relevant_docs": len(relset),
                "laff_relevant_count": len(laff_rel),
                "kg_relevant_count": len(kg_rel),
                "kg_extra_count": len(kg_extra_rel),
                "laff_only_count": len(laff_only_rel),
                "kg_extra_docs": " ".join(sorted(kg_extra_rel)),
                "laff_only_docs": " ".join(sorted(laff_only_rel)),
            }
        )

        per_query_debug[qid] = {
            "query": topic_map.get(qid, ""),
            "relset": relset,
            "laff_rel": laff_rel,
            "kg_rel": kg_rel,
            "kg_extra_rel": sorted(kg_extra_rel),
            "laff_only_rel": sorted(laff_only_rel),
            "seed_details": seed_details,
        }

    df = pd.DataFrame(rows).sort_values(["kg_extra_count", "qid"], ascending=[False, True])
    df.to_csv(args.out_csv, index=False)
    print(f"Saved CSV: {args.out_csv}")

    if df.empty:
        print("No rows produced.")
        return

    inspect_qids = [str(args.inspect_qid)] if args.inspect_qid is not None else df[df["kg_extra_count"] > 0]["qid"].astype(str).tolist()[: args.print_limit]

    for qid in inspect_qids:
        dbg = per_query_debug.get(qid)
        if not dbg:
            continue

        query = dbg["query"]
        relset = dbg["relset"]
        laff_rel = dbg["laff_rel"]
        kg_rel = dbg["kg_rel"]
        kg_only_docs = dbg["kg_extra_rel"][: args.print_limit]
        laff_only_docs = dbg["laff_only_rel"][: args.print_limit]

        print_query_summary(qid, query, relset, laff_rel, kg_rel, dbg["kg_extra_rel"], dbg["laff_only_rel"])
        print(f"Query entities       : {entity_names(query_entity_map.get(qid, []))[:20]}")

        for docno in kg_only_docs:
            ents = get_passage_entities_from_db(conn, docno)
            text = get_passage_text(docstore, docno)
            seed_hits = []
            for sd in dbg["seed_details"]:
                comp = sd["kg_comp_map_all"].get(docno)
                if comp is None:
                    continue
                seed_hits.append(
                    {
                        "seed": sd["seed"],
                        "neighbor_k": args.neighbor_k,
                        "laff_in_topk": docno in sd["laff_topk_set"],
                        "kg_in_topk": docno in sd["kg_topk_set"],
                        "kg_raw": getattr(comp, "kg_raw", 0.0),
                    }
                )
            print_doc_compact(
                label="KG captured this relevant doc, LAFF missed it",
                qid=qid,
                query=query,
                docno=docno,
                qrel_label=qrel_label_map.get((qid, str(docno)), -1),
                ents=ents,
                text=text,
                seed_hits=seed_hits,
            )

        for docno in laff_only_docs:
            ents = get_passage_entities_from_db(conn, docno)
            text = get_passage_text(docstore, docno)
            seed_hits = []
            for sd in dbg["seed_details"]:
                comp = sd["kg_comp_map_all"].get(docno)
                if comp is None:
                    continue
                seed_hits.append(
                    {
                        "seed": sd["seed"],
                        "neighbor_k": args.neighbor_k,
                        "laff_in_topk": docno in sd["laff_topk_set"],
                        "kg_in_topk": docno in sd["kg_topk_set"],
                        "kg_raw": getattr(comp, "kg_raw", 0.0),
                    }
                )
            print_doc_compact(
                label="LAFF captured this relevant doc, KG missed it",
                qid=qid,
                query=query,
                docno=docno,
                qrel_label=qrel_label_map.get((qid, str(docno)), -1),
                ents=ents,
                text=text,
                seed_hits=seed_hits,
            )

    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()
