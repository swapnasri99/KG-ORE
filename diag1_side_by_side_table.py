import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from kg_scorer_unified_patched import create_scorer
import json
import sqlite3
import ir_datasets



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dl", type=int, default=19)
    ap.add_argument("--seed_k", type=int, default=5,
                    help="Top BM25 docs per query used as seeds")
    ap.add_argument("--neighbor_k", type=int, default=32,
                    help="Top-k kept from each seed's 128 LAFF neighbors")
    ap.add_argument("--lk", type=int, default=128,
                    help="How many LAFF neighbors to fetch before reranking")

    # For pure KG test use: alpha=0 beta=0 gamma=1
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kg_mode", type=str, default="log")

    ap.add_argument("--passage_el", type=str, default=None)
    ap.add_argument("--passage_el_db", type=str, default=None)
    ap.add_argument("--query_el", type=str, default=None)
    ap.add_argument("--freebase_dir", type=str, default=None)

    ap.add_argument("--out_csv", type=str, default="diag1_side_by_side_table.csv")

    args = ap.parse_args()

    if not pt.java.started():
        pt.java.init()

    eval_dataset = pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{args.dl}/judged")
    topics = eval_dataset.get_topics()
    qrels = eval_dataset.get_qrels()

    query_entity_map = load_query_entities(args.query_el)

    conn = None
    if args.passage_el_db is not None:
        conn = sqlite3.connect(args.passage_el_db)

    # keep empty for now; later we can enable if needed
    freebase_edges = set()
    # freebase_edges = load_freebase_edges(args.freebase_dir)


    # Relevant docs only (rel >= 2)
    qrels_map = (
        qrels[qrels["label"] >= 2]
        .groupby("qid")["docno"]
        .apply(set)
        .to_dict()
    )

    # BM25 seeds
    bm25 = pt.terrier.Retriever.from_dataset(
        "msmarco_passage",
        "terrier_stemmed",
        wmodel="BM25",
        num_results=max(100, args.seed_k)
    )
    bm25_res = bm25(topics)

    # LAFF graph
    laff_graph = pta.Artifact.from_hf(
        "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
    ).to_limit_k(args.lk)

    # KG scorer
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

    for qid, group in bm25_res.groupby("qid"):
        qid = str(qid)
        relset = qrels_map.get(qid, set())

        if len(relset) == 0:
            continue

        seeds = group.sort_values("score", ascending=False)["docno"].astype(str).tolist()[:args.seed_k]

        laff_union = set()
        kg_union = set()

        for seed in seeds:
            neighbors, weights = laff_graph.neighbours(seed, weights=True)
            neighbor_docnos = [str(n) for n in neighbors]
            weights = np.array(list(weights), dtype=float)

            # Default LAFF ranking
            laff_pairs = sorted(
                zip(neighbor_docnos, weights.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            laff_topk = [docno for docno, _ in laff_pairs[:args.neighbor_k]]
            laff_union.update(laff_topk)

            # KG reranking on the same candidate pool
            rescored = kg_scorer.rescore_neighbors(
                docno=seed,
                neighbor_docnos=neighbor_docnos,
                laff_weights=weights,
                qid=qid,
            )
            kg_topk = [docno for docno, _, _ in rescored[:args.neighbor_k]]
            kg_union.update(kg_topk)

        # Relevant docs captured by each method
        laff_rel = laff_union & relset
        kg_rel = kg_union & relset

        # Shared and unique
        both_rel = laff_rel & kg_rel
        kg_extra_rel = kg_rel - laff_rel       # captured by KG, missed by LAFF
        laff_only_rel = laff_rel - kg_rel      # captured by LAFF, missed by KG

        rows.append({
            "qid": qid,
            "total_relevant_docs": len(relset),
            "laff_relevant_count": len(laff_rel),
            "kg_relevant_count": len(kg_rel),
            "both_relevant_count": len(both_rel),
            "kg_extra_count": len(kg_extra_rel),
            "laff_only_count": len(laff_only_rel),
            "laff_relevant_docs": " ".join(sorted(laff_rel)),
            "kg_relevant_docs": " ".join(sorted(kg_rel)),
            "kg_extra_docs": " ".join(sorted(kg_extra_rel)),
            "laff_only_docs": " ".join(sorted(laff_only_rel)),
        })

    df = pd.DataFrame(rows).sort_values("qid")
    df.to_csv(args.out_csv, index=False)

    pd.set_option("display.max_colwidth", 200)
    table = df[[
    "qid",
    "total_relevant_docs",
    "laff_relevant_count",
    "kg_relevant_count",
    "both_relevant_count",
    "kg_extra_count",
    "laff_only_count"
    ]].reset_index(drop=True)

    table.insert(0, "No", table.index + 1)

    print("=" * 110)
    print("DIAG 1: SIDE-BY-SIDE QUERY TABLE (LAFF vs KG)")
    print("=" * 110)
    print(table.to_string(index=False))

   
    

    """for _, row in df.iterrows():
        print(f"\nQID: {row['qid']}")
        print(f"  Total relevant docs : {row['total_relevant_docs']}")
        print(f"  LAFF relevant count : {row['laff_relevant_count']}")
        print(f"  KG relevant count   : {row['kg_relevant_count']}")
        print(f"  Both count          : {row['both_relevant_count']}")
        print(f"  KG extra count      : {row['kg_extra_count']}")
        print(f"  LAFF only count     : {row['laff_only_count']}")
        print(f"  LAFF relevant docs  : {row['laff_relevant_docs']}")
        print(f"  KG relevant docs    : {row['kg_relevant_docs']}")
        print(f"  KG extra docs       : {row['kg_extra_docs']}")
        print(f"  LAFF only docs      : {row['laff_only_docs']}")"""

    laff_recall_mean = (df["laff_relevant_count"] / df["total_relevant_docs"]).mean()
    kg_recall_mean = (df["kg_relevant_count"] / df["total_relevant_docs"]).mean()

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"Queries                    : {len(df)}")
    print(f"Mean LAFF recall           : {laff_recall_mean:.4f}")
    print(f"Mean KG recall             : {kg_recall_mean:.4f}")
    print(f"Queries where KG > LAFF    : {(df['kg_relevant_count'] > df['laff_relevant_count']).sum()}")
    print(f"Queries where LAFF > KG    : {(df['kg_relevant_count'] < df['laff_relevant_count']).sum()}")
    print(f"Queries where equal        : {(df['kg_relevant_count'] == df['laff_relevant_count']).sum()}")
    print(f"Saved CSV                  : {args.out_csv}")
    print("\n" + "=" * 110)
    print("EXAMPLE DIAGNOSTIC CASES")
    print("=" * 110)

# Query where KG performed better
    kg_better = df[df["kg_relevant_count"] > df["laff_relevant_count"]]

    if not kg_better.empty:
        kg_row = kg_better.sort_values("kg_extra_count", ascending=False).iloc[0]

        example_doc = ""
        if isinstance(kg_row["kg_extra_docs"], str) and kg_row["kg_extra_docs"].strip():
            example_doc = kg_row["kg_extra_docs"].split()[0]

        print("\nKG BETTER CASE")
        print(f"Query ID                : {kg_row['qid']}")
        print(f"Total relevant docs     : {kg_row['total_relevant_docs']}")
        print(f"LAFF relevant captured  : {kg_row['laff_relevant_count']}")
        print(f"KG relevant captured    : {kg_row['kg_relevant_count']}")
        print(f"KG extra relevant docs  : {kg_row['kg_extra_count']}")
        print(f"Example KG-only doc     : {example_doc}")

    else:
        print("\nNo query where KG outperformed LAFF.")


# Query where LAFF performed better
    laff_better = df[df["laff_relevant_count"] > df["kg_relevant_count"]]

    if not laff_better.empty:
        laff_row = laff_better.sort_values("laff_only_count", ascending=False).iloc[0]

        example_doc = ""
        if isinstance(laff_row["laff_only_docs"], str) and laff_row["laff_only_docs"].strip():
            example_doc = laff_row["laff_only_docs"].split()[0]

        print("\nLAFF BETTER CASE")
        print(f"Query ID                : {laff_row['qid']}")
        print(f"Total relevant docs     : {laff_row['total_relevant_docs']}")
        print(f"LAFF relevant captured  : {laff_row['laff_relevant_count']}")
        print(f"KG relevant captured    : {laff_row['kg_relevant_count']}")
        print(f"LAFF-only relevant docs : {laff_row['laff_only_count']}")
        print(f"Example LAFF-only doc   : {example_doc}")

    else:
        print("\nNo query where LAFF outperformed KG.")
        print("\n" + "=" * 110)
    print("QUALITATIVE ANALYSIS: LAFF CAPTURED, KG MISSED")
    print("=" * 110)

    if not laff_better.empty:
        laff_row = laff_better.sort_values("laff_only_count", ascending=False).iloc[0]

        laff_only_doc = ""
        if isinstance(laff_row["laff_only_docs"], str) and laff_row["laff_only_docs"].strip():
            laff_only_doc = laff_row["laff_only_docs"].split()[0]

        if laff_only_doc:
            qid = str(laff_row["qid"])
            query_ents = query_entity_map.get(qid, [])
            passage_ents = get_passage_entities_from_db(conn, laff_only_doc) if conn else []
            relations = find_direct_relations(query_ents, passage_ents, freebase_edges)

            print_case_analysis(
                case_name="LAFF-ONLY RELEVANT DOCUMENT (KG MISSED THIS DOC)",
                qid=qid,
                docno=laff_only_doc,
                topics=topics,
                query_entities=query_ents,
                passage_entities=passage_ents,
                freebase_relations=relations,
            )
        else:
            print("No LAFF-only document found for analysis.")
    else:
        print("No query where LAFF outperformed KG.")
    
    if conn is not None:
        conn.close()

    
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
                    name = e.get("entity_name") or e.get("name") or ""
                    if eid:
                        cleaned.append((eid, name))
            qid_to_ents[qid] = cleaned
    return qid_to_ents


def get_passage_entities_from_db(conn, docno):
    cur = conn.cursor()

    # adjust table/column names if needed for your DB
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
                return [(str(r[0]), r[1]) for r in rows]
        except Exception:
            continue

    return []


def load_freebase_edges(freebase_dir):
    """
    Very lightweight loader:
    expects files with triples like: head<TAB>relation<TAB>tail
    Modify if your local Freebase format is different.
    """
    import os
    edges = set()

    if freebase_dir is None or not os.path.isdir(freebase_dir):
        return edges

    for fname in os.listdir(freebase_dir):
        fpath = os.path.join(freebase_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # adjust extensions if needed
        if not (fname.endswith(".txt") or fname.endswith(".tsv")):
            continue

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    edges.add((h, r, t))
    return edges


def find_direct_relations(query_ents, passage_ents, freebase_edges, max_show=10):
    qids = {eid for eid, _ in query_ents}
    pids = {eid for eid, _ in passage_ents}

    found = []
    for h, r, t in freebase_edges:
        if h in qids and t in pids:
            found.append((h, r, t))
        elif h in pids and t in qids:
            found.append((h, r, t))
        if len(found) >= max_show:
            break
    return found


def print_case_analysis(case_name, qid, docno, topics, query_entities, passage_entities, freebase_relations):
    ds = ir_datasets.load("msmarco-passage")
    docstore = ds.docs_store()

    query_text = topics[topics["qid"].astype(str) == str(qid)]["query"].values[0]
    doc = docstore.get(str(docno))
    passage_text = doc.text if doc is not None else "[Passage not found]"

    print("\n" + "=" * 120)
    print(case_name)
    print("=" * 120)
    print(f"QID        : {qid}")
    print(f"DOCNO      : {docno}")
    print(f"QUERY TEXT : {query_text}")
    print(f"\nPASSAGE TEXT:\n{passage_text}")

    print("\nQUERY ENTITIES:")
    if query_entities:
        for eid, name in query_entities:
            print(f"  - {name} ({eid})")
    else:
        print("  [No query entities found]")

    print("\nPASSAGE ENTITIES:")
    if passage_entities:
        for eid, name in passage_entities:
            print(f"  - {name} ({eid})")
    else:
        print("  [No passage entities found]")

    print("\nDIRECT QUERY↔PASSAGE KG RELATIONS:")
    if freebase_relations:
        for h, r, t in freebase_relations:
            print(f"  - {h} --{r}--> {t}")
    else:
        print("  [No direct relation found in quick check]")

if __name__ == "__main__":
    main()