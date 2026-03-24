import argparse
import os
from pathlib import Path

import pandas as pd
import pyterrier as pt
from ir_measures import nDCG, R


def load_run(path: str) -> pd.DataFrame:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["qid", "Q0", "docno", "rank", "score", "tag"],
        compression="infer",
    )
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    return df


def metric_columns(budget: int):
    return [nDCG @ 10, nDCG @ budget, R(rel=2) @ budget]


def dataset_for_year(dl: int):
    return pt.get_dataset(f"irds:msmarco-passage/trec-dl-20{dl}/judged")


def default_gar_path(dl: int, budget: int) -> str:
    return f"runs/adaptive/dl{dl}/gbm25/GAR.c{budget}.res.gz"


def default_quam_path(dl: int, budget: int) -> str:
    return f"runs/adaptive/dl{dl}/gbm25/QuAM.c{budget}.res.gz"


def default_kg_path(dl: int, budget: int, kg_tag: str) -> str:
    return f"runs/adaptive/dl{dl}/kg_ore/{kg_tag}/ORE_{kg_tag}.c{budget}.DL{dl}.res.gz"


def print_main_table(result: pd.DataFrame, budget: int):
    rcol = f"R(rel=2)@{budget}"
    ncol = f"nDCG@{budget}"

    print("\n" + "=" * 88)
    print("RESULTS")
    print("=" * 88)
    print(f"{'Method':<28s} | {'nDCG@10':>8s} | {ncol:>8s} | {rcol:>12s}")
    print("-" * 88)
    for _, row in result.iterrows():
        print(
            f"{row['name']:<28s} | "
            f"{row['nDCG@10']:>8.4f} | "
            f"{row[ncol]:>8.4f} | "
            f"{row[rcol]:>12.4f}"
        )
    print("=" * 88)



def print_pairwise_significance(
    runs,
    names,
    topics,
    qrels,
    budget: int,
    kg_name: str,
    baseline_name: str,
    correction: str,
):
    measures = metric_columns(budget)
    baseline_idx = names.index(baseline_name)

    result = pt.Experiment(
        runs,
        topics,
        qrels,
        measures,
        names=names,
        baseline=baseline_idx,
        correction=correction,
    )

    kg_row = result[result["name"] == kg_name].iloc[0]
    base_row = result[result["name"] == baseline_name].iloc[0]

    metric_names = ["nDCG@10", f"nDCG@{budget}", f"R(rel=2)@{budget}"]

    print("\n" + "-" * 88)
    print(f"KG-ORE vs {baseline_name}  (correction={correction})")
    print("-" * 88)
    for metric in metric_names:
        kg_val = float(kg_row[metric])
        base_val = float(base_row[metric])
        diff = kg_val - base_val
        pct = (diff / base_val * 100.0) if base_val != 0 else 0.0

        p_col = f"{metric} p-value corrected"
        reject_col = f"{metric} reject"

        if p_col in result.columns and pd.notna(kg_row.get(p_col, pd.NA)):
            pval = float(kg_row[p_col])
            reject = bool(kg_row[reject_col]) if reject_col in result.columns else False
            sig = "YES" if reject else "no"
            print(
                f"{metric:<15s}  {baseline_name}={base_val:.4f}  "
                f"KG-ORE={kg_val:.4f}  diff={diff:+.4f} ({pct:+.2f}%)  "
                f"p={pval:.6f}  significant={sig}"
            )
        else:
            print(
                f"{metric:<15s}  {baseline_name}={base_val:.4f}  "
                f"KG-ORE={kg_val:.4f}  diff={diff:+.4f} ({pct:+.2f}%)"
            )



def run_one_year(dl: int, budget: int, gar_path: str, quam_path: str, kg_path: str,
                 kg_name: str, correction: str):
    print("\n" + "#" * 100)
    print(f"TREC DL 20{dl}")
    print("#" * 100)
    print(f"GAR   : {gar_path}")
    print(f"QuAM  : {quam_path}")
    print(f"KG-ORE: {kg_path}")

    dataset = dataset_for_year(dl)
    topics = dataset.get_topics()
    qrels = dataset.get_qrels()

    runs = [load_run(gar_path), load_run(quam_path), load_run(kg_path)]
    names = ["GAR", "QuAM", kg_name]

    summary = pt.Experiment(
        runs,
        topics,
        qrels,
        metric_columns(budget),
        names=names,
    )
    print_main_table(summary, budget)

    print_pairwise_significance(
        runs=runs,
        names=names,
        topics=topics,
        qrels=qrels,
        budget=budget,
        kg_name=kg_name,
        baseline_name="GAR",
        correction=correction,
    )
    print_pairwise_significance(
        runs=runs,
        names=names,
        topics=topics,
        qrels=qrels,
        budget=budget,
        kg_name=kg_name,
        baseline_name="QuAM",
        correction=correction,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance test for saved KG-ORE, GAR, and QuAM run files"
    )
    parser.add_argument("--dl", type=int, choices=[19, 20], default=None,
                        help="Run only one benchmark year. Omit to run both 19 and 20.")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--correction", choices=["bonferroni", "holm"], default="bonferroni")
    parser.add_argument("--kg-name", type=str, default="KG-ORE UNION")

    parser.add_argument("--dl19-gar", type=str, default=None)
    parser.add_argument("--dl19-quam", type=str, default=None)
    parser.add_argument("--dl19-kg", type=str, default=None)

    parser.add_argument("--dl20-gar", type=str, default=None)
    parser.add_argument("--dl20-quam", type=str, default=None)
    parser.add_argument("--dl20-kg", type=str, default=None)

    parser.add_argument(
        "--kg-tag",
        type=str,
        default=None,
        help=(
            "Optional folder/run tag used for both years, for example E0.3_K0.7_CERKG_UNION. "
            "If given, the KG path is auto-built as "
            "runs/adaptive/dlXX/kg_ore/<kg_tag>/ORE_<kg_tag>.c<budget>.DLXX.res.gz"
        ),
    )

    args = parser.parse_args()

    if not pt.started():
        pt.java.init()

    years = [args.dl] if args.dl is not None else [19, 20]

    for dl in years:
        if dl == 19:
            gar_path = args.dl19_gar or default_gar_path(19, args.budget)
            quam_path = args.dl19_quam or default_quam_path(19, args.budget)
            kg_path = args.dl19_kg or (default_kg_path(19, args.budget, args.kg_tag) if args.kg_tag else None)
        else:
            gar_path = args.dl20_gar or default_gar_path(20, args.budget)
            quam_path = args.dl20_quam or default_quam_path(20, args.budget)
            kg_path = args.dl20_kg or (default_kg_path(20, args.budget, args.kg_tag) if args.kg_tag else None)

        missing = []
        if not gar_path:
            missing.append(f"--dl{dl}-gar")
        if not quam_path:
            missing.append(f"--dl{dl}-quam")
        if not kg_path:
            missing.append(f"--dl{dl}-kg or --kg-tag")
        if missing:
            raise ValueError(f"For DL{dl}, missing required path arguments: {', '.join(missing)}")

        run_one_year(
            dl=dl,
            budget=args.budget,
            gar_path=gar_path,
            quam_path=quam_path,
            kg_path=kg_path,
            kg_name=args.kg_name,
            correction=args.correction,
        )


if __name__ == "__main__":
    main()