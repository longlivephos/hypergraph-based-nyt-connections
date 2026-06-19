import argparse
import csv
import math
import time
import itertools
from collections import defaultdict
from pathlib import Path

import nltk
import pandas as pd


def _ensure_wordnet() -> None:
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"[Setup] Downloading NLTK resource: {resource} …")
            nltk.download(resource, quiet=True)

_ensure_wordnet()
from nltk.corpus import wordnet as wn      



def load_dataset(csv_path: str) -> list[dict]:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")

    puzzles = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            words = []
            for i in range(1, 17):
                w = row.get(f"word_{i}", "").strip().upper()
                if w:
                    words.append(w)

            solution = []
            for c in range(1, 5):
                grp = frozenset(
                    row.get(f"cat{c}_word{k}", "").strip().upper()
                    for k in range(1, 5)
                ) - {""}
                if len(grp) == 4:
                    solution.append(grp)

            if len(words) == 16 and len(solution) == 4:
                puzzles.append({
                    "puzzle_id": int(row["puzzle_id"]),
                    "date":      row["date"].strip(),
                    "words":     words,
                    "solution":  solution,
                    "categories": ["Yellow","Green","Blue","Purple"],
                })

    print(f"[Load] {len(puzzles)} puzzle dimuat dari '{csv_path}'")
    return puzzles



_sim_cache: dict[tuple[str, str], float] = {}


def _path_sim(word_a: str, word_b: str) -> float:

    key = (word_a, word_b) if word_a <= word_b else (word_b, word_a)
    if key in _sim_cache:
        return _sim_cache[key]

    def to_wn(w: str) -> str:
        return w.lower().replace(" ", "_").replace("'", "")

    syns_a = wn.synsets(to_wn(word_a))[:3]
    syns_b = wn.synsets(to_wn(word_b))[:3]

    if not syns_a or not syns_b:
        _sim_cache[key] = 0.0
        return 0.0

    best = 0.0
    for sa in syns_a:
        for sb in syns_b:
            try:
                v = wn.path_similarity(sa, sb)
                if v and v > best:
                    best = v
            except Exception:
                pass

    _sim_cache[key] = best
    return best


def coherence(group: tuple | frozenset | list) -> float:
    words = list(group)
    pairs = list(itertools.combinations(words, 2))
    if not pairs:
        return 0.0
    total = sum(_path_sim(a, b) for a, b in pairs)
    return total / len(pairs)



def generate_candidate_hyperedges(
    words: list[str],
    solution: list[frozenset],
    tau: float = 0.10,
) -> list[frozenset]:
    sol_set = set(solution)
    candidates = []

    for combo in itertools.combinations(words, 4):
        fset = frozenset(combo)
        if fset in sol_set:         
            continue
        k = coherence(fset)
        if k >= tau:
            candidates.append(fset)

    return candidates




def build_hypergraph(
    words: list[str],
    solution: list[frozenset],
    ecand: list[frozenset],
) -> dict:
    return {
        "V":     words,
        "Esol":  solution,
        "Ecand": ecand,
        "E":     solution + ecand,
    }


def ambiguity_score(H: dict) -> float:
    V = H["V"]
    E = H["E"]        

    degree: dict[str, int] = defaultdict(int)
    for edge in E:
        for w in edge:
            degree[w] += 1

    total = sum(max(degree[v] - 1, 0) for v in V)
    return total / len(V)


def structural_overlap_score(H: dict) -> float:
    Esol  = H["Esol"]
    Ecand = H["Ecand"]

    if not Ecand:
        return 0.0

    total = 0
    for sol_edge in Esol:
        best = max(len(sol_edge & cand) for cand in Ecand)
        total += best

    return total / len(Esol)


def ambiguity_score_per_category(H: dict) -> dict:
    E = H["E"]
    degree = defaultdict(int)
    for edge in E:
        for w in edge:
            degree[w] += 1

    results = {}
    labels = H.get("labels", ["Yellow","Green","Blue","Purple"])

    for label, edge in zip(labels, H["Esol"]):
        total = sum(max(degree[w] - 1, 0) for w in edge)
        results[f"AS_{label.lower()}"] = total / len(edge)

    return results


def structural_overlap_score_per_category(H: dict) -> dict:
    Ecand = H["Ecand"]
    labels = H.get("labels", ["Yellow","Green","Blue","Purple"])

    results = {}
    for label, sol_edge in zip(labels, H["Esol"]):
        best = max((len(sol_edge & cand) for cand in Ecand), default=0)
        results[f"SOS_{label.lower()}"] = float(best)

    return results



def evaluate_puzzle(puzzle: dict, tau: float = 0.10, verbose: bool = False) -> dict:
    pid   = puzzle["puzzle_id"]
    words = puzzle["words"]
    sol   = puzzle["solution"]

    t0 = time.perf_counter()

    ecand = generate_candidate_hyperedges(words, sol, tau=tau)

    H = build_hypergraph(words, sol, ecand)
    H["labels"] = ["Yellow","Green","Blue","Purple"]

    AS  = ambiguity_score(H)
    SOS = structural_overlap_score(H)
    as_cat = ambiguity_score_per_category(H)
    sos_cat = structural_overlap_score_per_category(H)

    elapsed = time.perf_counter() - t0

    if verbose:
        _print_puzzle_detail(puzzle, H, AS, SOS, tau, elapsed)

    return {
        "puzzle_id":               pid,
        "date":                    puzzle["date"],
        "tau":                     tau,
        "n_words":                 len(words),
        "n_solution_hyperedges":   len(H["Esol"]),
        "n_candidate_hyperedges":  len(H["Ecand"]),
        "n_total_hyperedges":      len(H["E"]),
        "ambiguity_score":         round(AS,  6),
        "structural_overlap_score": round(SOS, 6),
        "elapsed_s":               round(elapsed, 3),
        **as_cat,
        **sos_cat,
    }


def _print_puzzle_detail(puzzle, H, AS, SOS, tau, elapsed):
    """Helper print detail satu puzzle (untuk mode verbose/single)."""
    pid = puzzle["puzzle_id"]
    print(f"\n{'═'*65}")
    print(f"  Puzzle #{pid}  |  {puzzle['date']}")
    print(f"{'─'*65}")
    print(f"  16 Kata: {puzzle['words']}")
    print(f"\n  Solusi (Esol):")
    for i, s in enumerate(H["Esol"], 1):
        print(f"    [{i}] {sorted(s)}")
    print(f"\n  Threshold τ = {tau}")
    print(f"SOS  : {SOS:.4f}")
    print(f"Ecand: {len(H['Ecand'])}")
    print(f"Time : {elapsed:.2f}s")
    if H["Ecand"]:
        top5 = sorted(H["Ecand"], key=lambda x: coherence(x), reverse=True)[:5]
        print(f"\n  Top-5 candidate hyperedge (κ tertinggi):")
        for cand in top5:
            k = coherence(cand)
            overlaps = [len(cand & s) for s in H["Esol"]]
            print(f"    κ={k:.3f}  overlap_max={max(overlaps)}  {sorted(cand)}")



def evaluate_all(
    csv_path: str,
    tau: float = 0.10,
) -> pd.DataFrame:
    puzzles = load_dataset(csv_path)
    results = []
    total   = len(puzzles)

    print(f"\n{'═'*65}")
    print(f"  Evaluasi {total} puzzle  |  τ = {tau}")
    print(f"{'═'*65}")
    print(f"  {'#':>3}  {'puzzle_id':>9}  {'Ecand':>6}  {'AS':>8}  {'SOS':>6}  {'t(s)':>5}")
    print(f"  {'─'*3}  {'─'*9}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*5}")

    for i, puzzle in enumerate(puzzles, 1):
        try:
            res = evaluate_puzzle(puzzle, tau=tau, verbose=False)
            print(
                f"  {i:>3}  #{res['puzzle_id']:>8}  "
                f"{res['n_candidate_hyperedges']:>6}  "
                f"{res['ambiguity_score']:>8.4f}  "
                f"{res['structural_overlap_score']:>6.4f}  "
                f"{res['elapsed_s']:>5.1f}"
            )
            results.append(res)
        except Exception as exc:
            print(f"  {i:>3}  #{puzzle['puzzle_id']:>8}  ERROR: {exc}")
            results.append({
                "puzzle_id": puzzle["puzzle_id"],
                "date":      puzzle["date"],
                "error":     str(exc),
            })

    print(f"{'─'*65}")
    print(f"  Selesai. {len([r for r in results if 'ambiguity_score' in r])}/{total} berhasil.")

    return pd.DataFrame(results)




def print_summary(df: pd.DataFrame) -> None:
    """Cetak statistik AS dan SOS per tingkat kesulitan."""

    print(f"\n{'═'*65}")
    print("  RINGKASAN BERDASARKAN TINGKAT KESULITAN")
    print(f"{'═'*65}")

    difficulties = ["yellow", "green", "blue", "purple"]

    print("\nTABEL AS")
    rows = []
    for d in difficulties:
        col = f"AS_{d}"
        if col in df.columns:
            rows.append({
                "Difficulty": d.capitalize(),
                "Mean AS": round(df[col].mean(), 4),
                "Std": round(df[col].std(), 4),
                "Median": round(df[col].median(), 4),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nTABEL SOS")
    rows = []
    for d in difficulties:
        col = f"SOS_{d}"
        if col in df.columns:
            rows.append({
                "Difficulty": d.capitalize(),
                "Mean SOS": round(df[col].mean(), 4),
                "Std": round(df[col].std(), 4),
                "Median": round(df[col].median(), 4),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'═'*65}")

def print_global_statistics(df: pd.DataFrame) -> None:
    print("\n" + "="*65)
    print("DESCRIPTIVE STATISTICS OF ALL PUZZLES")
    print("="*65)

    rows = []

    metrics = {
        "AS": "ambiguity_score",
        "SOS": "structural_overlap_score",
        "Ecand": "n_candidate_hyperedges"
    }

    for name, col in metrics.items():
        rows.append({
            "Metric": name,
            "Mean": round(df[col].mean(), 4),
            "Std": round(df[col].std(), 4),
            "Median": round(df[col].median(), 4),
            "Min": round(df[col].min(), 4),
            "Max": round(df[col].max(), 4),
        })

    print(pd.DataFrame(rows).to_string(index=False))


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Simpan DataFrame ke CSV, kolom output yang relevan saja."""
    col_order = [
        "puzzle_id", "date", "tau",
        "ambiguity_score", "structural_overlap_score",
        "n_solution_hyperedges", "n_candidate_hyperedges", "n_total_hyperedges",
        "n_words", "elapsed_s",
        "AS_yellow","AS_green","AS_blue","AS_purple",
        "SOS_yellow","SOS_green","SOS_blue","SOS_purple",
    ]
    out_cols = [c for c in col_order if c in df.columns]
    df[out_cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Save] Hasil disimpan ke '{output_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NYT Connections — Ambiguity Score (AS) & Structural Overlap Score (SOS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python connections_analysis.py
  python connections_analysis.py --csv data/puzzles.csv --tau 0.12 --output hasil.csv
  python connections_analysis.py --single 1019     # debug satu puzzle
  python connections_analysis.py --single 1000 --tau 0.08
        """,
    )
    parser.add_argument(
        "--csv",
        default="nyt_connections_reselected_40.csv",
        help="Path ke file CSV puzzle (default: nyt_connections_reselected_40.csv)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Threshold coherence κ untuk filter candidate hyperedge (default: 0.10)",
    )
    parser.add_argument(
        "--output",
        default="connections_results.csv",
        help="Path output CSV hasil (default: connections_results.csv)",
    )
    parser.add_argument(
        "--single",
        type=int,
        default=None,
        metavar="PUZZLE_ID",
        help="Evaluasi hanya satu puzzle (untuk debugging/inspeksi detail)",
    )

    args = parser.parse_args()

    wall_start = time.perf_counter()

    if args.single is not None:
        puzzles = load_dataset(args.csv)
        target  = next((p for p in puzzles if p["puzzle_id"] == args.single), None)
        if target is None:
            print(f"[Error] Puzzle #{args.single} tidak ditemukan di '{args.csv}'.")
            return
        result = evaluate_puzzle(target, tau=args.tau, verbose=True)
        print("\n[Result dict]")
        for k, v in result.items():
            if k != "words":
                print(f"  {k:<30}: {v}")

    else:
        df = evaluate_all(args.csv, tau=args.tau)

        print_global_statistics(df)
        print_summary(df)

        save_results(df, args.output)

    wall_total = time.perf_counter() - wall_start
    print(f"\n[Done] Total waktu: {wall_total:.1f}s")


if __name__ == "__main__":
    main()