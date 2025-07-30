#!/usr/bin/env python3
from __future__ import annotations
import math

# Helper function for numerically stable log-combination
def _log_comb(n: int, k: int) -> float:
    """Return log(C(n,k)) using lgamma to avoid OverflowError."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union, Optional
 # Placeholder; re‑defined inside each function where needed.
_background_all: Set[str] = set()

try:
    import numpy as np  # noqa: F401
except ImportError as _e:
    raise SystemExit("numpy is required: %s" % _e)

try:
    import pandas as pd
except ImportError as _e:
    raise SystemExit("pandas is required: %s" % _e)

from statsmodels.stats.multitest import multipletests


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

TOKEN_RE = re.compile(r"[\s,;]+")


def parse_gene_input(raw: Union[str, Sequence[str]]) -> List[str]:
    """Parse user gene input into an *order‑preserving*, duplicate‑free list.

    Handles strings (comma/whitespace separated) or any iterable of strings.
    Surrounding single or double‑quotes are removed from each token.
    """
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in TOKEN_RE.split(raw) if tok.strip()]
    else:
        tokens = [str(tok).strip() for tok in raw if str(tok).strip()]

    # Strip wrapping quotes (both 'gene' and "gene")
    tokens = [tok.strip('"').strip("'") for tok in tokens]

    # order‑preserving dedupe
    return list(dict.fromkeys(tokens))


def normalize_fitdb(fitdb_raw):
    return {k: (v if isinstance(v, set) else set(parse_gene_input(v)))
            for k,v in fitdb_raw.items() if v}

def load_fitdb_json(path: Union[str, Path]) -> Dict[str, Set[str]]:
    """Load JSON mapping {set_name: [gene1, gene2, ...]} -> dict of sets."""
    path = Path(path)
    with path.open() as fh:
        obj = json.load(fh)
    out: Dict[str, Set[str]] = {}
    for k, v in obj.items():
        out[k] = set(parse_gene_input(v))
    return out


# ------------------------------------------------------------------
# Core enrichment
# ------------------------------------------------------------------
def compare_gene_sets_fast(
    my_genes: Union[str, Sequence[str]],
    fitdb: Mapping[str, Set[str]],
    *,
    min_overlap: int = 1,
    build_matrix: bool = True,
    background_input: Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
    background_db:    Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
):
    """One-pass enrichment of *my_genes* against *fitdb* (dict of gene-sets).

    Parameters
    ----------
    my_genes : list/str
        Query gene list or path/CSV string.
    fitdb : dict[str, set[str]]
        Mapping of set_name → member genes.
    min_overlap : int, default 1
        Require at least this many overlaps to compute a p-value.
    build_matrix : bool, default True
        Also return a genes×sets membership matrix.
    background_input : list/str/dict or None
        Universe for *my_genes*.  If None, inferred (see below).
    background_db : list/str/dict or None
        Per-set or global universe(s) for DB sets.

    Universe logic
    --------------
    For each gene-set we construct:
        • background   =  (b_input ∩ b_set)  if both supplied
                        =  b_set             if only per-set supplied
                        =  b_input           if only input supplied
                        =  union(*fitdb.values())   fallback
        • n_set        =  |query ∩ background|
    This guarantees   n_set ≤ N  and  k ≤ n_set,  avoiding NaN.
    """

    # --------------------------------------------------
    # Parse query list
    # --------------------------------------------------
    genes = parse_gene_input(my_genes)
    my_raw_set = set(genes)

    # --------------------------------------------------
    # Load / normalise background_db
    # --------------------------------------------------
    if background_db is None:
        b_db_map  = None
        b_db_glob = None
    elif isinstance(background_db, Mapping):
        b_db_map  = {
            k: (v if isinstance(v, set) else set(parse_gene_input(v)))
            for k, v in background_db.items()
        }
        b_db_glob = set().union(*b_db_map.values())
    else:
        b_db_map  = None
        b_db_glob = set(parse_gene_input(background_db))

    # --------------------------------------------------
    # Derive input background (b_input)
    # --------------------------------------------------
    if background_input is None:
        if isinstance(background_db, Mapping):
            whole_key = next((k for k in background_db if k.lower() in {
                "whole_genome", "wholegenome", "genome"}), None)
            b_input = set(background_db[whole_key]) if whole_key else set().union(*background_db.values())
        else:
            b_input = set().union(*fitdb.values())
    elif isinstance(background_input, Mapping):
        b_input = set().union(*(
            v if isinstance(v, set) else set(parse_gene_input(v))
            for v in background_input.values()))
    else:
        b_input = set(parse_gene_input(background_input))

    # --------------------------------------------------
    # Fallback global universe
    # --------------------------------------------------
    global _background_all
    _background_all = set().union(*fitdb.values())

    # --------------------------------------------------
    # Prepare results containers
    # --------------------------------------------------
    gene_rank = {g: i for i, g in enumerate(genes)}

    names, K_vec, k_vec, p_vec, overlaps_serialized = [], [], [], [], []
    mat, col_names = None, None
    if build_matrix:
        import numpy as _np
        mat = _np.zeros((len(genes), len(fitdb)), dtype=_np.uint8)
        col_names = list(fitdb.keys())

    # --------------------------------------------------
    # Iterate gene sets
    # --------------------------------------------------
    for j, (set_name, members) in enumerate(fitdb.items()):
        # per-set background
        b_set = b_db_map.get(set_name) if b_db_map is not None else b_db_glob

        if b_set is not None:
            background = b_input & b_set if b_input is not None else b_set
        else:
            background = b_input if b_input is not None else _background_all
        if not background:
            continue

        # query restricted to this universe
        n_set = len(my_raw_set & background)
        if n_set < min_overlap:
            continue

        members_in_bg = members & background
        K = len(members_in_bg)
        if K == 0:
            continue

        ov = my_raw_set & members_in_bg
        k = len(ov)
        if k < min_overlap:
            continue
        if k > K or k > n_set:
            continue  # impossible table

        # optional matrix
        if build_matrix and k:
            import numpy as _np
            idx = [gene_rank[g] for g in members_in_bg if g in gene_rank]
            if idx:
                mat[_np.asarray(idx, dtype=int), j] = 1

        N = len(background)

        from scipy.stats import hypergeom
        p = float(hypergeom.sf(k - 1, N, K, n_set))

        names.append(set_name)
        K_vec.append(len(members))
        k_vec.append(k)
        p_vec.append(p)
        overlaps_serialized.append(",".join(sorted(ov, key=lambda g: gene_rank[g])))

    # --------------------------------------------------
    # Build DataFrame outputs
    # --------------------------------------------------
    results_df = pd.DataFrame({
        "name": names,
        "K": K_vec,
        "k": k_vec,
        "ratio": [k / K for k, K in zip(k_vec, K_vec)],
        "p": p_vec,
        "overlap": overlaps_serialized,
    })

    fdrs = multipletests(results_df["p"].to_numpy(), method="fdr_bh")[1] 
    print(fdrs)
    results_df["fdr"] = fdrs

    results_df.sort_values(["p", "fdr"], inplace=True, ignore_index=True)
    # rearrange columns so that overlap is last
    cols = list(results_df.columns)
    cols.remove("overlap")
    cols.append("overlap")
    results_df = results_df[cols]

    matrix_df = pd.DataFrame(mat, index=genes, columns=col_names) if build_matrix else None
    matrix_df = matrix_df.transpose() 
    return results_df, matrix_df



# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def df_to_tsv_str(df: pd.DataFrame) -> str:
    """Return a TSV representation (string) of a DataFrame."""
    buf = StringIO()
    df.to_csv(buf, sep='\t', index=not df.index.equals(pd.RangeIndex(len(df))))
    return buf.getvalue()

def df_to_csv_str(df: pd.DataFrame) -> str:
    """Return a CSV representation (string) of a DataFrame."""
    buf = StringIO()
    df.to_csv(buf, sep=',', index=not df.index.equals(pd.RangeIndex(len(df))))
    return buf.getvalue()

def save_compare_outputs(
    results_df: pd.DataFrame,
    matrix_df: Optional[pd.DataFrame] = None,
    out_path: Optional[Union[str, Path]] = None,
    matrix_path: Optional[Union[str, Path]] = None,
    sep: str = '\t',
) -> Tuple[Optional[Path], Optional[Path]]:
    out_written = None
    mat_written = None
    if out_path is not None:
        out_p = Path(out_path)
        results_df.to_csv(out_p, sep=sep, index=False)
        out_written = out_p
    if matrix_df is not None and matrix_path is not None:
        mat_p = Path(matrix_path)
        matrix_df.to_csv(mat_p, sep=sep)
        mat_written = mat_p
    return out_written, mat_written


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> int:
    ap = argparse.ArgumentParser(description="Compare a gene list to a gene-set database.")
    ap.add_argument("--genes", type=str, help="Gene list file (txt, newline/comma-delimited) or comma string.")
    ap.add_argument("--fitdb", type=str, help="Gene-set DB file (.tsv or .json).", required=False)
    ap.add_argument("--min-overlap", type=int, default=1, help="Minimum overlap to report.")
    ap.add_argument("--out", type=str, help="Write results table (.tsv).  Default: inferred.", default=None)
    ap.add_argument("--matrix-out", type=str, help="Write matrix table (.tsv).  Default: inferred.", default=None)
    ap.add_argument("--sep", type=str, default='\t', help="Field separator for output files (default: TAB). Use ',' for CSV.")
    ap.add_argument("--no-print", action="store_true", help="Suppress pretty-print of results to stdout.")
    ap.add_argument("--heatmap-all", action="store_true", help="Use all gene sets (ignore FDR filter) when plotting heatmap.")
    ap.add_argument("--no-matrix", action="store_true",
               help="Skip building membership matrix (faster, less memory).")
    ap.add_argument("--input-bg", type=str,
                    help="Input background gene list (file path or comma‑separated string).")
    ap.add_argument("--db-bg", type=str,
                    help="Database background gene list (file path or comma‑separated string).")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Genes
    # ------------------------------------------------------------------
    genes_file: Optional[Path] = None
    raw: str = ""

    if args.genes:
        candidate = Path(args.genes)
        try:
            if candidate.expanduser().exists():
                # print()
                genes_file = candidate
                raw = candidate.read_text()
            else:
                raise FileNotFoundError
        except (OSError, FileNotFoundError):
            # Treat the argument literally as a comma / whitespace list
            raw = args.genes

    genes = [g for g in parse_gene_input(raw) if g]  # remove empties

    # FitDB
    if args.fitdb:
        p = Path(args.fitdb)
        if not p.exists():
            ap.error(f"Gene-set DB file not found: {p}")
        fitdb = load_fitdb_json(p)

    # Optional background universes
    def _load_bg(raw_arg):
        """
        Load an optional background definition.

        ‑ If the argument is a path to a TSV/CSV with ≥2 columns (index = genes,
          columns = set names containing 0/1 flags), return a
          **dict[str, set[str]]** mapping each column to the gene universe
          where the value != 0.
        ‑ Otherwise, treat it as a plain gene list (file or comma/whitespace
          string) and return a **list[str]**.
        """
        if raw_arg is None:
            return None

        p = Path(raw_arg)
        if p.exists():
            # try matrix first
            try:
                import pandas as _pd
                df = _pd.read_csv(
                    p,
                    sep='\t' if p.suffix.lower() != ".csv" else ',',
                    index_col=0
                )
                if df.shape[1] >= 2 and df.select_dtypes(include=["number"]).shape[1] == df.shape[1]:
                    return {col: set(df.index[df[col] != 0]) for col in df.columns}
            except Exception:
                pass  # fall‑through to plain list

            # plain gene list file
            return parse_gene_input(p.read_text())

        # raw_arg not a file – treat as gene list string
        return parse_gene_input(raw_arg)
    input_bg = _load_bg(args.input_bg)
    db_bg    = _load_bg(args.db_bg)

    # After loading fitdb_
    fitdb = normalize_fitdb(fitdb)

    res_df, mat_df = compare_gene_sets_fast(
        genes,
        fitdb,
        min_overlap=args.min_overlap,
        build_matrix=not args.no_matrix,
        background_input=input_bg,
        background_db=db_bg,
    )
    # res_df = res_df[res_df["fdr"] < 0.05]  # filter by FDR < 0.05

    # Determine default output paths if not specified
    if args.out is None and genes_file is not None:
        out_path = genes_file.with_suffix(".enrich.tsv" if args.sep == '\t' else ".enrich.csv")
    elif args.out is None:
        out_path = Path("results.tsv" if args.sep == '\t' else "results.csv")
    else:
        out_path = Path(args.out)

    if args.matrix_out is None and genes_file is not None:
        matrix_path = genes_file.with_suffix(".matrix.tsv" if args.sep == '\t' else ".matrix.csv")
    elif args.matrix_out is None:
        matrix_path = Path("matrix.tsv" if args.sep == '\t' else "matrix.csv")
    else:
        matrix_path = Path(args.matrix_out)

    out_written, mat_written = save_compare_outputs(
        res_df, mat_df, out_path=out_path, matrix_path=matrix_path, sep=args.sep
    )

    if not args.no_print:
        pd.set_option("display.max_rows", None)
        print("# Results:")
        print(res_df.to_string(index=False))
        print()
        if mat_df is not None:
            print("# Matrix (genes x sets):")
            print(mat_df.to_string())
        print()
        print(f"[Saved results -> {out_written}]\n[Saved matrix  -> {mat_written}]")
    else:
        # Minimal notice when suppressed
        print(f"{len(res_df)} sets written to {out_written}; matrix -> {mat_written}; ")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
