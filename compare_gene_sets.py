#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union, Optional

try:
    import numpy as np  # noqa: F401
except ImportError as _e:
    raise SystemExit("numpy is required: %s" % _e)

try:
    import pandas as pd
except ImportError as _e:
    raise SystemExit("pandas is required: %s" % _e)

# SciPy hypergeom (preferred)
_HAVE_SCIPY = True
try:
    from scipy.stats import hypergeom
except Exception:
    _HAVE_SCIPY = False

# statsmodels for FDR (preferred)
_HAVE_STATSMODELS = True
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    _HAVE_STATSMODELS = False


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

TOKEN_RE = re.compile(r"[\s,;]+")


def parse_gene_input(raw: Union[str, Sequence[str]]) -> List[str]:
    """Parse user gene input into an *order-preserving*, duplicate-free list."""
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in TOKEN_RE.split(raw) if tok.strip()]
    else:
        tokens = [str(tok).strip() for tok in raw if str(tok).strip()]
    return list(dict.fromkeys(tokens))  # order-preserving dedupe


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
# Benjamini-Hochberg FDR fallback
# ------------------------------------------------------------------

def _bh_fdr(pvals: Sequence[float]):
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda kv: kv[1])
    adj = [0.0] * m
    prev = 1.0
    for rank, (idx, pv) in enumerate(indexed, start=1):
        val = pv * m / rank
        if val > prev:
            val = prev
        prev = val
        adj[idx] = min(val, 1.0)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    return adj


# ------------------------------------------------------------------
# Core enrichment
# ------------------------------------------------------------------
def compare_gene_sets_fast(
    my_genes: Union[str, Sequence[str]],
    fitdb: Mapping[str, Set[str]],
    *,
    min_overlap: int = 1,
    build_matrix: bool = True,
):
    """
    Vectorized enrichment against a fitdb whose values are *sets*.
    Much faster for large DBs than the Python-loop version.
    """
    genes = parse_gene_input(my_genes)
    if not genes:
        raise ValueError("Input gene list is empty.")
    myset = set(genes)
    n = len(myset)

    # rank map for stable overlap ordering
    gene_rank = {g: i for i, g in enumerate(genes)}

    # Universe
    # (fitdb already normalized to sets)
    background = set().union(*fitdb.values())
    N = len(background)
    if N == 0:
        raise ValueError("Gene-set database is empty.")

    # Pre-allocate results
    names = []
    K_vec = []
    k_vec = []
    p_vec = []
    overlaps_serialized = []

    # (optional) matrix
    if build_matrix:
        import numpy as _np
        mat = _np.zeros((len(genes), len(fitdb)), dtype=_np.uint8)
        col_names = list(fitdb.keys())

    # iterate sets
    for j, (set_name, members) in enumerate(fitdb.items()):
        K = len(members)
        if K == 0:
            continue

        ov = myset & members
        k = len(ov)

        if build_matrix:
            idx = [gene_rank[g] for g in members if g in gene_rank]
            if idx:
                mat[_np.asarray(idx, dtype=int), j] = 1

        if k < min_overlap:
            continue

        # hypergeom tail
        if _HAVE_SCIPY:
            p = float(hypergeom.sf(k - 1, N, K, n))
        else:
            from math import comb
            tail = 0.0
            max_x = min(K, n)
            for x in range(k, max_x + 1):
                tail += comb(K, x) * comb(N - K, n - x)
            p = tail / comb(N, n)

        names.append(set_name)
        K_vec.append(K)
        k_vec.append(k)
        p_vec.append(p)
        overlaps_serialized.append(",".join(sorted(ov, key=lambda g: gene_rank[g])))

    # Build results DF
    results_df = pd.DataFrame({
        "name": names,
        "K": K_vec,
        "k": k_vec,
        "ratio": [k / K for k, K in zip(k_vec, K_vec)],
        "p": p_vec,
        "overlap": overlaps_serialized,
    })

    # BH FDR
    if _HAVE_STATSMODELS:
        _, fdrs, _, _ = multipletests(results_df["p"].to_numpy(), method="fdr_bh")
    else:
        fdrs = _bh_fdr(results_df["p"].to_list())
    results_df["fdr"] = fdrs

    results_df.sort_values(["p", "fdr"], inplace=True, ignore_index=True)

    # Matrix DF
    matrix_df = None
    if build_matrix:
        matrix_df = pd.DataFrame(mat, index=genes, columns=col_names)

    return results_df, matrix_df

def compare_gene_sets(
    my_genes: Union[str, Sequence[str]],
    fitdb,
    min_overlap: int = 1,
    return_matrix: bool = True,
):
    """Compare an input gene list against a gene-set database."""
    genes = parse_gene_input(my_genes)
    myset = set(genes)
    n = len(myset)
    if n == 0:
        raise ValueError("Input gene list is empty.")

    # Background universe (computed once)
    background = set().union(*(set(v) if isinstance(v, set) else set(parse_gene_input(v)) for v in fitdb.values()))
    N = len(background)
    if N == 0:
        raise ValueError("Gene-set database is empty.")

    matrix_cols = []
    matrix_rows = genes
    matrix_data = [[0] * 0 for _ in matrix_rows] if return_matrix else None

    records = []
    pvals = []

    # iterate sets
    for set_name, members_any in fitdb.items():
        members = members_any if isinstance(members_any, set) else set(parse_gene_input(members_any))
        K = len(members)
        if K == 0:
            continue

        overlap = myset & members
        k = len(overlap)

        if return_matrix:
            matrix_cols.append(set_name)
            for row_i, g in enumerate(matrix_rows):
                matrix_data[row_i].append(1 if g in members else 0)

        if k < min_overlap:
            continue

        if _HAVE_SCIPY:
            p = float(hypergeom.sf(k - 1, N, K, n))
        else:
            from math import comb
            tail = 0.0
            max_x = min(K, n)
            for x in range(k, max_x + 1):
                tail += comb(K, x) * comb(N - K, n - x)
            p = tail / comb(N, n)

        records.append({
            "name": set_name,
            "K": K,
            "k": k,
            "ratio": k / K if K else 0.0,
            "p": p,
            "overlap": ",".join(sorted(overlap, key=lambda g: genes.index(g) if g in myset else g)),
        })
        pvals.append(p)

    if not records:
        results_df = pd.DataFrame(columns=["name", "K", "k", "ratio", "p", "fdr", "overlap"])
        matrix_df = pd.DataFrame(matrix_data, index=matrix_rows, columns=matrix_cols) if return_matrix else None
        return results_df, matrix_df

    if _HAVE_STATSMODELS:
        _, fdrs, _, _ = multipletests(pvals, method="fdr_bh")
    else:
        fdrs = _bh_fdr(pvals)

    for rec, fdr in zip(records, fdrs):
        rec["fdr"] = float(fdr)

    results_df = pd.DataFrame.from_records(records)
    results_df.sort_values(["p", "fdr"], inplace=True, ignore_index=True)

    matrix_df = pd.DataFrame(matrix_data, index=matrix_rows, columns=matrix_cols) if return_matrix else None

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


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    cols: Optional[List[str]] = None,
    figsize_scale: float = 0.25,
    dpi: int = 150,
) -> Path:
    """Plot a white/blue presence-absence heatmap.

    Parameters
    ----------
    matrix_df
        DataFrame of 0/1 membership (genes x sets).
    out_path
        Where to write the PNG.
    cols
        Optional ordered list of columns to display (subset/reorder). Default: all columns.
    light_blue
        Hex color for 1's (default: HTML 'lightblue' #add8e6).
    figsize_scale
        Inches per column/row; figure size = (max(3, n_cols*scale), max(3, n_rows*scale)).
    dpi
        Figure DPI.

    Returns
    -------
    Path to the written PNG (same as input).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless safe
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"matplotlib required for heatmap plotting: {e}")

    if cols is None:
        data = matrix_df.values
        col_labels = matrix_df.columns.to_list()
    else:
        col_labels = [c for c in cols if c in matrix_df.columns]
        data = matrix_df[col_labels].values

    row_labels = matrix_df.index.to_list()

    # colormap: 0 -> white, 1 -> light_blue
    cmap = ListedColormap(["#ffffff", "#0025AC"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    n_rows, n_cols = data.shape
    width = max(3.0, n_cols * figsize_scale)
    height = max(3.0, n_rows * figsize_scale)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # ticks & labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # light gridlines
    ax.set_xticks([x-0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y-0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.5)
    ax.tick_params(which="both", length=0)

    ax.set_xlabel("Gene Sets")
    ax.set_ylabel("Genes")
    ax.set_title("Gene Membership Matrix")

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

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
    ap.add_argument("--heatmap", type=str, default=None, help="Write heatmap PNG for the membership matrix.")
    ap.add_argument("--heatmap-all", action="store_true", help="Use all gene sets (ignore FDR filter) when plotting heatmap.")
    ap.add_argument("--no-matrix", action="store_true",
               help="Skip building membership matrix (faster, less memory).")
    args = ap.parse_args()

    # Genes
    genes_file = None
    if args.genes and Path(args.genes).exists():
        genes_file = Path(args.genes)
        raw = genes_file.read_text()
    else:
        raw = args.genes or ""
    genes = parse_gene_input(raw)
    # remove trailing and leading whitespace and quotes
    genes = [g.strip().strip('"').strip("'") for g in genes if g.strip()]

    # FitDB
    if args.fitdb:
        p = Path(args.fitdb)
        if not p.exists():
            ap.error(f"Gene-set DB file not found: {p}")
        if p.suffix.lower() == ".json":
            fitdb = load_fitdb_json(p)
        else:
            fitdb = load_fitdb_tsv(p)

    # After loading fitdb_
    fitdb = normalize_fitdb(fitdb)

    res_df, mat_df = compare_gene_sets_fast(
        genes, fitdb,
        min_overlap=args.min_overlap,
        build_matrix=not args.no_matrix,  # add CLI flag
    )
    res_df = res_df[res_df["fdr"] < 0.05]  # filter by FDR < 0.05

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

    
    # Heatmap path
    if args.heatmap is not None:
        heatmap_path = Path(args.heatmap)
    else:
        # derive from matrix_path
        heatmap_path = matrix_path.with_suffix(".png")
    
    # Determine columns to plot
    if not args.heatmap_all and not res_df.empty:
        heatmap_cols = res_df["name"].tolist()  # canonical
    else:
        heatmap_cols = None
    
    try:
        if mat_df is not None:
            plot_matrix_heatmap(mat_df, heatmap_path, cols=heatmap_cols)
            heatmap_written = heatmap_path
        else:
            heatmap_written = None
    except Exception as e:
        print(f"[WARN] Heatmap failed: {e}")
        heatmap_written = None
    

    if not args.no_print:
        pd.set_option("display.max_rows", None)
        print("# Results:")
        print(res_df.to_string(index=False))
        print()
        if mat_df is not None:
            print("# Matrix (genes x sets):")
            print(mat_df.to_string())
        print()
        print(f"[Saved results -> {out_written}]\n[Saved matrix  -> {mat_written}]\n[Saved heatmap -> {heatmap_written}]")
    else:
        # Minimal notice when suppressed
        print(f"{len(res_df)} sets written to {out_written}; matrix -> {mat_written}; heatmap -> {heatmap_written}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
