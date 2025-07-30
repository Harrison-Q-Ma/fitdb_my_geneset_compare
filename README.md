## Required packages
- numpy
- scipy
- pandas
- matplotlib
- statsmodels

## Run parameters
```
python compare_gene_sets.py \
--genes test_gene_set.txt \
--fitdb gene_set_mouse.json \
--matrix-out matrix.tsv \
--no-print \
--input-bg test_input_bg.txt \
--db-bg bg_genes.csv
```

## Update Logs
### July 29, 2025
1. Bug fix: FDR values is now correctly updated per comparison;
2. No longer draws a `.png` file for the matrix;
3. Update to statistical testing: hypergeometric test now correctly take into account of background space when calculating p-values. The `.py` script now takes in 2 extra parameters: `--input-bg` (supplied by user, optional) and `--db-bg` (supplied by FITdb). The **N** in hypergeometric test will be the intersect between reference library and input library if `--input-bg` is provided and just the reference library if `--input-bg` is not provided. 
4. Example inputs uploaded (`bg_genes.csv`, `test_gene_set.txt`, `test_input_bg.txt`). 
5. Output matrix file is now transposed for better displaying formatting. 
6. Removed definitions for functions that were no longer used. 