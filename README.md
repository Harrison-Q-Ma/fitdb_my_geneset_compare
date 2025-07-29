Required packages:
- numpy
- scipy
- pandas
- matplotlib
- statsmodels

Run parameters:  
```
python compare_gene_sets.py \
--genes test_gene_set.txt \
--fitdb /home/qma/fitdb/gene_set/gene_set_mouse.json \
--matrix-out matrix.tsv \
--no-print \
--input-bg /home/qma/fitdb/gene_set/test_input_bg.txt \
--db-bg /home/qma/fitdb/gene_set/bg_genes.csv
```
