import os
import pandas as pd
import numpy as np
import glob
import random

def top_gene_selection(path, top_n=1000):
    genes=glob.glob(path+'/*/*.tsv')
    # genes=random.sample(genes,k=500)
    top_genes_list = []
    for gene in genes:
        if 'Coord' in gene:
            continue

        print(gene)

        df = pd.read_csv(gene, delimiter='\t', index_col=0, header=0)
        # calculate the variance of each row, to find the top 1000 genes across all spots
        col_var = df.var(axis=0)
        # print(row_var)

        # find top 1000, which has large variance
        df_sort = col_var.sort_values(ascending=False)
        top_1000 = df_sort.head(int(1.5*top_n))
        top_1000_index = list(top_1000.index)
        # print('index')
        # print(top_1000_index)
        top_genes_list.append(top_1000_index)

    result_gene = set(top_genes_list[0])
    for i, top in enumerate(top_genes_list):
        result_gene = result_gene.intersection(set(top))

    random.seed(1553)
    result_gene=random.sample(result_gene,250)
    random.seed()
    np.save('HBC_Result_Gene.npy',np.array(result_gene))
    print(len(result_gene))
    return result_gene

top_gene_selection('/your/path/to/')
