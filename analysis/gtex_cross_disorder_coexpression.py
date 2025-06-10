
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def cross_disorder_coexpression(expr_file, genes_csv, gene_symbols, output_prefix, top_n=None):
    """
    Calculate Pearson correlation using GTEx expression and mapped gene symbols.

    Args:
        expr_file (str): Path to GTEx BED expression file.
        genes_csv (str): CSV mapping Ensembl_ID to Gene_Symbol.
        gene_symbols (list[str]): Gene symbols to analyze.
        output_prefix (str): Output prefix for result files.
        top_n (int, optional): Select top N correlated genes. Default None (all genes).
    """

    # Load expression data (BED file)
    expr_df = pd.read_csv(expr_file,  sep="\t")
    expr_df['gene_id'] = expr_df['gene_id'].str.split('.').str[0]
    expr_df.set_index('gene_id', inplace=True)
    expr_data = expr_df.iloc[:, 3:]  # skip chr, start, end

    # Load gene mapping (genes_output.csv)
    mapping_df = pd.read_csv(genes_csv)
    mapping_df.columns = ['Ensembl_ID', 'Gene_Symbol']
    mapping_df['Ensembl_ID'] = mapping_df['Ensembl_ID'].str.split('.').str[0]

    # Map gene symbols to Ensembl IDs
    gene_mapping = mapping_df.set_index('Gene_Symbol')['Ensembl_ID'].to_dict()

    # Get Ensembl IDs for provided gene symbols
    ensembl_ids = []
    missing_genes = []
    for gene in gene_symbols:
        if gene in gene_mapping:
            ensembl_ids.append(gene_mapping[gene])
        else:
            missing_genes.append(gene)

    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found: {missing_genes}")

    # Filter expression data
    genes_found = expr_data.index.intersection(ensembl_ids).tolist()

    if not genes_found:
        print("Error: None of the specified genes were found in the dataset.")
        sys.exit(1)

    expr_filtered = expr_data.loc[genes_found]

    # Replace Ensembl IDs with gene symbols for readability
    inverse_mapping = {v: k for k, v in gene_mapping.items()}
    expr_filtered.index = [inverse_mapping.get(ensg, ensg) for ensg in expr_filtered.index]

    # Transpose dataframe (genes as columns)
    expr_filtered_T = expr_filtered.T

    # Pearson correlation
    corr_matrix = expr_filtered_T.corr(method='pearson')

    # Select top N correlated genes if top_n is specified
    if top_n is not None and top_n < len(corr_matrix):
        print(f"Selecting top {top_n} correlated genes...")
        # Compute the mean absolute correlation for each gene
        corr_abs_mean = corr_matrix.abs().mean().sort_values(ascending=False)
        top_genes = corr_abs_mean.head(top_n).index
        corr_matrix = corr_matrix.loc[top_genes, top_genes]

    # Save correlation matrix
    corr_filename = f"{output_prefix}_corr_matrix.csv"
    corr_matrix.to_csv(corr_filename)
    print(f"Correlation matrix saved as '{corr_filename}'")

    # Plot heatmap clearly annotated, legend positioned left to cover empty space
    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create annotation matrix explicitly for lower triangle
    annot_matrix = corr_matrix.round(2).astype(float)
    # annot_matrix.values[np.triu_indices_from(annot_matrix, k=1)] = ''

    # Plot heatmap with custom annotations
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        annot=annot_matrix,
        fmt='',
        annot_kws={"size": 8},
        linewidths=.05,
        square=True,
        cbar_kws={"shrink": .6, "location": "right", "pad": 0.02}
    )

    plt.title(f"Gene Co-expression ({output_prefix})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved as '{output_prefix}_heatmap.png'")


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage:")
        print(f"{sys.argv[0]} <GTEx_BED_file> <genes_output.csv> <gene_symbols_comma_separated> <output_prefix> [top_n (optional)]")
        sys.exit(1)

    expr_file = sys.argv[1]
    genes_csv = sys.argv[2]
    gene_symbols = sys.argv[3].split(',')
    output_prefix = sys.argv[4]
    top_n = int(sys.argv[5]) if len(sys.argv) == 6 else None

    cross_disorder_coexpression(expr_file, genes_csv, gene_symbols, output_prefix, top_n)
