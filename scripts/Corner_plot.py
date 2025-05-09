import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
cv_filename="results/cv dataframes/cv_dataframe_1000_RDor_2.csv"
pd.set_option('display.float_format', lambda x: '%.2e' % x)
interest_cols = ['lambda_sparsity', 'entropy', 'TSV', 'lambda_TV', 'cross_val_score']
df = pd.read_csv(cv_filename, index_col=0)
df_interest = df[interest_cols]
regularizers = ['lambda_sparsity', 'entropy', 'TSV', 'lambda_TV']
size=int(len(regularizers))
fig, ax = plt.subplots(size, size, figsize=(3.5*size, 3*size))
for i, regularizer_i in enumerate(regularizers):
    for j, regularizer_j in enumerate(regularizers[:i+1]):
        if i==j:
            # Make a new df that for each regularizer, 
            # contains the unique values and the sum of the cross_val_score for that value
            df_unique = df_interest.groupby(regularizer_i).sum()
            X = df_unique.index
            Y = df_unique['cross_val_score']
            ax[i, j].plot(X, Y)
            ax[i, j].set_xscale('log')
        else:
            X = df_interest[regularizer_j]
            Y = df_interest[regularizer_i]
            Z = df_interest['cross_val_score']
            ax[i, j].tricontourf(X, Y, Z, levels=14, cmap='RdYlBu')
            ax[i, j].set_xscale('log')
            ax[i, j].set_yscale('log')
            fig.colorbar(ax[i, j].tricontourf(X, Y, Z, levels=14), ax=ax[i, j], orientation='vertical')
        if j == 0:
            ax[i, j].set_ylabel(regularizer_i)
        if i == size-1:
            ax[i, j].set_xlabel(regularizer_j)
    for j in range(i+1, size):
        ax[i, j].axis('off')
plt.tight_layout()
plt.savefig('results/cv dataframes/cv_dataframe_1000_RDor_2.pdf', format='pdf', bbox_inches='tight')