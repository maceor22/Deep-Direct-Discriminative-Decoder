import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rnalysis.filtering import CountFilter, DESeqFilter



if __name__ == '__main__':

    res_path = 'LimmaVoom_condition_defeated_vs_control.csv'

    if os.path.exists(res_path):

        dea = DESeqFilter(res_path, log2fc_col = 'logFC', padj_col = 'adj.P.Val')
    
    else:

        out_path = '/home/maceor22/Thesis/Deep-Direct-Discriminative-Decoder/BB 581/'

        data = CountFilter('reads_CPM.csv', is_normalized = True)

        dea = data.differential_expression_limma_voom(
            design_matrix = 'design_matrix.csv',
            comparisons = (('condition', 'defeated', 'control'),),
            r_installation_folder = 'auto',
            output_folder = out_path,
        )[0]

        print(dea)


    counts = np.log2(pd.read_csv('reads_CPM.csv', index_col = 0).values.T)
    design = pd.read_csv('design_matrix.csv', index_col = 0)

    pca = PCA()
    all_samples = pca.fit_transform(counts)

    def_samples = all_samples[np.where(design['condition'] == 'defeated')]
    con_samples = all_samples[np.where(design['condition'] == 'control')]

    print(counts.shape, def_samples.shape, con_samples.shape)

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    fig.suptitle('Principle Component Analysis of Samples')

    ax[0,0].plot(def_samples[:,0], def_samples[:,1], 'o', color = 'orange', ms = 6, label = 'defeated')
    ax[0,0].plot(con_samples[:,0], con_samples[:,1], 'o', color = 'blue', ms = 6, label = 'control')
    ax[0,0].set_xlabel('PC1')
    ax[0,0].set_ylabel('PC2')
    ax[0,0].legend(fontsize = 6)

    ax[1,0].plot(def_samples[:,1], def_samples[:,2], 'o', color = 'orange', ms = 6, label = 'defeated')
    ax[1,0].plot(con_samples[:,1], con_samples[:,2], 'o', color = 'blue', ms = 6, label = 'control')
    ax[1,0].set_xlabel('PC2')
    ax[1,0].set_ylabel('PC3')
    ax[1,0].legend(fontsize = 6)

    ax[0,1].plot(
        np.arange(1,pca.n_components_+1), 
        pca.explained_variance_ratio_.cumsum(0),
        color = 'k', lw = 2,
        )
    ax[0,1].set_xlabel('Number of PCs')
    ax[0,1].set_ylabel('Explained Variance Ratio')
    ax[0,1].yaxis.tick_right()
    ax[0,1].yaxis.set_label_position('right')

    for n in range(3):
        ax[1,1] = sns.kdeplot(all_samples[n], label = f'PC{n+1}')
    ax[1,1].set_xlabel('Principle Components')
    ax[1,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position('right')
    plt.legend()


    alpha = 0.05
    log2fc_threshold = 1

    fig = dea.volcano_plot(
        alpha = alpha, log2fc_threshold = log2fc_threshold, 
        title = f'Differential Expression Volcano Plot | alpha: {alpha}, logFC: {log2fc_threshold}', title_fontsize = 14,
        label_fontsize = 12, tick_fontsize = 8, annotation_fontsize = 10,
        point_size = 16, opacity = 0.65, 
    )
    
    inq = pd.read_csv('LimmaVoom_condition_defeated_vs_control.csv', index_col = 0)
    inq = inq[['logFC', 'adj.P.Val']]
    inq['adj.P.Val'] = -np.log10(inq['adj.P.Val'])
    
    print(inq.loc['ENSMUSG00000034997'])
    print(inq.loc['ENSMUSG00000029868'])

    # HTR2A: ENSMUSG00000034997
    # TRPV6: ENSMUSG00000029868


    plt.show()

















