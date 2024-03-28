import pandas as pd
import numpy as np


def format_design(df, phenotype_expressed, group_name):
    condition = pd.Series(
        data = ['control']*len(df.columns), 
        index = df.columns, name = 'condition',
        )
    
    for i in range(len(condition)):
        if condition.index[i] in phenotype_expressed:
            condition[i] = 'defeated'

    group = pd.Series(
        data = [group_name]*len(df.columns), 
        index = df.columns, name = 'group',
        )
    
    return pd.merge(
        condition, group, how = 'inner', left_index = True, right_index = True)


if __name__ == '__main__':

    df1 = pd.read_csv('GSE109315_vHPC_counts_logCPM.txt', sep = '\t')
    df1 = df1[df1.columns[-11:]]

    df2 = pd.read_csv('GSE109315_mPFC_counts_logCPM.txt', sep = '\t')
    df2 = df2[df2.columns[-14:]]

    mutual = []
    for col in df2.columns:
        if col in df1.columns:
            mutual.append(col)
    df2 = df2.drop(columns = mutual)


    defeated1 = ['D4.3', 'D4.7', 'D5.4', 'D5.6', 'D5.8']
    defeated2 = ['D10.20', 'D10.21', 'D10.5', 'D10.7', 'D10.9', 'D8.14']

    design_df = pd.concat([
        format_design(df1, defeated1, 'vHPC'),
        format_design(df2, defeated2, 'mPFC'),
    ], axis = 0)

    count_df = pd.merge(df1, df2, how = 'inner', left_index = True, right_index = True)


    design_df.to_csv('design_matrix.csv')
    (2**count_df).to_csv('reads_CPM.csv')
    


