import pandas as pd
import os
import numpy as np

#  calculate a CC's truth value based on the labels of the truth value of its ANT, CON and causality.
def truthcc(df, tv_ant, tv_con, causality, tv_cc):
    # Compute the truth value of a CC based on the equation.
    for i in range(0, len(df)):
        if df.iloc[i][causality] == -1 or df.iloc[i][tv_ant] == 1 or df.iloc[i][tv_con] ==1:
             df.loc[i, tv_cc] = '-1'
             df.loc[i, tv_cc + "_MisInf"] = 'MisInf'
        elif df.iloc[i][causality] == 0 and df.iloc[i][tv_ant] !=1 and df.iloc[i][tv_con] != 1:
             df.loc[i, tv_cc] = '0'
             df.loc[i, tv_cc + "_MisInf"] = 'NA'
        elif df.iloc[i][causality] == 1 and df.iloc[i][tv_ant] !=1 and df.iloc[i][tv_con] != 1:
             df.loc[i, tv_cc] = '1'
             df.loc[i, tv_cc + "_MisInf"] = 'NA'
        else:
             df.loc[i, tv_cc] = 999
             df.loc[i, tv_cc + "_MisInf"] = 'NA'

    return df

# a sub-claim is seen as true only if it has been classifed as verifiable and true
def vf_truthcc(df, vf_ant, tv_ant, vf_con, tv_con, causality, tv_cc):
    # Compute the truth value of a CC based the equation
    for i in range(0, len(df)):
        if df.iloc[i][vf_ant] == 1 and df.iloc[i][tv_ant] == 1:
            denial = 1
        elif df.iloc[i][vf_con] == 1 and df.iloc[i][tv_con] ==1:
            denial = 1
        else:
            denial = 0
        if  denial ==1 or df.iloc[i][causality] == -1:
             df.loc[i, tv_cc] = '-1'
             df.loc[i, tv_cc + "_MisInf"] = 'MisInf'
        elif df.iloc[i][causality] == 0 and df.iloc[i][tv_ant] !=1 and df.iloc[i][tv_con] != 1:
             df.loc[i, tv_cc] = '0'
             df.loc[i, tv_cc + "_MisInf"] = 'NA'
        elif df.iloc[i][causality] == 1 and df.iloc[i][tv_ant] !=1 and df.iloc[i][tv_con] != 1:
             df.loc[i, tv_cc] = '1'
             df.loc[i, tv_cc + "_MisInf"] = 'NA'
        else:
             df.loc[i, tv_cc] = 999
             df.loc[i, tv_cc + "_MisInf"] = 'NA'

    return df

# check and write whether the prediction of a model is consistent w.r.t. each claim
def consistency(df, model):
    mc = 'Causality'
    new_column = [''] * len(df)
    for i in range(0, len(df)):
        new_column[i] = ''
        if int(df.iloc[i]['_'.join([model, 'TV_Ant'])]) == 1 and int(df.iloc[i]['_'.join([model, mc])]) ==1 and int(df.iloc[i]['_'.join([model, 'TV_Con'])]) != 1 :
            new_column[i] = model+'_Ant_Con_'+mc
        elif int(df.iloc[i]['_'.join([model, 'Vf_Ant'])]) == -1 and int(df.iloc[i]['_'.join([model, 'TV_Ant'])]) != 0:
            new_column[i] = model + "_Vf_TV_Ant"
        elif int(df.iloc[i]['_'.join([model, 'Vf_Con'])])== -1 and int(df.iloc[i]['_'.join([model, 'TV_Con'])]) != 0:
            new_column[i] = model + '_Vf_TV_Con'
    df[model + '_inconsistency'] = new_column

    return df

if __name__ == '__main__':
    input_File = './data/miscc_baseline.csv'
    df = pd.read_csv(input_File)
    model = 'Llama3_'
    df_new = truthcc(df, model+ 'TV_Ant', model+ 'TV_Con', model+ 'Causality', model+ 'TV-cc')
    df_new.to_csv('_'.join([input_File[:-4], model, 'TV', '.csv']), index=False)


