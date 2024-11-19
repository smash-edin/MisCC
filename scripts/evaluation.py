# Cross Validation Classification LogLoss
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
import json
from datetime import date
import pdb
from scipy.stats import pearsonr


def jsonLog(data):
    today = str(date.today())
    file = './data/inconsistencies'  + today + '.json'

    with open(file, mode='a+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent = 4)
        f.close()
    return


def eva(inp_file, columnpairs, outreport_File):
    # pdb.set_trace()
    df1 = pd.read_csv(inp_file)
    dataframe= df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
    print(dataframe.keys(), '\n')

    report_flag = ''
    if os.path.isfile(outreport_File):
        df2 = pd.read_csv(outreport_File)
        report_flag = 'exist'
    invalid = ['nan', 'na', '', np.NaN]

    for (GS, Prediction, label_set, task, model) in columnpairs:
        y_true = []
        y_pred = []
        category = '-----'.join([GS, Prediction, str(label_set), task])
        print(category)
        if task == '':
            y_true = dataframe[GS].tolist()
            y_pred = dataframe[Prediction].tolist()
        # sub-claim-together: put antcedent and consequent together as the set of data
        elif task == 'sub-claim-together':
            for sub_claim in ['_Ant', '_Con']:
                print('\n--------------append sub-claims of ', sub_claim)
                for i in range(0, len(dataframe)):
                    y_true.append(int(float(dataframe.iloc[i][GS + sub_claim])))
                    y_pred.append(int(float(dataframe.iloc[i][Prediction + sub_claim])))
        elif task == 'verifiable_only':
            vef_column = 'Vf_' + GS.split('_')[-1]
            print('vefiability column is ', vef_column)
            for i in range(0, len(dataframe)):
                if int(float(dataframe.iloc[i][vef_column])) == 1:
                    y_true.append(int(float(dataframe.iloc[i][GS])))
                    y_pred.append(int(float(dataframe.iloc[i][Prediction])))
        elif task == 'sub-claim-together and consistent only':
            for sub_claim in ['_Ant', '_Con']:
                print('\nappend sub-claims of ', sub_claim)
                for i in range (0, len(dataframe)):
                    if dataframe.iloc[i][model + '_inconsistency'] in invalid:
                        y_true.append(int(float(dataframe.iloc[i][GS + sub_claim])))
                        y_pred.append(int(float(dataframe.iloc[i][Prediction + sub_claim])))
        elif task == 'sub-claim-together and consistent only':
            for sub_claim in ['_Ant', '_Con']:
                print('\nappend sub-claims of ', sub_claim)
                for i in range(0, len(dataframe)):
                    if int(float(dataframe.iloc[i]['Vf' + sub_claim])) == 1:
                        y_true.append(int(float(dataframe.iloc[i][GS + sub_claim])))
                        y_pred.append(int(float(dataframe.iloc[i][Prediction + sub_claim])))
        elif task == 'sub-claim-together and verifiable consistent only':
            for sub_claim in ['_Ant', '_Con']:
                print('\nappend sub-claims of ', sub_claim)
                for i in range(0, len(dataframe)):
                    if int(float(dataframe.iloc[i]['Vf' + sub_claim])) == 1 and dataframe.iloc[i][model + '_inconsistency'] in invalid:
                        y_true.append(int(float(dataframe.iloc[i][GS + sub_claim])))
                        y_pred.append(int(float(dataframe.iloc[i][Prediction + sub_claim])))
        elif task == 'nonVF_0':
            for i in range(0, len(dataframe)):
                if str(dataframe.iloc[i][GS]) == '-2':
                    y_true.append(int(0))
                    y_pred.append(int(dataframe.iloc[i][Prediction]))
                else:
                    y_true.append(dataframe.iloc[i][GS])
                    y_pred.append(int(dataframe.iloc[i][Prediction]))

        elif task == 'consistent_only':
            for i in range (0, len(dataframe)):
                if dataframe.iloc[i][model + '_inconsistency'] in invalid:
                    y_true.append(dataframe.iloc[i][GS])
                    y_pred.append(dataframe.iloc[i][Prediction])
        elif task == 'mis_consistent_only':
            for i in range (0, len(dataframe)):
                if 'vf' not in str(dataframe.iloc[i][model + '_inconsistency']).lower():
                    y_true.append(dataframe.iloc[i][GS])
                    y_pred.append(dataframe.iloc[i][Prediction])
        elif task =='subset_classification_only':
            for i in range(0, len(dataframe)):
                if dataframe.iloc[i]['Reformed'] == 'N':
                    y_true.append(dataframe.iloc[i][GS])
                    y_pred.append(dataframe.iloc[i][Prediction])
        else:
            print('\n\nError: unknown evaluation task.\n')
        print('\n\nThe lengths of y_true and y_pred are ', len(y_true), len(y_pred))
        print('\nThe values of y_true and y_pred are ', set(y_true), set(y_pred))


        report_dict = pd.DataFrame(classification_report(y_true, y_pred, labels= label_set, output_dict=True))
        report_dict['Prediction'] = np.nan
        report_dict.loc[len(report_dict), 'Prediction'] = category
        print(report_dict)

        if report_flag == 'exist':
            df2 = pd.concat([df2, report_dict])
        else:
            df2 = report_dict
            report_flag = 'exist'
        df2.to_csv(outreport_File,  index=False)
    return


def F_Score():
    input_File = './miscc_baseline_rep.csv'
    report = ''
    # input_File = '/Users/xueli/Library/CloudStorage/OneDrive-UniversityofEdinburgh/code/miscc_1/2024_data/miscc_data/baseline/semval_classification_2.csv'

    predict_columns =  ['env_tv_myllama3_'+str(i) for i in range(13)]
    eva2(input_File, 'Causality', predict_columns, [-1, 0, 1], report)
    # lacking of 'Llama3_OnePromp1',
    eva2(input_File, 'TV-CC', ['GPT4_OnePrompt', 'Llama3_OnePrompt',  'Llama3_TV-cc', 'GPT4_TV-cc', 'Llama3_Al3', 'Llama3_Al3_new', 'GPT4_Al3', 'Llama3_Al2_new',], [-1, 0, 1], 'report_eva_all2806_ternary.csv')
    # evaluate two classes of misinf and na.
    # 'GPT_algorithm', 'Llama3_Algorithm_omitVerifiable', 'Llama3_Algorithm', 'Llama3_tv_PP2', 'Llama3_tv_PP1'

    eva(df,     [('TV_Ant', 'Llama3_TV_Ant'), ('TV_Ant', 'GPT4_TV_Ant'), ('TV_Con', 'Llama3_TV_Con'), ('TV_Con', 'GPT4_TV_Con')], [-1, 0, 1], 'report_eva_ternary.csv', 'all')

    eva_consis(input_File, 'TV-CC', ['Llama3_OnePrompt',  'Llama3_TV-cc', 'Llama3_Al3', 'Llama3_Al3_new', 'Llama3_Al2_new', 'GPT4_OnePrompt', 'GPT4_Al3', 'GPT4_TV-cc'], [-1, 0, 1], 'report_eva_all3006_ternary_consistent.csv')
    # [('TV-CC_MisInf', 'GPT4_PP1'),

def pvalue(input_File, logfile, predict, gs, DV, IVs):
    get_rid(input_File)
    df_temp = pd.read_csv(input_File)
    df = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
    df.to_csv(input_File)
    statis_info = []

    for V in IVs:
        DV_value = []
        IV_value = []
        for i in range(0, len(df)):
            if df.iloc[i][predict] != df.iloc[i][gs]:
                DV_value.append(df.iloc[i][DV])
                IV_value.append(df.iloc[i][V])
        corr, vpalue = pearsonr(IV_value, DV_value)
        statis_info.append( {'file': input_File,
                       'variable1': DV,
                       'variable2': V,
                       'correlation': corr,
                       'pvalue': vpalue})
    with open(logfile, mode='a+', encoding='utf-8') as f:
        json.dump(statis_info, f, ensure_ascii=False, indent = 4)
        f.close()
    return


if __name__ == '__main__':
    #----------------To modify below  ----------------
    # give the input CSV file and output report file below
    input_File = './data/miscc_baseline.csv'
    report = './data/report.csv'
    # tasks can be found in the eva function, e.g.,'sub-claim-together and consistent only'
    task = ''
    temp = 0
    model = 'GPT4'
    # give each pair of GS and the predicted column as a member of the following list to generate the evaluation report.
    columnpairs = [('Vf_Ant', model + '_Vf_Ant', [-1,0,1], '', model),
                   ('Vf_Ant', model + '_Vf_Ant', [-1, 0, 1], task, model),
                   ('Vf_Con', model + '_Vf_Con', [-1,0,1], '', model),
                   ('Vf_Con', model + '_Vf_Con', [-1, 0, 1], task, model),
                   ('TV_Ant', model + '_TV_Ant', [-1,0,1], '', model),
                   ('TV_Ant', model + '_TV_Ant', [-1, 0, 1], task, model),
                   ('TV_Con', model + '_TV_Con', [-1,0,1], '', model),
                   ('TV_Con', model + '_TV_Con', [-1, 0, 1], task, model),
                    ('Causality', model + '_Causality', [-1, 0, 1], '', model),
                    ('Causality', model + '_Causality', [-1, 0, 1], task, model),
                    ('TV-CC', model + '_TV-cc', [-1, 0, 1], '', model),
                    ('TV-CC', model + '_TV-cc', [-1, 0, 1], task, model),
                    ('TV-CC', model + '_OnePrompt', [-1,0,1], '', model),
                   ('TV-CC', model + '_OnePrompt', [-1, 0, 1], task, model),
                   ('TV-CC', model + '_Al3', [-1,0,1], '', model),
                   ('TV-CC', model + '_Al3', [-1, 0, 1], task, model)]
    #----------------To modify above  ----------------
    eva(input_File, columnpairs, report)
