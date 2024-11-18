# Import the necessary libraries
import os
import datetime
import pdb
import pandas as pd
from tqdm import tqdm
import warnings
import json
import openai
from openai import OpenAI
import ollama

warnings.filterwarnings('ignore') # setting ignore as a parameter

# prompting template
templates = {
'CC_classification': '''A counterfactual claim (CC) has the following two features: it is a conditional claim written in subjunctive mood. A CC describe hypothetical scenarios. Is the Claim below a counterfactual claim? Answer in JSON format with fields "Answer" and "Explaination". In "Answer", use Y for Yes or N for No. 
Claim: {}'''
'Vf': '''A claim is verifiable if its truth value can be derived or tested to be true or false based on specified knowledge. Is Claim1 verifiable? Claim1 is originally from the CC in an online conversation.
Claim1: {}
CC: {}
PPlease respond with 1 for Yes or -1 for No or 0 for unsure. Do not provide any additional information or explanation. Only respond with -1, 1, 0''',
'TV': '''Claim1 is originally from the Claim2 in an online conversation. Is Claim1 true?
Claim1: {}
Claim2: {}
Please respond with 1 for yes or -1 for No or 0 for unsure. Do not provide any additional information or explanation. Only respond with -1, 1, 0''',
'Causality': '''Does Claim1 cause Claim2? 
    Claim1: {} 
    Claim2: {}
    Please respond with 1 for Yes or -1 for No or 0 for unsure.''',
'OnePrompt':'''Claim 1 is a counterfactual claim from an online conversation. Is Claim1 true? Claim1: "{}".  Please respond with 1 for yes or -1 for No or 0 for unsure. Do not provide any additional information or explanation. Only respond with -1, 1, 0''',
'Al3': '''A counterfactual claim is false if its antecedent is true or its consequent is true or its antecedent does not cause the consequent. In addition, a counterfactual claim is true if both its antecedent and consequent are false and its antecedent causes its consequent. Otherwise, it is unkonwn.
Given the counterfactual claim: {}
Its antecedent is: {}
Its consequent is:{}
Is the counterfactual claims true? Please respond with 1 for true or -1 for false or 0 for unknown'''
}

# query gpt4
def query_gpt(CC, temp, query_prompting, log):
    client = OpenAI(api_key="") # add your api_key in the quotes
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model_name_gpt = 'gpt-4'

    full_answer1 = client.chat.completions.create(
        model=model_name_gpt,  # gpt4, text-davinci-003", #"text-davinci-002", openai.Model.list()
        messages=[
            {
                "role": "user",
                "content": query_prompting
            }],
        temperature=temp,  # column with name GPT4 is of temp 1, column with name GPT4_temp0 is of temp 0,
        max_tokens=128)
    answer = full_answer1.choices[0].message.content
    query_ans = {"CC": CC, "model": "gpt-4", "temprature": temp, "max_tokens": 128, "query": query_prompting,
                 "ans": answer, "time": datetime.datetime.utcnow().isoformat() + "Z"}
    log.write(str(query_ans) + '\n')
    try:
        label = eval(answer)["Answer"]
    except:
        label = ''

    return answer


# query llama3-8B via ollama
def query_llama3(CC, temp, query_prompting, log):
    model_name_llama3 = 'myllama3:latest'   # change it into your own model name.
    full_answer1 = ollama.generate(model=model_name_llama3, prompt=query_prompting, format= "json")['response']
    query_ans = {"CC": CC, "model": model_name_llama3, "temprature": temp, "query": query_prompting,
                 "ans": full_answer1}
    log.write(str(query_ans) + '\n')
    # llama3 return bad format responses so need a bit of format correction.
    label = full_answer1.replace('  ', '').replace('\t', '').replace('\n', '').replace(' ,', ',').replace('} ', '}').replace(' }', '}')
    if full_answer1[-1] == '}':
        label = eval(full_answer1)["Answer"]
    else:
        try:
            label = eval(full_answer1 + '}')["Answer"]
        except:
            print(full_answer1)

    return label

# query using the template
def query(row, temp, out_column, model, log):
    column = out_column.split('temp')[1]
    if 'OnePrompt' in column:
        prompting = templates['OnePrompt_Basic'].format(row['CC'])
    elif column == '_Vf_ANT':
        prompting = templates['Vf'].format(row['Antecedent'], row['CC'])
    elif column == '_Vf_CON':
        prompting = templates['Vf'].format(row['Consequent'], row['CC'])
    elif column  == '_TV_ANT':
        prompting = templates['TV'].format(row['Antecedent'], row['CC'])
    elif column == '_TV_CON':
        prompting = templates['TV'].format(row['Consequent'], row['CC'])
    elif column == '_Causality':
        prompting = templates[column].format(row['Antecedent'], row['Consequent'])
    elif 'Al3' in column:
        prompting = templates['Al3'].format(row['CC'], row['Antecedent'], row['Consequent'])
    else:
        print('ERROR: invalid column name')
        return ''
    # either query gpt4 or llama3
    if 'gpt' in model.lower():
        label = query_gpt(row['CC'], temp, prompting, log)
    else:
        label = query_llama3(row['CC'], temp, prompting, log)
    return label


def run(dataframe, out_file, temp, out_column, model, log):
    number_lines = len(dataframe)
    chunksize = 12

    if (out_file is None):
        already_done = pd.DataFrame().reindex(columns=dataframe.columns)
        start_line = 0

    elif isinstance(out_file, str):
        if os.path.isfile(out_file):
            already_done = pd.read_csv(out_file)
            start_line = len(already_done)
        else:
            already_done = pd.DataFrame().reindex(columns=dataframe.columns)
            start_line = 0
    else:
        print('ERROR: "out_file" is of the wrong type, expected str')

    for i in tqdm(range(start_line, number_lines, chunksize)):
        sub_df = dataframe.iloc[i: i + chunksize]
        sub_df[out_column] = sub_df.apply(lambda x: query(x, temp, out_column, model, log), axis=1, result_type='expand')
        already_done = pd.concat([already_done, sub_df])
        already_done.to_csv(out_file, index=False)

    return already_done

if __name__ == "__main__":
    # ----------------Please modify the value of inputs/outputs below  ----------------
    # Give the temperature and the name of your model, input file and the log file.
    temp = 0
    model_name = 'GPT4'
    input_file = './data/Miscc_truth_values.csv'
    df = pd.read_csv(input_file, encoding='UTF-8-SIG', on_bad_lines='skip', lineterminator='\n')
    log_file_path = './data/log.jsonl'
    log = open(log_file_path, "a+")
    # Give the output columns for the model to predict.
    cols = ['_'.join([model_name, 'temp' + str(temp), c]) for c in ['OnePrompt', 'Al3']]
    # ----------------To modify above  ------------------------------------------------
    for i in cols:
        print('column ', i)
        df = run(df.loc[:, ~df.columns.str.contains('^Unnamed')], input_file.replace('.csv', '_'+ i + '.csv'), temp, i, model_name, log)
    log.close()
    print('finished')
