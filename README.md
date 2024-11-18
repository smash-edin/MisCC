# MisCC
This is a repository for the project on misinformation detection on counterfactual claims. 

## Dataset
The data was collected from subreddits about climate change using Reddit's public API. Subreddits are `climateskeptics', `climatechange', `ClimateOffensive', `climate\_science',   `RenewableEnergy', `climatedisalarm', `conspiracy' and the posts date from between 2017 and 2022.  The data file is given under the data folder, and named **Miscc_truth_values.csv** contains all counterfactual claims from these posts that have been annotated with truth values. The columns and their descriptions are given below.
- CC: the counterfactual claim. 
- Antecedent: the antecedent of that CC.
- Consequent: the consequent of that CC. 
- Vf_Ant: the verifiability of the antecedent.
- Vf_Con:  the verifiability of the consequent.
- TV_Ant: the truth value of the antecedent.
- TV_Con: the truth value of the consequent.
- Causality: the truth value of the fact that the antecedent causes the consequent.
- *__conf: the confidence on the * label.
- 'TV-CC': the truth value of the CC calculated based on labels of the true value of ANT, CON and causality.

The baseline file is also given in the data folder, and it is named **MisCC_baseline.csv**. It contains LLMs' predictions. Here we used GPT4 and Llama3-8B with temperature 0.
- '*_Vf_Ant': the verifiablity of ANT predicated by LLM *.
- '*_Vf_Con': the verifiablity of CON predicated by LLM *.
- '*_TV_Ant': the truth value of ANT predicated by LLM *.
- '*_TV_Con': the truth value of CON predicated by LLM *.
- '*_Causality': whether the ANT causes the CON predicted by LLM *.
- '*_TV-CC': the truth value of the CC calculated based on the predicted labels of the true value of ANT, CON and causality by LLM *.
- '*_Al3': the truth value of the CC predicted by LLM * with the algorithm expressed in the prompting, which is sort of chain of thoughts.
- '*_OnePrompt': the truth value of the CC predicted by LLM * by directly asking a CC's truth value without any other informaiton in the prompting.
- '*_inconsistency': whether predictions given by LLM * are consistent.

