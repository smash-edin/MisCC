# MisCC
This is a repository for the project on misinformation detection on counterfactual claims. 

## Dataset
The data was collected from subreddits about climate change using Reddit's public API. Subreddits are `climateskeptics', `climatechange', `ClimateOffensive', `climate\_science',   `RenewableEnergy', `climatedisalarm', `conspiracy' and the posts date from between 2017 and 2022.  The file **Miscc_truth_values.csv** contains all counterfactual claims from these posts that have been annotated with truth values. The columns and their descriptions are given below.
- CC: the counterfactual claim. 
- Antecedent: the antecedent of that CC.
- Consequent: the consequent of that CC. 
- Vf_Ant: the verifiability of the antecedent.
- Vf_Con:  the verifiability of the consequent.
- TV_Ant: the truth value of the antecedent.
- TV_Con: the truth value of the consequent.
- Causality: the truth value of the fact that the antecedent causes the consequent.
- *__conf: the confidence on the * label.
