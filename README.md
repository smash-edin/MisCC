# MisCC
This is a repository for the project on misinformation detection on counterfactual claims (CCs). The paper about this dataset and baseline is available [here](https://www.research.ed.ac.uk/en/publications/miscc-misinformation-detection-on-counterfactual-claims). GPT4 and Llama3-8B are used as the baseline for the MisCC dataset.

## Dataset
The data was collected from subreddits about climate change using Reddit's public API. Subreddits are `climateskeptics', `climatechange', `ClimateOffensive', `climate\_science',   `RenewableEnergy', `climatedisalarm', `conspiracy' and the posts date from between 2017 and 2022.  The data file is given under the data folder.

- **MisCC_CC_Classification.csv** contains the classification of claims. The main columns are given below.
  - Text: the claim.
  - CC_Category: whether it is a counterfactual claim.
  - Ill-formed: whether the claim is reformed to be a CC from a claim that is conditional but not counterfactual.
  - CNotCC: whether the clam is conditional but not CC.
  - llama3_classification: the classification label by Llama3-8B.

- **Miscc_truth_values.csv** contains all counterfactual claims from these posts that have been annotated with truth values. The columns and their descriptions are given below.
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

- **MisCC_baseline.csv** is the baseline file, which contains LLMs' predictions. Here we used GPT4 and Llama3-8B with temperature 0.
  - *_Vf_Ant: the verifiablity of ANT predicated by LLM *.
  - *_Vf_Con: the verifiablity of CON predicated by LLM *.
  - *_TV_Ant: the truth value of ANT predicated by LLM *.
  - *_TV_Con: the truth value of CON predicated by LLM *.
  - *_Causality: whether the ANT causes the CON predicted by LLM *.
  - *_TV-CC: the truth value of the CC calculated based on the predicted labels of the true value of ANT, CON and causality by LLM *.
  - *_Al3: the truth value of the CC predicted by LLM * with the algorithm expressed in the prompting, i.e., an approach using chain of thoughts.
  - *_OnePrompt: the truth value of the CC predicted by LLM * by directly asking a CC's truth value without any other information in the prompting.
  - *_inconsistency: whether predictions given by LLM * are consistent.

## Scripts
- query.py sends queries to GPT4 or Llama3-8B via Ollama and saves their response to a log file and CSV file.

- cc_truthvalue.py computes the truth value of a CC based on the corresponding truth values of ANT, CON and Causality. The equation is given below, where unary function T returns the truth value of its argument: ⊤ for true, ⊥ for false and ω for unknown. Thus, ¬⊥ = ⊤ and ¬ω = ω; Ca is its antecedent, Cc is its consequent, and Ccaus the causality.
<p align="center">
  <img width="371" alt="image" src="https://github.com/user-attachments/assets/085c29b4-c4e8-4762-8839-fbc319edea3b">
</p>

- evaluation.py get the statistics of a predicted column with its gold standard column, i.e., precision, recall and F1 score, and save the report in a CSV file.
