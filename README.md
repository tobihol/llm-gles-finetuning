# Addressing Systematic Non-response Bias with Supervised Fine-Tuning of Large Language Models: A Case Study on German Voting Behaviour
Authors: Tobias Holtdirk, Dennis Assenmacher, Arnim Bleier, Claudia Wagner

This repository contains the code for the paper.

Preprint DOI: [10.31219/osf.io/udz28](https://doi.org/10.31219/osf.io/udz28)

### Abstract
A major challenge for survey researchers is dealing with missing data, which restricts the scope of analysis and the reliability of inferences that can be drawn. Recently, researchers have started investigating the potential of Large Language Models (LLMs) to role-play a pre-defined set of "characters" and simulate their survey responses with little or no additional training data and costs. Previous research has mostly focused on zero-shot LLM predictions. However, often other survey responses are at least partially available. This work investigates the viability and robustness of supervised fine-tuning on these responses to simulate systematic and random item-level non-responses in the context of German voting behaviour. Our results show when systematic item non-responses are present, fine-tuned LLMs outperform traditional classification approaches on survey data. Fine-tuned LLMs also seem to be more robust to changes in the set of features that the model can use to make predictions. Finally, we see that fine-tuned LLMs match the performance of traditional classification methods when survey responses are missing completely at random.

---

### Setup

To setup the environment, install the uv package manager and run: 

```bash
uv sync
```

Download the GLES 2017 dataset from GESIS (https://doi.org/10.4232/1.13410) and put it in the `datasets` folder.


### Experiments

Run to reproduce the experiments from the paper:

```bash
python evaluation/rq1_gles2017_vote_cls.py
python evaluation/rq1_gles2017_vote_cls_no_party_id.py
python evaluation/rq2_gles2017_vote_cls_uni_and_school.py
python evaluation/rq2_gles2017_vote_cls_uni_and_school_no_id.py
python evaluation/rq2_gles2017_vote_cls_party_id_exclusion.py
```
(add a wandb user and project id to the evaluation scripts for logging)

---

### Funding
This work received funding from the German Research Foundation (DFG) under project no. 504226141.