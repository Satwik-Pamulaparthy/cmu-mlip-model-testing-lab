# cmu-mlip-model-testing-lab

# Lab 4: Model Testing with Zeno and LLM

In this lab, This project analyzes sentiment in tweets using GPT-2 and RoBERTa, leveraging Zeno for evaluation and error analysis. The goal is to understand model weaknesses, improve accuracy, and generate new test cases.


## Deliverables
- [x] Successfully start a local Zeno server on the dataset provided, with metrics and model predictions
- [x] Create 5 slices in the Zeno interface, derive meaningful insights and showcase them to the TA
- [x] Write down 3 additional slices you want to create and successfully generate 10 examples for one selected slice


## Getting started
- Clone the starter code from this [Git repository](https://github.com/malusamayo/cmu-mlip-model-testing-lab).
- The repository includes a python notebook which contains the starter code.

## Installation instructions
- python 3.10 version is needed for the zeno packages to run correctly
- pip install zenoml datasets transformers tqdm

## Code related details
- Finish all 7 steps mentioned in the python notebook
- If you have trouble downloading the datasets and/or running model inference, use `tweets.csv` shared in the folder
- If you have trouble starting a local Zeno server, copy the code in `zenohub.py` to the notebook and follow the steps
- If you have trouble using the GPTs provided, use plain ChatGPT for test case generation

Slice Name	                                            Purpose
Tweets with Hashtags	                  Tests model performance on trending topics.
Strong Positive Words	        Checks if models over-rely on specific words like ‘love’ or ‘great’.
Very Long Tweets	                    Evaluates sentiment retention over longer contexts.
Tweets with Questions	            Tests if models struggle with rhetorical vs. factual questions.
High Model Disagreement	        Captures tweets where GPT-2 and RoBERTa give different predictions.


Key Observations
- [x]RoBERTa consistently outperforms GPT-2, especially for longer tweets (70% vs. 37% accuracy).
- [x]GPT-2 struggles with sarcasm and negation (e.g., "I don’t hate it" misclassified as negative).
- [x]Questions and ambiguous tweets lead to higher misclassification rates in both models.
