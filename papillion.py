#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('pip', 'install dspy-ai==2.5.41 sglang[all] datasets huggingface python-dotenv')


# In[2]:


# %pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ --force-reinstall


# In[4]:


PORT_NUMBER = 7501 # You can change the port number here

# Run
# tmux new -s sglang_server
# source ~/miniconda3/bin/activate && conda activate ./env
# CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct

# Run notebook
# tmux new -s papillion
# source ~/miniconda3/bin/activate && conda activate ./env
# jupyter nbconvert --to notebook --execute papillion.ipynb


# Run this command to start the server
# 
# ```bash
# tmux new -s sglang_server
# 
# source ~/miniconda3/bin/activate && conda activate ./env
# CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.1-8B-Instruct
# ```

# In[3]:


import dspy
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


local_lm = dspy.LM('openai/sglang/Llama-3.1-8B-Instruct', api_base=f"http://127.0.0.1:{PORT_NUMBER}/v1", api_key="", max_tokens=4000)
dspy.configure(lm=local_lm)

openai_lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=4000)


# In[18]:


import dspy

class CreateOnePrompt(dspy.Signature):
    """
    You are a helpful assistant that is very mindful of user privacy. You have access to a powerful large language model that you can query. Given a user request, create a prompt for your large language model that preserves user privacy, so that this model can help you complete the user request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY GENERATE A PROMPT.
    """
    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    createdPrompt = dspy.OutputField()

class InfoAggregator(dspy.Signature):
    """
    You are a helpful assistant. Respond to queries from the user.
    """

    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    modelExampleResponses = dspy.InputField(desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information.")
    finalOutput = dspy.OutputField()

class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.prompt_creater = dspy.ChainOfThought(CreateOnePrompt)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        try:
            prompt = self.prompt_creater(userQuery=user_query).createdPrompt
            response = self.untrusted_model(prompt)[0]
            output = self.info_aggregator(userQuery=user_query, modelExampleResponses=response)
        except Exception:
            return dspy.Prediction(prompt="", output="", gptResponse="")

        return dspy.Prediction(prompt=prompt, output=output.finalOutput, gptResponse=response)


# In[8]:


from datasets import load_dataset

pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")
pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")


# In[14]:


# Let's first take a look at some data examples
from IPython.display import display
display(pupa_tnb["train"].to_pandas().head(3))
print(pupa_tnb["train"]["user_query"][0])
print(pupa_tnb["train"]["pii_units"][0])
print(pupa_tnb["train"]["redacted_query"][0])


# In[20]:


examples = [
    dspy.Example({
        "target_response": x["target_response"],
        "user_query": x["user_query"],
        "pii_str": x["pii_units"]
    }).with_inputs("user_query")
    for x in pupa_new["train"]
]

trainset, devset, testset = examples[:150], examples[150:300], examples[300:]
print(f"Loaded {len(trainset)} training examples, {len(devset)} dev examples, and {len(testset)} test examples.")


# In[15]:


class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()

class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that are simultaneously (i) forms of PII and (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()

class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set(pii_str.split("||"))) # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)

dspy.configure(experimental=True)
llm_judge = LLMJudge()
llm_judge.set_lm(openai_lm)


# In[16]:


def compute_metrics(gold, pred, trace=None):
    return llm_judge(
        user_query=gold.user_query,
        new_resp=pred.output,
        og_resp=gold.target_response,
        updated_query=pred.prompt,
        pii_str=gold.pii_str,
    )

def compute_quality(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).quality

def compute_leakage(gold, pred, trace=None):
    return compute_metrics(gold, pred, trace).leakage

def compute_overall_score(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score >= 1.0 if trace is not None else overall_score


# In[21]:


zeroshot = PAPILLON(untrusted_model=openai_lm)

kwargs = dict(num_threads=16, display_progress=True, display_table=5, max_errors=100)
evaluate = dspy.Evaluate(metric=compute_overall_score, devset=devset, **kwargs)


# In[22]:


# Let's evaluate response quality!
# evaluate(zeroshot, metric=compute_quality)


# In[23]:


# Let's evaluate PII leakage!
# evaluate(zeroshot, metric=compute_leakage)


# In[29]:


# Let's print an example user_query from the devset
print("\nExample user_query from devset:")
if len(devset) > 0:
    example = devset[8]
    print(example.user_query)
else:
    print("No examples available in the devset.")


# In[31]:


user_query = """
From the list of following programs: BIDA from CMU, Machine Learning from Columbia University, Data Science from JHU, AI from JHU, and MBA from JHU, which are the best compliment programs if I want to study for 2 Masters programs and to become a technical leader at the Boeing Company?
"""
# print(zeroshot(user_query=user_query))


# In[35]:


# dspy.inspect_history(n=2)
# local_lm.inspect_history(n=2)


# In[36]:


# openai_lm.inspect_history(n=1)


# In[37]:


models = dict(prompt_model=openai_lm, task_model=local_lm)
optimizer = dspy.MIPROv2(metric=compute_overall_score, auto="medium", num_threads=16, **models)

kwargs = dict(minibatch_size=35, max_bootstrapped_demos=5, max_labeled_demos=0)
opt_papillon = optimizer.compile(zeroshot, trainset=trainset, **kwargs)


# In[ ]:


opt_papillon.save("optimized_papillion.json")

