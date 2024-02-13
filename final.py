#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import json
import pickle
from datasets import load_dataset, Dataset
import os
from equation_generate import build_equations 
import random


# In[2]:


be = build_equations()

num_total_per_etype = 20000
num_test_per_etype = 0
digits = 1
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 0
digits = 2
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 3
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 4
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 5
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)



num_total_per_etype = 20000
num_test_per_etype = 0
digits = 1
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 0
digits = 2
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 3
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 4
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 5
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 6
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 50000
num_test_per_etype = 100
etype = '/'

etype = '/'
digits = 2
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=0,digits=digits)

etype = '/'
digits = 3
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

etype = '/'
digits = 4
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

etype = '/'
digits = 5
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)


num_total_per_etype = 20000
num_test_per_etype = 0
digits = 1
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 0
digits = 2
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 3
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 4
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 5
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)


'''
+3:1.0
+4:1.0
+5:1.0
-3:1.0
-4:1.0
-5:1.0
-6:1.0
/3:1.0
/4:1.0
/5:0.99
*3:1.0
*4:1.0
*5:1.0
'''


# In[3]:


#8d,16d +
num_total_per_etype = 50000
num_test_per_etype = 100
digits = 8
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 50000
num_test_per_etype = 100
digits = 16
etype = '+'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

#8d,16d -
num_total_per_etype = 100000
num_test_per_etype = 100
digits = 8
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 100000
num_test_per_etype = 100
digits = 16
etype = '-'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)


# In[4]:


#6d,8d, 16*1d
num_total_per_etype = 20000
num_test_per_etype = 100
digits = 6
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 8
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 20000
num_test_per_etype = 100
digits = 16
digits2 = 1
etype = '*'
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits,digits2=digits2)


# In[5]:


num_total_per_etype = 50000
num_test_per_etype = 100
etype = '/'
digits = 6
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 200000
etype = '/'
digits = 12
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

etype = '/'
digits = 16
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits)

num_total_per_etype = 200
etype = '/'
digits = 12
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits,digits2=6)

num_total_per_etype = 200
etype = '/'
digits = 16
be.create_eqs(etype=etype,num_total_per_etype=num_total_per_etype,num_test_per_etype=num_test_per_etype,digits=digits,digits2=1)


# In[6]:


be.build_dataset()


# In[7]:


print('num of training equations:',len(be.train_data))
print('num of test equations:',sum([len(i) for i in be.test_data.values()]))


# In[8]:


import sys
sys.path.append('../teach_llm_cal/TinyLlama/sft/')


# In[ ]:


from finetune import *

args.model_name_or_path = '../teach_llm_cal/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T/'
args.dataset = '../teach_llm_cal/TinyLlama/oasst_top1_2023-08-25/'
args.do_eval = training_args.do_eval = False
args.save_strategy = training_args.save_strategy = 'No'
args.evaluation_strategy = training_args.evaluation_strategy = 'No'
args.learning_rate = training_args.learning_rate = 1e-5

checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
model, tokenizer = get_accelerate_model(args, checkpoint_dir)

model.config.use_cache = False
print('loaded model')


# In[ ]:


set_seed(args.seed)
data_module = make_data_module(tokenizer=tokenizer, args=args)


# In[ ]:


train_dataset = [{'input':'','output':d} for d in be.train_data]
train_dataset = Dataset.from_list(train_dataset)
data_module['train_dataset'] = train_dataset


# In[ ]:


trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )


# In[ ]:


# Verifying the datatypes and parameter counts before training.
print_trainable_parameters(args, model)
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items(): total+= v
for k, v in dtypes.items():
    print(k, v, v/total)

all_metrics = {"run_name": args.run_name}


# In[ ]:


train_result = trainer.train()


# In[ ]:


from transformers import GenerationConfig
import math
generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
generation_config.do_sample = False


# In[ ]:


def evaluate(eqs):
    correct = 0
    outputs_texts = []
    for eq in eqs:
        left,right = eq.split('=')
        left += '='
        inputs = tokenizer(left, return_tensors="pt",padding=False).to('cuda')
        outputs = model.generate(**inputs, generation_config=generation_config)
        outputs_text = tokenizer.decode(outputs[0])
        pred_result = outputs_text.split('=')[-1].replace('</s>','')
        #print(outputs_text)
        #print(eq)
        #print('-'*30)
        if pred_result == right:
            correct += 1
        outputs_texts.append(outputs_text)
    return correct / len(eqs),outputs_texts
    
    
def evaluate_accuracy(test_data):
    results = {}
    for etype in test_data:
        td = test_data[etype]
        precision, outputs_text = evaluate(td)
        results[etype] = [precision, outputs_text]
    return results


# In[ ]:


results = evaluate_accuracy(be.test_data)
for k in results:
    print(f'{k}:{results[k][0]}')

