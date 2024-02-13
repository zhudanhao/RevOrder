# RevOrder
the data and code for paper:

RevOrder: A Novel Method for Enhanced Arithmetic in Language Models. 

This paper presents RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks. Our method significantly reduces the Count of Sequential Intermediate Digits (CSID) to O(1), a new metric we introduce to assess equation complexity. Through comprehensive testing, RevOrder not only achieves perfect accuracy in basic arithmetic operations but also substantially boosts LLM performance in division tasks, particularly with large numbers where traditional models struggle. Implementation of RevOrder is cost-effective for both training and inference phases. Moreover, applying RevOrder to fine-tune the LLaMA2-7B model on the GSM8K math task results in a considerable improvement, reducing equation calculation errors by 46% and increasing overall scores from 41.6 to 44.4.


https://arxiv.org/abs/2402.03822

## dependencies
python=3.10.6

torch=2.1.1+cu121

transformers=4.36.3.dev0

accelerate=0.26.1

## Train the model
### clone TinyLLama
git clone https://github.com/jzhang38/TinyLlama

### download their pretrained checkpoints
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

Note: Whether to use the pretrained model does not affect the results.  You can also choose to initialize the model randomly.

### Train for 2 epochs with lr=1e-4 and save the checkpoint

Change the corresponding settings in final.sh.

sh final.sh

### Train for 1 epoch with lr=1e-5 on the previous checkpoint
change the corresponding settings in final.sh.

sh final.sh

## evaluate
open evauate.ipynb, and run step by step.


## some notes
### The instructions above should guide you to reproduce our results exactly. Feel free to contact me with any questions,  Danhao Zhu, 229369897@qq.com
### Simply use an initial lr=1e-4, and a linear or cosin lr schedule can also reproduce our results. The instructions in this document have to be consistent with our paper. But the complication is not neccessary in practice.
### Indeed, we have found, with a much less training data (maybe 1/5-10/1) than our paper, and much less training steps, RevOrder can also achieve the perfect results on arithmetic problems. Just tune the lr~ But since this paper is under review, we do not update our paper and the code.
