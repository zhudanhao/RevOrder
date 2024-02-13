# RevOrder
the data and code for paper:

RevOrder: A Novel Method for Enhanced Arithmetic in Language Models. 

This paper presents RevOrder, a novel technique aimed at improving arithmetic operations in large language models (LLMs) by reversing the output digits in addition, subtraction, and n-digit by 1-digit (nD by 1D) multiplication tasks. Our method significantly reduces the Count of Sequential Intermediate Digits (CSID) to O(1), a new metric we introduce to assess equation complexity. Through comprehensive testing, RevOrder not only achieves perfect accuracy in basic arithmetic operations but also substantially boosts LLM performance in division tasks, particularly with large numbers where traditional models struggle. Implementation of RevOrder is cost-effective for both training and inference phases. Moreover, applying RevOrder to fine-tune the LLaMA2-7B model on the GSM8K math task results in a considerable improvement, reducing equation calculation errors by 46% and increasing overall scores from 41.6 to 44.4.


https://arxiv.org/abs/2402.03822

## dependencies
python=3.10.6
torch=2.1.1+cu121
transformers=4.36.3.dev0

## Train the model
###


