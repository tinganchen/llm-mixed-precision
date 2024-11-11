# LLM Project - Mixed-precision LLM Models by Bit Allocation for Efficiency
Mixed-precision LLMs, Transformer Blocks, Quantization Error, Bit Allocation

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Overview



## Experiments

Task                | LLM Models               | Evaluation Metric   | Datasets  
---                  |---                  |---                                    |---                    
Text Generation |GPT-2 (124M) & OPT (1.3B, 2.7B, 6.7B) & LLaMA (7B, 13B, 30B, 65B)           | Perplexity                                    | WikiText-2 & PTB & C4           
Other Tasks (eg. Question Answering)  |GPT-2 (124M) & OPT (1.3B, 2.7B, 6.7B) & LLaMA (7B, 13B, 30B, 65B)           | Perplexity / Accuracy (%)                                   | [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

## Preparation

### Benchmark Datasets & Evaluation

```shell
cd text_generation/
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```

```shell
cd qa/
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```

ps. After the repo is cloned, code needs to be modified subtly to be consistent with the inference code.

### Calculation of FLOPs

```shell
cd flops_calculation/
python3 cal_flops.py
```

## Implementation

### Task - Text Generation (eg. LLaMA-13B)

#### Search and Bit Allocation

```shell
cd text_generation/
bash search.sh
```

#### Inference of Mixed-precision LLM Models

```shell
cd text_generation/
bash run.sh
```

## Reference

* Download of Datasets & Evaluation of LLM Models - [Github Link](https://github.com/EleutherAI/lm-evaluation-harness)
