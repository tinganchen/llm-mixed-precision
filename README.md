# LLM Project - Mixed-precision LLM Models by Bit Allocation for Efficiency
(@NVIDIA Research Team & NTU)
Mixed-precision LLMs, Transformer Blocks, Quantization Error, Bit Allocation

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Motivations
Quantization errors are varying within different transformer blocks, especially for larger LLMs (see ["fig/intro_error.pdf"](fig/intro_error.pdf)). That is, some blocks are much sensitive to bitwidth compression, which may lead to severe performance degradation. Therefore, in this project, we tend to assign more bits to these sensitive blocks and fewer bits to the other blocks for the maintenance of efficiency.



## Overview

* Overview of Mixed-prcision LLMs (see ["fig/overview.pdf"](fig/overview.pdf))

* Illustration of Bit Allocation Stategy (see ["fig/overview_attention.pdf"](fig/overview_attention.pdf)) - Measurement of block adversaries, i.e., quantization errors 


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
## Results

See ["fig/result.png"](fig/result.png).

## Reference

* Download of Datasets & Evaluation of LLM Models - [Github Link](https://github.com/EleutherAI/lm-evaluation-harness)
* Mixed-precision CNN Models - [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cai_ZeroQ_A_Novel_Zero_Shot_Quantization_Framework_CVPR_2020_paper.pdf)
