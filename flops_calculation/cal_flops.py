from fvcore.nn.flop_count import FlopCountAnalysis 
from fvcore.nn.parameter_count import parameter_count, parameter_count_table

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, OPTForCausalLM

import torch

from importlib import import_module

import json

'''
## GPT
model = GPT2LMHeadModel.from_pretrained('gpt2') 

tensor = (torch.rand(1, 1024).long(),)

# analyze FLOPs
flops = FlopCountAnalysis(model, tensor)
print("GFLOPs: ", flops.total()/10**9)

with open('gpt2_layer_flops.json', 'w') as fp:
    json.dump(dict(flops.by_operator().items()), fp)

with open('gpt2_module_flops.json', 'w') as fp:
    json.dump(dict(flops.by_module().items()), fp)
    
# analyze parameters
print(parameter_count(model))

with open('gpt2_param.json', 'w') as param:
    json.dump(dict(parameter_count(model).items()), param)
'''

## OPT
for n in ['opt-125m', 'opt-6.7b']:
    model = OPTForCausalLM.from_pretrained(f"facebook/{n}")
    
    tensor = (torch.rand(1, 1024).long(),)
    
    # analyze FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("GFLOPs: ", flops.total()/10**9)
    
    with open(f'{n}_layer_flops.json', 'w') as fp:
        json.dump(dict(flops.by_operator().items()), fp)
    
    with open(f'{n}_module_flops.json', 'w') as fp:
        json.dump(dict(flops.by_module().items()), fp)
        
    # analyze parameters
    print(parameter_count(model))
    
    with open(f'{n}_param.json', 'w') as param:
        json.dump(dict(parameter_count(model).items()), param)
    

## LLAMA

for n in ['llama-7b', 'llama-13b', 'llama-30b', 'llama-65b']:
    model = LlamaForCausalLM.from_pretrained(f"decapoda-research/{n}-hf")
    
    tensor = (torch.rand(1, 1024).long(),)
    '''
    # analyze FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("GFLOPs: ", flops.total()/10**9)
    
    with open(f'{n}_layer_flops.json', 'w') as fp:
        json.dump(dict(flops.by_operator().items()), fp)
    
    with open(f'{n}_module_flops.json', 'w') as fp:
        json.dump(dict(flops.by_module().items()), fp)
    '''
    # analyze parameters
    print(parameter_count(model))
    
    with open(f'{n}_param.json', 'w') as param:
        json.dump(dict(parameter_count(model).items()), param)