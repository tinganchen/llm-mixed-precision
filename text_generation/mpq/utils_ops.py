import torch
import numpy as np
import math
import json
from utils.options import args

def layer_param_calculator():
    
    if args.model == 'gpt2':
        FILE_NAME = "../flops_calculation/gpt2_param.json"
        NUM_LAYERS = 12

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
            
        layer_params = []
        
        for layer in range(NUM_LAYERS):
            
            num_param = 0
            
            layer_prefix = f'transformer.h.{layer}'
            
            for k, v in param_data.items():
                if layer_prefix in k:
                    num_param += v
            
            layer_params.append(num_param)
    
    elif 'opt' in args.model:
        if 'opt-125m' in args.model:
            FILE_NAME = "../flops_calculation/opt-125m_param.json"
            NUM_LAYERS = 12
        
        elif 'opt-350m' in args.model:
            FILE_NAME = "../flops_calculation/opt-350m_param.json"
            NUM_LAYERS = 24
        
        elif 'opt-1.3b' in args.model:
            FILE_NAME = "../flops_calculation/opt-1.3b_param.json"
            NUM_LAYERS = 24
            
        elif 'opt-2.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-2.7b_param.json"
            NUM_LAYERS = 32
        
        elif 'opt-6.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-6.7b_param.json"
            NUM_LAYERS = 32
            
        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
            
        layer_params = []
        
        for layer in range(NUM_LAYERS):
            
            num_param = 0
            
            layer_prefix = f'model.decoder.layers.{layer}'
            
            for k, v in param_data.items():
                if layer_prefix in k:
                    num_param += v
            
            layer_params.append(num_param)
            
    elif 'llama' in args.model:
        if 'llama-7b' in args.model:
            FILE_NAME = "../flops_calculation/llama-7b_param.json"
            NUM_LAYERS = 32
        
        elif 'llama-13b' in args.model:
            FILE_NAME = "../flops_calculation/llama-13b_param.json"
            NUM_LAYERS = 40
        
        elif 'llama-30b' in args.model:
            FILE_NAME = "../flops_calculation/llama-30b_param.json"
            NUM_LAYERS = 60
            
        elif 'llama-65b' in args.model:
            FILE_NAME = "../flops_calculation/llama-65b_param.json"
            NUM_LAYERS = 80
            
        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
            
        layer_params = []
        
        for layer in range(NUM_LAYERS):
            
            num_param = 0
            
            layer_prefix = f'model.layers.{layer}'
            
            for k, v in param_data.items():
                if layer_prefix in k:
                    num_param += v
            
            layer_params.append(num_param)
            
        
    return layer_params


def param_calculator(bitcfg):
    
    if args.model == 'gpt2':
        FILE_NAME = "../flops_calculation/gpt2_param.json"
        NUM_LAYERS = 12
        
        if bitcfg is None:
            bitcfg = np.array([args.bitW]*NUM_LAYERS)
   

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
        
        org_model_size = param_data['transformer']*4 / 1e6
        
        layer_quant_param = []
        layer_float_param = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_param = 0
            num_float_param = 0
            
            layer_quant_prefix = [f'transformer.h.{layer}.attn',
                                  f'transformer.h.{layer}.mlp']
            
            layer_float_prefix = [f'transformer.h.{layer}.ln_1',
                                  f'transformer.h.{layer}.ln_2']
            
            for k, v in param_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_param += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_float_param += v
            
            layer_quant_param.append(num_quant_param)
            layer_float_param.append(num_float_param)
        
        num_other_float_param = 0
        
        for k, v in param_data.items():
            if k in ['transformer.wte', 'transformer.wpe', 
                     'transformer.ln_f']:
                num_other_float_param += v
                
        quant_model_size = (sum(bitcfg * np.array(layer_quant_param)) / 8 + (sum(layer_float_param) + num_other_float_param) * 4)  / 1e6
        
        avg_bit = sum(bitcfg * np.array(layer_quant_param)) / sum(layer_quant_param)
        
    elif 'opt' in args.model:
        if 'opt-125m' in args.model:
            FILE_NAME = "../flops_calculation/opt-125m_param.json"
            NUM_LAYERS = 12
        
        elif 'opt-350m' in args.model:
            FILE_NAME = "../flops_calculation/opt-350m_param.json"
            NUM_LAYERS = 24
        
        elif 'opt-1.3b' in args.model:
            FILE_NAME = "../flops_calculation/opt-1.3b_param.json"
            NUM_LAYERS = 24
            
        elif 'opt-2.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-2.7b_param.json"
            NUM_LAYERS = 32
        
        elif 'opt-6.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-6.7b_param.json"
            NUM_LAYERS = 32
            
        if bitcfg is None:
            bitcfg = np.array([args.bitW]*NUM_LAYERS)

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
        
        org_model_size = param_data['']*4 / 1e6
        
        layer_quant_param = []
        layer_quant_fixed_param = []
        layer_float_param = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_param = 0
            num_quant_fixed_param = 0
            num_float_param = 0
            
            layer_quant_prefix = [f'model.decoder.layers.{layer}.self_attn']
            
            layer_quant_fixed_prefix = []
            
            layer_float_prefix = [f'model.decoder.layers.{layer}.self_attn_layer_norm',
                                  f'model.decoder.layers.{layer}.fc1',
                                  f'model.decoder.layers.{layer}.fc2']
            
            for k, v in param_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_param += v
                
                for prefix in layer_quant_fixed_prefix:
                    if prefix in k:
                        num_quant_fixed_param += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_float_param += v
            
            layer_quant_param.append(num_quant_param)
            layer_quant_fixed_param.append(num_quant_fixed_param)
            layer_float_param.append(num_float_param)
        
        num_other_quant_fixed_param = 0
        
        for k, v in param_data.items():
            if k in ['model.decoder.project_out', 
                     'model.decoder.project_in']:
                num_other_quant_fixed_param += v
                
        num_other_float_param = 0
        
        for k, v in param_data.items():
            if k in ['model.decoder.embed_tokens', 
                     'model.decoder.embed_positions', 
                     'model.decoder.final_layer_norm', 
                     'lm_head']:
                num_other_float_param += v
                
        quant_model_size = ((sum(bitcfg * np.array(layer_quant_param)) + sum(args.bitW * np.array(layer_quant_fixed_param)) + args.bitW * num_other_quant_fixed_param) / 8 + (sum(layer_float_param) + num_other_float_param) * 4)  / 1e6
        
        avg_bit = sum(bitcfg * np.array(layer_quant_param)) / sum(layer_quant_param)
   
    elif 'llama' in args.model:
        if 'llama-7b' in args.model:
            FILE_NAME = "../flops_calculation/llama-7b_param.json"
            NUM_LAYERS = 32
        
        elif 'llama-13b' in args.model:
            FILE_NAME = "../flops_calculation/llama-13b_param.json"
            NUM_LAYERS = 40
        
        elif 'llama-30b' in args.model:
            FILE_NAME = "../flops_calculation/llama-30b_param.json"
            NUM_LAYERS = 60
            
        elif 'llama-65b' in args.model:
            FILE_NAME = "../flops_calculation/llama-65b_param.json"
            NUM_LAYERS = 80
            
        if bitcfg is None:
            bitcfg = np.array([args.bitW]*NUM_LAYERS)

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            param_data = json.load(file)
        
        org_model_size = param_data['']*4 / 1e6
        
        layer_quant_param = []
        layer_quant_fixed_param = []
        layer_float_param = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_param = 0
            num_quant_fixed_param = 0
            num_float_param = 0
            
            layer_quant_prefix = [f'model.layers.{layer}.self_attn']
            
            layer_quant_fixed_prefix = [f'model.decoder.layers.{layer}.mlp']
            
            layer_float_prefix = [f'model.layers.{layer}.input_layernorm',
                                  f'model.layers.{layer}.post_attention_layernorm']
            
            for k, v in param_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_param += v
                
                for prefix in layer_quant_fixed_prefix:
                    if prefix in k:
                        num_quant_fixed_param += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_float_param += v
            
            layer_quant_param.append(num_quant_param)
            layer_quant_fixed_param.append(num_quant_fixed_param)
            layer_float_param.append(num_float_param)
        
        num_other_quant_fixed_param = 0
        
        for k, v in param_data.items():
            if k in ['_']:
                num_other_quant_fixed_param += v
                
        num_other_float_param = 0
        
        for k, v in param_data.items():
            if k in ['model.embed_tokens', 
                     'model.norm', 
                     'lm_head']:
                num_other_float_param += v
                
        quant_model_size = ((sum(bitcfg * np.array(layer_quant_param)) + sum(args.bitW * np.array(layer_quant_fixed_param)) + args.bitW * num_other_quant_fixed_param) / 8 + (sum(layer_float_param) + num_other_float_param) * 4)  / 1e6
        
        avg_bit = sum(bitcfg * np.array(layer_quant_param)) / sum(layer_quant_param)
   
    return avg_bit, org_model_size, quant_model_size  # (MB)


def bop_calculator(wbitcfg, abitcfg):
    
    if args.model == 'gpt2':
        FILE_NAME = "../flops_calculation/gpt2_module_flops.json"
        NUM_LAYERS = 12
        
        if wbitcfg is None:
            wbitcfg = np.array([args.bitW]*NUM_LAYERS)
            
        if abitcfg is None:
            abitcfg = np.array([args.abitW]*NUM_LAYERS)

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            flop_data = json.load(file)
        
        
        layer_quant_op = []
        layer_flop = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_op = 0
            num_flop = 0
            
            layer_quant_prefix = [f'transformer.h.{layer}.attn',
                                  f'transformer.h.{layer}.mlp']
            
            layer_float_prefix = [f'transformer.h.{layer}.ln_1',
                                  f'transformer.h.{layer}.ln_2']
            
            for k, v in flop_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_op += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_flop += v
            
            layer_quant_op.append(num_quant_op)
            layer_flop.append(num_flop)
        
        num_other_flop = 0
        
        for k, v in flop_data.items():
            if k in ['transformer.wte', 'transformer.wpe', 
                     'transformer.ln_f']:
                num_other_flop += v
                
        quant_bop = int((sum(np.array(layer_quant_op) * wbitcfg*abitcfg) + (sum(layer_flop) + num_other_flop) * 32*32) / 1e9)
    
    elif 'opt' in args.model:
        if 'opt-125m' in args.model:
            FILE_NAME = "../flops_calculation/opt-125m_module_flops.json"
            NUM_LAYERS = 12
        
        elif 'opt-350m' in args.model:
            FILE_NAME = "../flops_calculation/opt-350m_module_flops.json"
            NUM_LAYERS = 24
            
        elif 'opt-1.3b' in args.model:
            FILE_NAME = "../flops_calculation/opt-1.3b_module_flops.json"
            NUM_LAYERS = 24
            
        elif 'opt-2.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-2.7b_module_flops.json"
            NUM_LAYERS = 32
        
        elif 'opt-6.7b' in args.model:
            FILE_NAME = "../flops_calculation/opt-6.7b_module_flops.json"
            NUM_LAYERS = 32
            
        if wbitcfg is None:
            wbitcfg = np.array([args.bitW]*NUM_LAYERS)
            
        if abitcfg is None:
            abitcfg = np.array([args.abitW]*NUM_LAYERS)

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            flop_data = json.load(file)
        
        
        layer_quant_op = []
        layer_quant_fixed_op = []
        layer_flop = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_op = 0
            num_quant_fixed_op = 0
            num_flop = 0
            
            layer_quant_prefix = [f'model.decoder.layers.{layer}.self_attn']
            
            layer_quant_fixed_prefix = [f'model.decoder.layers.{layer}.fc1',
                                        f'model.decoder.layers.{layer}.fc2']
            
            layer_float_prefix = [f'model.decoder.layers.{layer}.self_attn_layer_norm']
            
            for k, v in flop_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_op += v
                        
                for prefix in layer_quant_fixed_prefix:
                    if prefix in k:
                        num_quant_fixed_op += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_flop += v
            
            layer_quant_op.append(num_quant_op)
            layer_quant_fixed_op.append(num_quant_fixed_op)
            layer_flop.append(num_flop)
        
        num_other_quant_fixed_op = 0
        
        for k, v in flop_data.items():
            if k in ['model.decoder.project_out', 
                     'model.decoder.project_in']:
                num_other_quant_fixed_op += v
                
        num_other_flop = 0
        
        for k, v in flop_data.items():
            if k in ['model.decoder.embed_tokens', 
                     'model.decoder.embed_positions', 
                     'model.decoder.final_layer_norm', 
                     'lm_head']:
                num_other_flop += v
                
        layer_quant_bop = (sum(np.array(layer_quant_op)* wbitcfg*abitcfg) + sum(np.array(layer_quant_fixed_op) * args.bitW*32))
        
        layer_flop = sum(layer_flop) * 32*32
        
        other_quant_bop = num_other_quant_fixed_op * args.bitW*32
        
        other_flop = num_other_flop * 32*32
        
        quant_bop = int((layer_quant_bop + layer_flop + other_quant_bop + other_flop) / 1e9)

    elif 'llama' in args.model:
        if 'llama-7b' in args.model:
            FILE_NAME = "../flops_calculation/llama-7b_module_flops.json"
            NUM_LAYERS = 32
        
        elif 'llama-13b' in args.model:
            FILE_NAME = "../flops_calculation/llama-13b_module_flops.json"
            NUM_LAYERS = 40
        
        elif 'llama-30b' in args.model:
            FILE_NAME = "../flops_calculation/llama-30b_module_flops.json"
            NUM_LAYERS = 60
            
        elif 'llama-65b' in args.model:
            FILE_NAME = "../flops_calculation/llama-65b_module_flops.json"
            NUM_LAYERS = 80
        
        
        if wbitcfg is None:
            wbitcfg = np.array([args.bitW]*NUM_LAYERS)
            
        if abitcfg is None:
            abitcfg = np.array([args.abitW]*NUM_LAYERS)

        with open(FILE_NAME) as file:
            # Load its content and make a new dictionary
            flop_data = json.load(file)
        
        
        layer_quant_op = []
        layer_quant_fixed_op = []
        layer_flop = []
        
        for layer in range(NUM_LAYERS):
            
            num_quant_op = 0
            num_quant_fixed_op = 0
            num_flop = 0
            
            layer_quant_prefix = [f'model.layers.{layer}.self_attn']
            
            layer_quant_fixed_prefix = [f'model.decoder.layers.{layer}.mlp']
            
            layer_float_prefix = [f'model.layers.{layer}.input_layernorm',
                                  f'model.layers.{layer}.post_attention_layernorm']
            
            for k, v in flop_data.items():
                for prefix in layer_quant_prefix:
                    if prefix in k:
                        num_quant_op += v
                        
                for prefix in layer_quant_fixed_prefix:
                    if prefix in k:
                        num_quant_fixed_op += v
                
                for prefix in layer_float_prefix:
                    if prefix in k:
                        num_flop += v
            
            layer_quant_op.append(num_quant_op)
            layer_quant_fixed_op.append(num_quant_fixed_op)
            layer_flop.append(num_flop)
        
        num_other_quant_fixed_op = 0
        
        for k, v in flop_data.items():
            if k in ['_']:
                num_other_quant_fixed_op += v
                
        num_other_flop = 0
        
        for k, v in flop_data.items():
            if k in ['model.embed_tokens', 
                     'model.norm', 
                     'lm_head']:
                num_other_flop += v
                
        layer_quant_bop = (sum(np.array(layer_quant_op)* wbitcfg*abitcfg) + sum(np.array(layer_quant_fixed_op) * args.bitW*32))
        
        layer_flop = sum(layer_flop) * 32*32
        
        other_quant_bop = num_other_quant_fixed_op * args.bitW*32
        
        other_flop = num_other_flop * 32*32
        
        quant_bop = int((layer_quant_bop + layer_flop + other_quant_bop + other_flop) / 1e9)

    return quant_bop  # (G)
