import os
import torch
import torch.nn as nn
import numpy as np
import utils.common as utils
from utils.options import args
from utils.load_dict import load_weight
from tensorboardX import SummaryWriter
from importlib import import_module

from data import data_loader

#from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, LlamaConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.config import GPT2Config, GPT2mConfig, LLaMAConfig, LLaMA13bConfig, LLaMA30bConfig, LLaMA65bConfig

from transformers import LlamaForCausalLM, LlamaTokenizer

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import json

from mpq import ompq_search, ptqvit_search, mpl_search
from mpq.utils_ops import param_calculator, bop_calculator

import warnings
warnings.filterwarnings('ignore')

#device = torch.device("cuda", args.local_rank) #device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


# Task: https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7

if args.mgpus:
    dist.init_process_group(backend='nccl')
    dist.barrier()
    world_size = dist.get_world_size()
    device = args.local_rank
    
gpus = [int(gpu) for gpu in args.gpus.split(',')]

if len(gpus) > 1:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    
if gpus[0] != -1:
    device = torch.device(f"cuda:{gpus[0]}")
    cast_device = 'cuda'
else:
    device = 'cpu'  
    cast_device = 'cpu'

 

def main():
    
    best_prec1 = 0.
    best_prec5 = 0.
    
    quant_info_file = os.path.join(args.search_dir, "quant_info.json")
    
    if os.path.exists(quant_info_file):
        with open(quant_info_file) as file:
            # Load its content and make a new dictionary
            quant_info = json.load(file) 
        
        bitcfg = quant_info['layer_bit']
        
        wbitcfg = bitcfg
        abitcfg = bitcfg
        
        if args.fix_bitW == 'True':
            wbitcfg = np.array([args.bitW]*len(bitcfg))
        
        if args.fix_abitW == 'True':
            abitcfg = np.array([args.abitW]*len(bitcfg))
        
        avg_bit, org_model_size, quant_model_size = param_calculator(wbitcfg)
        quant_bop = bop_calculator(wbitcfg, abitcfg)
     
        print_logger.info(f"Original model size: {org_model_size:.2f} (MB)")
        print_logger.info(f"Average bit: {avg_bit:.2f}")
        print_logger.info(f"Quantization model size: {quant_model_size:.2f} (MB)")
        print_logger.info(f"Quantization model BOPs: {quant_bop} (G)")
        
        print_logger.info(f"Layer-wise bits: {bitcfg}")
        
        if args.qmethod == 'mpl_plus':
            bitcfgs = quant_info['layer_bit'], quant_info['layer_qbit'], quant_info['layer_kbit']
            qbitcfg = quant_info['layer_qbit'] 
            kbitcfg = quant_info['layer_kbit'] 
            
            print_logger.info(f"Layer-wise query bits: {qbitcfg}")
            print_logger.info(f"Layer-wise key bits: {kbitcfg}")
    
    else:
        wbitcfg = None
        abitcfg = None
        
        avg_bit, org_model_size, quant_model_size = param_calculator(wbitcfg)
        quant_bop = bop_calculator(wbitcfg, abitcfg)
     
        print_logger.info(f"Original model size: {org_model_size:.2f} (MB)")
        print_logger.info(f"Average bit: {avg_bit:.2f}")
        print_logger.info(f"Quantization model size: {quant_model_size:.2f} (MB)")
        print_logger.info(f"Quantization model BOPs: {quant_bop} (G)")
        
    if args.fix_bitW == 'True':
        print_logger.info(f"Bits (W/A): f{args.bitW}/m{args.abitW}\n")   
    elif args.fix_abitW == 'True':
        print_logger.info(f"Bits (W/A): m{args.bitW}/f{args.abitW}\n")   
    else:
        print_logger.info(f"Bits (W/A): {args.bitW}/{args.abitW}\n")
    
    # Create model
    print('=> Building model...')
    
    ## Load pre-trained model (weights)
    if 'opt' in args.model:
        if args.qmethod == 'rtn':
            model_file_name = 'model.opt_quant'
        else:
            model_file_name = f'model.opt_quant_{args.qmethod}'
        
        if len(gpus) > 1:
            model_file_name = model_file_name.replace('opt', 'pipe_opt')
        
        if args.qmethod in ['mpl_plus']:
            opt_class = import_module(model_file_name).__dict__[args.target_model](args, bitcfgs)
        elif args.qmethod in ['ptqvit', 'minsen', 'mpl', 'mpl_lm']:
            opt_class = import_module(model_file_name).__dict__[args.target_model](args, bitcfg)
        else:
            opt_class = import_module(model_file_name).__dict__[args.target_model](args)
    
        tokenizer = opt_class.tokenizer
        model_t = opt_class.model
    
    elif 'gpt2' in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(f'{args.model}') 
        model = GPT2LMHeadModel.from_pretrained(f'{args.model}') 
        state_dict = model.state_dict()
        del model
        
        ## Create trained model (architecture)
        if args.model == 'gpt2':
            config = GPT2Config()
        elif args.model == 'gpt2-medium':
            config = GPT2mConfig()
        
        if args.qmethod == 'rtn':
            model_file_name = 'model.gpt2_quant'
        else:
            model_file_name = f'model.gpt2_quant_{args.qmethod}'
        
        if args.qmethod in ['mpl_plus']:
            model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfgs)
        elif args.qmethod in ['ptqvit', 'minsen', 'mpl', 'mpl_lm']:
            model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfg)
        else:
            model_t = import_module(model_file_name).__dict__[args.target_model](config)
        
        
        model_t = load_weight(model_t, state_dict)

        del state_dict

            
    elif 'llama' in args.model:
        # pip install bitsandbytes
        # pip install sentencepiece
        
        if args.model == 'llama-7b':
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            '''
            model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
            state_dict = model.state_dict()
            torch.save(state_dict, '/home/ta/research/model/pretrained_llama_7b.pt')
            '''
            
            state_dict = torch.load(os.path.join(f'{args.model_path}', 'pretrained_llama_7b.pt'))
            config = LLaMAConfig()
            
        elif args.model == 'llama-13b':
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")
          
            state_dict = torch.load(os.path.join(f'{args.model_path}', 'pretrained_llama_13b.pt'))
            config = LLaMA13bConfig()
            
        elif args.model == 'llama-30b':
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-30b")
          
            state_dict = torch.load(os.path.join(f'{args.model_path}', 'pretrained_llama_30b.pt'))
            config = LLaMA30bConfig()
        
        elif args.model == 'llama-65b':
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-65b")
       
            state_dict = torch.load(os.path.join(f'{args.model_path}', 'pretrained_llama_65b.pt'))
            config = LLaMA65bConfig()
        
        if args.qmethod == 'rtn':
            model_file_name = 'model.llama_quant'
        else:
            model_file_name = f'model.llama_quant_{args.qmethod}'
        
        if args.qmethod in ['mpl_plus']:
            model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfgs)
        elif args.qmethod in ['ptqvit', 'minsen', 'mpl', 'mpl_lm']:
            model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfg)
        else:
            model_t = import_module(model_file_name).__dict__[args.target_model](config)
        
  
        state_dict_t = dict()
        '''
        for k, v in model_t.state_dict().items():
            if k in state_dict:
                state_dict_t[k] = state_dict[k]
            else:
                state_dict_t[k] = model_t.state_dict()[k]
        '''
        
        for k, v in model_t.state_dict().items():
            state_dict_t[k] = state_dict[k]
            del state_dict[k]
        
        del state_dict
        
        model_t.load_state_dict(state_dict_t)
        
 
        state_dict = dict()
        
        if len(gpus) > 1:
            del model_t
            model_file_name = model_file_name.replace('llama', 'pipe_llama')
        
            if args.qmethod == 'mpl_plus':
                model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfgs, gpus)
            elif args.qmethod in ['ptqvit', 'minsen', 'mpl', 'mpl_lm']:
                model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfg, gpus)
            else:
                model_t = import_module(model_file_name).__dict__[args.target_model](config, gpus)
            
        else:
            if args.qmethod == 'mpl_plus':
                model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfgs)
            elif args.qmethod in ['ptqvit', 'minsen', 'mpl', 'mpl_lm']:
                model_t = import_module(model_file_name).__dict__[args.target_model](config, bitcfg)
            else:
                model_t = import_module(model_file_name).__dict__[args.target_model](config)
            
            
            
        for i, (k, v) in enumerate(model_t.state_dict().items()):
            state_dict[k] = list(state_dict_t.values())[i]
                
        model_t.load_state_dict(state_dict)
        
        del state_dict
        
        del state_dict_t
 
    # Data loading
    print('=> Preparing data..')
    loader = data_loader.Data(args, tokenizer)
    
    
    ## Load pretrained weights
    if args.finetuned == 'True':
        ckpt = torch.load(os.path.join(args.source_file, 'checkpoint/model_best.pt'), map_location = device)
        state_dict = ckpt['state_dict']
        
        state_dict_t = dict()
        
        for k, v in model_t.state_dict().items():
            if k in state_dict:
                state_dict_t[k] = state_dict[k]
            else:
                state_dict_t[k] = v
        model_t.load_state_dict(state_dict_t)
        model_t = model_t.to(device)
        
        del ckpt, state_dict, state_dict_t
        print('=> Finish Loading.')
    
    if args.test_only == 'True':
        if len(gpus) == 1:
            model_t = model_t.to(device)
           
   
        # inference
        print('=> Start inference...')
        
        with torch.autocast("cuda"):
            methods, ppls = eval_ppl(args, loader.loader_test, model_t, tokenizer)
        
        for i in range(len(methods)):
            print_logger.info(f"{methods[i]} Best @ppl: {ppls[i]:.2f}\n")
            
        print('=> Done.')
        return
    
    model_t = model_t.to(device)
    
    # Set optimizer and scheduler
    print('=> Setting optimizer and scheduler...')
    
    optimizer = AdamW(model_t.parameters(), lr=args.lr)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total = -1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = -1)
    
    # Start training
    print('=> Start training...\n')
    for epoch in range(args.num_epochs):
        #loader.datasampler.set_epoch(epoch)
        scheduler.step(epoch)
        
        _ = train(args, loader.loader_train, model_t, tokenizer, optimizer, epoch)
        val_ppl = test(args, loader.loader_validation, model_t, tokenizer)

        is_best = best_ppl > val_ppl
        best_ppl = min(val_ppl, best_ppl)
        #best_prec5 = max(test_prec5, best_prec5)
        
        state = {
            'state_dict': model_t.state_dict(),
            'best_ppl': best_ppl,
            #'best_prec5': best_prec5,
            
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        
        #is_best = True
        checkpoint.save_model(state, epoch + 1, is_best)
        
    print_logger.info(f"Best @ppl: {best_ppl:.2f}")

    

def train(args, loader_train, model_t, tokenizer, optimizer, epoch):
    losses_t = utils.AverageMeter()
    ppl = utils.AverageMeter()

    # switch to train mode
    model_t.train()
        
    num_iterations = len(loader_train)
 
    for i, text in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        text = text.to(device)
        
        if 'opt' in args.model:
            decoder = get_ddp_model(model_t.model.decoder)
            
            outputs = decoder(text)    
            hidden_states = outputs[0]
            
            #print('Calculate logits..')
            lm_head = get_ddp_model(model_t.lm_head)
            logits = lm_head(hidden_states) 
        else:
            ## inference
            model_t = get_ddp_model(model_t)
            logits, _ = model_t(text)  
        
        shift_logits = logits[:, :-1, :]
        shift_labels = text[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape([-1, shift_logits.size(-1)]),
            shift_labels.reshape(-1),
        )                

        losses_t.update(loss.item(), text.size(0))
        
        if i % 4 == 0:
            optimizer.zero_grad()
            
        ## train weights                      
        loss.backward()
        
        losses_t.update(loss.item(), text.size(0))
        
        writer_train.add_scalar('Performance_loss', loss.item(), num_iters)
        
        if i % 4 == 0:
            optimizer.step()

        ## evaluate
        ppl_ = math.exp(losses_t.avg)  
                
        ppl.update(ppl_, text.size(0))
      
        writer_train.add_scalar('Train-ppl', ppl.avg, num_iters)

        ## print
 
        if i % args.print_freq == 0:
            print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Prec@5 {ppl.val:.3f} ({ppl.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    ppl = ppl))
    return ppl.avg

def eval_accuracy(args, loader_test, model_t, tokenizer):

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
 
    # switch to train mode
    model_t.eval()

    
    # text = next(iter(loader.loader_train))
    tmp_text_tens = None
  
    for i, text in enumerate(loader_test, 1):
 
        text_tens = torch.tensor(tokenizer.encode(text[0])).unsqueeze(0).to(device)
        #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if text_tens.size()[1] > args.max_seq_len:
            continue
        
        #The first text sequence in the sequence
        if not torch.is_tensor(tmp_text_tens):
            tmp_text_tens = text_tens
            continue
        else:
            if tmp_text_tens.size()[1] + text_tens.size()[1] > args.max_seq_len:
                work_text_tens = tmp_text_tens
                tmp_text_tens = text_tens
            else:
                tmp_text_tens = torch.cat([tmp_text_tens, text_tens[:,1:]], dim=1)
                continue
                
        
        ## inference
        #print('Decode..')
        if 'opt' in args.model:
            decoder = get_ddp_model(model_t.model.decoder)
            
            outputs = decoder(work_text_tens)    
            hidden_states = outputs[0]
            
            #print('Calculate logits..')
            lm_head = get_ddp_model(model_t.lm_head)
            logits = lm_head(hidden_states)           
        else:
            model_t = get_ddp_model(model_t)
            _, logits = model_t(work_text_tens, lm_labels = work_text_tens) 
        
        ## evaluate
        
        prec1, prec5 = utils.hr(logits, work_text_tens, topk = (1, 5))
        
        top1.update(prec1[0], work_text_tens.size(0))
        top5.update(prec5[0], work_text_tens.size(0))

    return top1.avg, top5.avg

def test(args, loader_test, model_t, tokenizer):

    # switch to train mode
    model_t.eval()

    losses_t = utils.AverageMeter()
    
    for i, text in enumerate(loader_test, 1):
        text = text.to(device)
        
        if 'opt' in args.model:
            decoder = get_ddp_model(model_t.model.decoder)
            
            outputs = decoder(text)    
            hidden_states = outputs[0]
            
            #print('Calculate logits..')
            lm_head = get_ddp_model(model_t.lm_head)
            logits = lm_head(hidden_states) 
        else:
            ## inference
            model_t = get_ddp_model(model_t)
            logits, _ = model_t(text)  
        
        shift_logits = logits[:, :-1, :]
        shift_labels = text[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape([-1, shift_logits.size(-1)]),
            shift_labels.reshape(-1),
        )                

        losses_t.update(loss.item(), text.size(0))
        
    ## evaluate
    ppl = math.exp(losses_t.avg)  
 
      
    return ppl

def eval_ppl(args, loaders_test, model_t, tokenizer):

    # switch to train mode
    model_t.eval()
    
    methods = []
    ppls = []
    
    with torch.no_grad():
        
        for k, loader_test in loaders_test.items():
            losses_t = utils.AverageMeter()
            print(f'==> Method {k}...')
            
            num_iterations = len(loader_test)
            
            for i, text in enumerate(loader_test, 1):
                text = text.to(device)
                
                if 'opt' in args.model:
                    decoder = get_ddp_model(model_t.model.decoder)
                    
                    outputs = decoder(text)    
                    hidden_states = outputs[0]
                    
                    #print('Calculate logits..')
                    lm_head = get_ddp_model(model_t.lm_head)
                    logits = lm_head(hidden_states) 
                else:
                    ## inference
                    model_t = get_ddp_model(model_t)
                    logits, _ = model_t(text)  
                
                shift_logits = logits[:, :-1, :]
                shift_labels = text[:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.reshape([-1, shift_logits.size(-1)]),
                    shift_labels.reshape(-1),
                )                
        
                losses_t.update(loss.item(), text.size(0))
                
                ## print
         
                if i % args.print_freq == 0:
                    print_logger.info(
                            'Epoch[0]({0}/{1}): \n'
                            'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'.format(
                            i, num_iterations, 
                            train_loss = losses_t))
                
            ## evaluate
            ppl = math.exp(losses_t.avg)  
            ppls.append(ppl)
            methods.append(k)
            
        
      
    return methods, ppls

def get_ddp_model(model):
    
    if args.mgpus:
        # parallel
        model = DDP(model, device_ids=[device], 
                    #output_device=args.local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
        
    return model
    
if __name__ == '__main__':
    main()


