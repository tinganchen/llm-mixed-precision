import torch
import torch.nn as nn
import numpy as np
import os
import math
import json
from scipy import optimize
from mpq.utils_ops import layer_param_calculator
from utils.options import args

#from backpack import backpack, extend
#from backpack.extensions import HMP, DiagHessian
#from backpack.hessianfree.hvp import hessian_vector_product
#from backpack.utils.examples import load_one_batch_mnist
# pip install backpack-for-pytorch

gpus = [int(gpu) for gpu in args.gpus.split(',')]


if gpus[0] != -1:
    device = torch.device(f"cuda:{gpus[0]}")
    cast_device = 'cuda'
else:
    device = 'cpu'  
    cast_device = 'cpu'
    
def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      #grad_input = grad_output
      return grad_output

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, bitW):
    super(weight_quantize_fn, self).__init__()
    #assert bitW <= 8 or bitW == 32
    self.bitW = bitW
    self.uniform_q = uniform_quantize(k=bitW)

  def forward(self, x):
    if self.bitW == 1:
      E = torch.mean(torch.abs(x)).detach()
      x = self.uniform_q(x / E) * E
    else:
      x = self.uniform_q(x)
    return x


def rademacher(shape, dtype=torch.float32, device=device):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)

def hutchinson_trace_hmp(param, V, V_batch=1):
    """Hessian trace estimate using BackPACK's HMP extension.

    Perform `V_batch` Hessian multiplications at a time.
    """
    V_count = 0
    trace = 0

    while V_count < V:
        V_missing = V - V_count
        V_next = min(V_batch, V_missing)

        v = rademacher((V_next, *param.shape))
        Hv = param.hmp(v).detach()
        vHv = torch.einsum("i,i->", v.flatten(), Hv.flatten().detach())
        trace += vHv / V

        V_count += V_next

    return trace

def minsen(model_name, model_t, data, target_avg_bit=8):

    with torch.autocast(cast_device):
        if 'opt' in model_name:
            decoder = model_t.model.decoder
            
            outputs = decoder(data)    
            hidden_states = outputs[0]
            
            #print('Calculate logits..')
            lm_head = model_t.lm_head
            logits = lm_head(hidden_states) 
        else:
            ## inference
            model_t = model_t
            logits, _ = model_t(data)  
        
        shift_logits = logits[:, :-1, :]
        shift_labels = data[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape([-1, shift_logits.size(-1)]),
            shift_labels.reshape(-1),
        )                
    
    loss.backward()
    
    quantize_fn = weight_quantize_fn(bitW=args.bitW)
    
    # calculate epsilon - quant. error with hessian
    omega = []
    
    if 'gpt2' in model_name:
        for n, p in model_t.named_parameters():
            if 'attn.c_proj' in n and 'weight' in n:
                qerr = -torch.sum(p.grad**2 * (quantize_fn(p) - p)**2).data.cpu()
                omega.append(qerr)
                
        omega = np.array(omega)
        
    
    elif 'opt' in model_name:
        for n, p in model_t.named_parameters():
            if 'self_attn.out_proj.weight' in n:
                qerr = -torch.sum(p.grad**2 * (quantize_fn(p) - p)**2).data.cpu()
                omega.append(qerr)
                
        omega = np.array(omega)
        
    
    else:
        for n, p in model_t.named_parameters():
            if 'self_attn.o_proj' in n and 'weight' in n:
                qerr = -torch.sum(p.grad**2 * (quantize_fn(p) - p)**2).data.cpu()
                omega.append(qerr)
                
        omega = np.array(omega)
    '''
    omega_rank = np.argsort(omega)
    
    min_omega = np.min(omega_rank)
    max_omega = np.max(omega_rank)
    
    min_tgt_bit = target_avg_bit - 1 #target_avg_bit / 4
    max_tgt_bit = target_avg_bit #target_avg_bit / 4

    bitcfg = np.round((omega_rank - min_omega) / (max_omega - min_omega) * (max_tgt_bit - min_tgt_bit) + min_tgt_bit)
    '''
    
    # re-map
    omega = (-omega - np.percentile(-omega, 95)) / np.std(-omega)
    
    
    # optimize

    num_layers = len(omega)
    
    # parameter calculation
    params = layer_param_calculator()
    
    # Objective function
    def func(x, omega=omega, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(x[i] * omega[i])

        return sum(sum_fuc)

    # Derivative function of objective function
    def func_deriv(x, omega=omega, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega[i])

        return np.array(diff)

    # Constraint function
    def constrain_func(x, params=params):
        """ constraint function """
        a = []
        for i in range(len(params)):
            a.append(x[i] * params[i])
        return np.array([target_avg_bit*sum(params) - sum(a)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((target_avg_bit-1, target_avg_bit))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': constrain_func}
            )

    result = optimize.minimize(func, x0=[target_avg_bit-1 for i in range(num_layers)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    
    '''Sequential linear-quadratic programming (SLQP)
    https://en.wikipedia.org/wiki/Sequential_linear-quadratic_programming
    
    [12]
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#rdd2e1855725e-12
    '''
    
    bitcfg = np.around(result.x)
    
    return bitcfg


