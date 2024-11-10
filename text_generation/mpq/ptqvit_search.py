import torch
import numpy as np
import math
import json
from scipy import optimize
from mpq.utils_ops import layer_param_calculator


def ptqvit(model_name, model_t, data, target_avg_bit=8):
    
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # calculate epsilon - quant. error
    eps_a_s = []
    eps_m_s = []
    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            eps_a_s.append(layer.eps_a.view(n, -1).data.cpu().numpy())
            eps_m_s.append(layer.eps_m.view(n, -1).data.cpu().numpy())
        
        omega = np.array([np.mean(eps_a_s[i] + eps_m_s[i]) for i in range(len(eps_a_s))])
        
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu().numpy())
            eps_m_s.append(layer.eps_m.view(-1).data.cpu().numpy())
        
        omega = np.array([np.mean(eps_a_s[i] + eps_m_s[i]) for i in range(len(eps_a_s))])
        
    
    else:
        for layer in model_t.model.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu())
            eps_m_s.append(layer.eps_m.view(-1).data.cpu())
    
    
        omega = np.array([(float(eps_a_s[i] + eps_m_s[i])/2) for i in range(len(eps_a_s))])
    
    '''
    num_layers = len(omega)
    
    # parameter calculation
    params = layer_param_calculator()
    
    # Objective function
    def func(x, omega=omega, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append( x[i] * omega[i] )

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
        """ constrain function """
        a = []
        for i in range(len(params)):
            a.append(x[i] * params[i])
        return np.array([target_avg_bit*sum(params) - sum(a)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((0.25, 0.5))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': constrain_func}
            )

    result = optimize.minimize(func,x0=[1 for i in range(num_layers)], jac=func_deriv, method='SLSQP', bounds=bnds, constraints=cons)

    bitcfg = np.around(result.x*2*target_avg_bit)
 
    '''
    omega_rank = np.argsort(omega)
    
    min_omega = np.min(omega_rank)
    max_omega = np.max(omega_rank)
    
    min_tgt_bit = target_avg_bit - 1 #target_avg_bit / 4
    max_tgt_bit = target_avg_bit + 1 #target_avg_bit / 4

    bitcfg = np.round((omega_rank - min_omega) / (max_omega - min_omega) * (max_tgt_bit - min_tgt_bit) + min_tgt_bit)
    
    
    return bitcfg


