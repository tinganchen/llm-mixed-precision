import torch
import numpy as np
import math
import json
from scipy import optimize
from mpq.utils_ops import layer_param_calculator
from scipy.stats import pearsonr

def pearson_corr(a, b):
    a = a.reshape([-1])
    b = b.reshape([-1])
    
    return np.mean(((a - np.mean(a))/np.std(a) * (b - np.mean(b))/np.std(b)))


def mpl0(model_name, model_t, data, target_avg_bit=8):
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # record the scale of q, k
    avg_q_s = []
 
    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            avg_q_s.append(layer.attn.avg_q.view(-1).data.cpu().numpy())
    
        att_corr_distort = np.array([(float(avg_q_s[i])) for i in range(len(avg_q_s))])
        
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu().numpy())
      
        att_corr_distort = np.array([(float(avg_q_s[i])) for i in range(len(avg_q_s))])
        
    else:
        for layer in model_t.model.layers:
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu())
     
        att_corr_distort = np.array([(float(avg_q_s[i])) for i in range(len(avg_q_s))])
        
    
    # re-map
    omega = (-att_corr_distort - np.percentile(-att_corr_distort, 75)) / (np.std(-att_corr_distort))
    
    
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
    

    att_corr_distort = [float(e) for e in list(att_corr_distort)]
    omega = [float(e) for e in list(omega)]
  
    return bitcfg, att_corr_distort, omega

def mpl(model_name, model_t, data, target_avg_bit=8):
    
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # calculate epsilon - quant. error
    eps_a_s = []

    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            eps_a_s.append(layer.eps_a.view(n, -1).data.cpu().numpy())
     
        
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu().numpy())
  
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        
    
    else:
        for layer in model_t.model.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu())
          
        att_corr_distort = np.array([(float(eps_a_s[i])) for i in range(len(eps_a_s))])
    
    # re-map
    omega = (-att_corr_distort - np.percentile(-att_corr_distort, 95)) / np.std(-att_corr_distort)
    
    
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
    

    att_corr_distort = [float(e) for e in list(att_corr_distort)]
    omega = [float(e) for e in list(omega)]
  
    return bitcfg, att_corr_distort, omega

def mpl_lm(model_name, model_t, data, target_avg_bit=8):
    
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # calculate epsilon - quant. error
    eps_a_s = []

    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            eps_a_s.append(layer.eps_a.view(n, -1).data.cpu().numpy())
     
        
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu().numpy())
  
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        
    
    else:
        for layer in model_t.model.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu())
          
        att_corr_distort = np.array([(float(eps_a_s[i])) for i in range(len(eps_a_s))])
    
    # re-map
    omega = (att_corr_distort - np.percentile(att_corr_distort, 75)) / np.std(att_corr_distort)
    
    
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

def mpl_plus0(model_name, model_t, data, target_avg_bit=8):
#with torch.autocast(cast_device):
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # calculate epsilon - quant. error
    eps_a_s = []
    
    # record the scale of q, k
    avg_q_s = []
    avg_k_s = []
    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            eps_a_s.append(layer.eps_a.view(n, -1).data.cpu().numpy())
            avg_q_s.append(layer.attn.avg_q.view(-1).data.cpu().numpy())
            avg_k_s.append(layer.attn.avg_k.view(-1).data.cpu().numpy())

        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        att_avg_q = np.array([np.mean(avg_q_s[i]) for i in range(len(avg_q_s))])
        att_avg_k = np.array([np.mean(avg_k_s[i]) for i in range(len(avg_k_s))])
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu().numpy())
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu().numpy())
            avg_k_s.append(layer.self_attn.avg_k.view(-1).data.cpu().numpy())
       
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        att_avg_q = np.array([np.mean(avg_q_s[i]) for i in range(len(avg_q_s))])
        att_avg_k = np.array([np.mean(avg_k_s[i]) for i in range(len(avg_k_s))])

    else:
        for layer in model_t.model.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu())
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu())
            avg_k_s.append(layer.self_attn.avg_k.view(-1).data.cpu())
       
        att_corr_distort = np.array([(float(eps_a_s[i])) for i in range(len(eps_a_s))])
        att_avg_q = np.array([(float(avg_q_s[i])) for i in range(len(avg_q_s))])
        att_avg_k = np.array([(float(avg_k_s[i])) for i in range(len(avg_k_s))])
    
    
    
    # Allocate layer-wise bits - by min. distortion of attention correlations (after quant.)

    ## re-map
    omega = (-att_corr_distort - np.percentile(-att_corr_distort, 75)) / np.std(-att_corr_distort)
    
    num_layers = len(omega)
    
    ## parameter calculation
    params = layer_param_calculator()
    
    ## objective function
    def func(x, omega=omega, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(x[i] * omega[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def func_deriv(x, omega=omega, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega[i])

        return np.array(diff)

    ## constraint function
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
    
    
    # Allocate layer-wise (q, k) bits - by min. distortion of attention weights (q*k)
    
    ## re-map
    omega_k = (-att_avg_k - np.percentile(-att_avg_k, 50)) / np.std(-att_avg_k)
    
    ## objective function
    def qfunc(q, omega_k=omega_k, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(q[i] * omega_k[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def qfunc_deriv(q, omega_k=omega_k, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega_k[i])

        return np.array(diff)

    ## constraint function
    def qconstrain_func(q, bitcfg=bitcfg):
        """ constraint function """
        return np.array([sum(bitcfg) - sum(q)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((bitcfg[i]-1, bitcfg[i]+1))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': qconstrain_func}
            )

    qresult = optimize.minimize(qfunc, x0=[bitcfg[i] for i in range(num_layers)], jac=qfunc_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    
    qbitcfg = np.around(qresult.x)
    
    
    ## re-map
    omega_q = (-att_avg_q - np.percentile(-att_avg_q, 50)) / np.std(-att_avg_q)
    
    ## objective function
    def kfunc(k, omega_q=omega_q, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(k[i] * omega_q[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def kfunc_deriv(k, omega_q=omega_q, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega_q[i])

        return np.array(diff)

    ## constraint function
    def kconstrain_func(k, bitcfg=bitcfg):
        """ constraint function """
        return np.array([sum(bitcfg) - sum(k)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((bitcfg[i]-1, bitcfg[i]+1))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': kconstrain_func}
            )

    kresult = optimize.minimize(kfunc, x0=[bitcfg[i] for i in range(num_layers)], jac=kfunc_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    
    kbitcfg = np.around(kresult.x)
    
    att_corr_distort = [float(e) for e in list(att_corr_distort)]
    omega = [float(e) for e in list(omega)]
    
  
    return bitcfg, qbitcfg, kbitcfg, att_corr_distort, omega

def mpl_plus(model_name, model_t, data, target_avg_bit=8):
#with torch.autocast(cast_device):
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # calculate epsilon - quant. error
    eps_a_s = []
    
    # record the scale of q, k
    avg_q_s = []
    avg_k_s = []
    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            eps_a_s.append(layer.eps_a.view(n, -1).data.cpu().numpy())
            avg_q_s.append(layer.attn.avg_q.view(-1).data.cpu().numpy())
            avg_k_s.append(layer.attn.avg_k.view(-1).data.cpu().numpy())

        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        att_avg_q = np.array([np.mean(avg_q_s[i]) for i in range(len(avg_q_s))])
        att_avg_k = np.array([np.mean(avg_k_s[i]) for i in range(len(avg_k_s))])
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu().numpy())
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu().numpy())
            avg_k_s.append(layer.self_attn.avg_k.view(-1).data.cpu().numpy())
       
        att_corr_distort = np.array([np.mean(eps_a_s[i]) for i in range(len(eps_a_s))])
        att_avg_q = np.array([np.mean(avg_q_s[i]) for i in range(len(avg_q_s))])
        att_avg_k = np.array([np.mean(avg_k_s[i]) for i in range(len(avg_k_s))])

    else:
        for layer in model_t.model.layers:
            eps_a_s.append(layer.eps_a.view(-1).data.cpu())
            avg_q_s.append(layer.self_attn.avg_q.view(-1).data.cpu())
            avg_k_s.append(layer.self_attn.avg_k.view(-1).data.cpu())
       
        att_corr_distort = np.array([(float(eps_a_s[i])) for i in range(len(eps_a_s))])
        att_avg_q = np.array([(float(avg_q_s[i])) for i in range(len(avg_q_s))])
        att_avg_k = np.array([(float(avg_k_s[i])) for i in range(len(avg_k_s))])
    
    
    
    # Allocate layer-wise bits - by min. distortion of attention correlations (after quant.)

    ## re-map
    omega = (att_corr_distort - np.percentile(att_corr_distort, 75)) / np.std(att_corr_distort)
    
    num_layers = len(omega)
    '''
    ## parameter calculation
    params = layer_param_calculator()
    
    ## objective function
    def func(x, omega=omega, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(x[i] * omega[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def func_deriv(x, omega=omega, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega[i])

        return np.array(diff)

    ## constraint function
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
    '''
    '''Sequential linear-quadratic programming (SLQP)
    https://en.wikipedia.org/wiki/Sequential_linear-quadratic_programming
    
    [12]
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#rdd2e1855725e-12
    '''
    
    #bitcfg = np.array(result.x)
    bitcfg = np.array([float(target_avg_bit-1)]+[float(target_avg_bit)]*(num_layers-1))
    
    
    # Allocate layer-wise (q, k) bits - by min. distortion of attention weights (q*k)
    
    ## re-map
    omega_k = (-att_avg_k - np.percentile(-att_avg_k, 50)) / np.std(-att_avg_k)
    
    ## objective function
    def qfunc(q, omega_k=omega_k, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(q[i] * omega_k[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def qfunc_deriv(q, omega_k=omega_k, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega_k[i])

        return np.array(diff)

    ## constraint function
    def qconstrain_func(q, bitcfg=bitcfg):
        """ constraint function """
        return np.array([sum(bitcfg) - sum(q)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((bitcfg[i]-1, bitcfg[i]+1))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': qconstrain_func}
            )

    qresult = optimize.minimize(qfunc, x0=[bitcfg[i] for i in range(num_layers)], jac=qfunc_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    
    qbitcfg = np.around(qresult.x)
    
    
    ## re-map
    omega_q = (-att_avg_q - np.percentile(-att_avg_q, 50)) / np.std(-att_avg_q)
    
    ## objective function
    def kfunc(k, omega_q=omega_q, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            sum_fuc.append(k[i] * omega_q[i])

        return sum(sum_fuc)

    ## derivative function of objective function
    def kfunc_deriv(k, omega_q=omega_q, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            diff.append(omega_q[i])

        return np.array(diff)

    ## constraint function
    def kconstrain_func(k, bitcfg=bitcfg):
        """ constraint function """
        return np.array([sum(bitcfg) - sum(k)])

    bnds = [] # bit search space: (0.25,0.5) for PTQ and (0.5,1.0) for QAT

    for i in range(num_layers):
        bnds.append((bitcfg[i]-1, bitcfg[i]+1))


    bnds = tuple(bnds)
    cons = ({'type': 'ineq',
             'fun': kconstrain_func}
            )

    kresult = optimize.minimize(kfunc, x0=[bitcfg[i] for i in range(num_layers)], jac=kfunc_deriv, method='SLSQP', bounds=bnds, constraints=cons)
    
    kbitcfg = np.around(kresult.x)
    
    att_corr_distort = [float(e) for e in list(att_corr_distort)]
    omega = [float(e) for e in list(omega)]
    
    return bitcfg, qbitcfg, kbitcfg, att_corr_distort, omega