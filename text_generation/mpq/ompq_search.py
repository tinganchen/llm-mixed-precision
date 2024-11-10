import torch
import numpy as np
import math
import json
from scipy import optimize
from mpq.utils_ops import layer_param_calculator

def sum_list(a, j):
    b = 0
    for i in range(len(a)):
        if i != j:
            b += a[i]
    return b

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    return x.dot(x.T)


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
    #if not np.allclose(gram, gram.T):
    #    raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def orm(gram_x, gram_y, debiased=False):
    """Compute ORM.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of ORM between X and Y.
  """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


# def feature_space_orm(features_x, features_y, debiased=False):
#     """Compute ORM with a linear kernel, in feature space.
#
#   This is typically faster than computing the Gram matrix when there are fewer
#   features than examples.
#
#   Args:
#     features_x: A num_examples x num_features matrix of features.
#     features_y: A num_examples x num_features matrix of features.
#     debiased: Use unbiased estimator of dot product similarity. ORM may still be
#       biased. Note that this estimator may be negative.
#
#   Returns:
#     The value of ORM between X and Y.
#   """
#     features_x = features_x - torch.mean(features_x, 0, keepdim=True)
#     features_y = features_y - torch.mean(features_y, 0, keepdim=True)
#
#     a = torch.mm(features_x.t(), features_y)
#     b = torch.mm(features_x.t(), features_x)
#     c = torch.mm(features_y.t(), features_y)
#     dot_product_similarity = torch.linalg.norm(a) ** 2
#     normalization_x = torch.linalg.norm(b)
#     normalization_y = torch.linalg.norm(c)
#
#     if debiased:
#         n = features_x.shape[0]
#         # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
#         sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
#         sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
#         squared_norm_x = np.sum(sum_squared_rows_x)
#         squared_norm_y = np.sum(sum_squared_rows_y)
#
#         dot_product_similarity = _debiased_dot_product_similarity_helper(
#             dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
#             squared_norm_x, squared_norm_y, n)
#         normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
#             squared_norm_x, squared_norm_x, n))
#         normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
#             squared_norm_y, squared_norm_y, n))
#
#     return dot_product_similarity / (normalization_x * normalization_y)


def feature_space_orm(features_x, features_y, debiased=False):
    """Compute ORM with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. ORM may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of ORM between X and Y.
  """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)

def ompq(model_name, model_t, data, target_avg_bit=8):
    
    logits, _ = model_t(data)
    
    n = data.size()[0]
    
    # feature retrieval
    feature = []
    
    if 'gpt2' in model_name:
        for layer in model_t.transformer.h:
            feature.append(layer.x.view(n, -1).data.cpu().numpy())
    
    elif 'opt' in model_name:
        for layer in model_t.model.decoder.layers:
            feature.append(layer.x.view(n, -1).data.cpu().numpy())
    
    else:
        for layer in model_t.model.layers:
            feature.append(layer.x.view(n, -1).data.cpu().numpy())
    
    # orm matrix
    orthogonal_matrix = np.zeros((len(feature), len(feature)))

    for i in range(len(feature)):
        for j in range(len(feature)):
            with torch.no_grad():
                orthogonal_matrix[i][j] = orm(gram_linear(feature[i]), 
                                              gram_linear(feature[j]))

    # importance score
    theta = []
    gamma = []

    for i in range(len(feature)):
        gamma.append( sum_list(orthogonal_matrix[i], i) )

    # e^-x
    for i in range(len(feature)):
        theta.append( 1 * math.exp(-1* gamma[i]) )
    theta = np.array(theta)
    theta = np.negative(theta)
    

    
    num_layers = len(feature)
    
    # parameter calculation
    params = layer_param_calculator()

    # Objective function
    def func(x, sign=1.0, theta=theta, num_layers=num_layers):
        """ Objective function """
        sum_fuc =[]
        for i in range(num_layers):
            temp = 0.
            for j in range(i,num_layers):
                temp += theta[j]
            sum_fuc.append( x[i] * (sign * temp / (num_layers-i)) )

        return sum(sum_fuc)

    # Derivative function of objective function
    def func_deriv(x, sign=1.0, theta=theta, num_layers=num_layers):
        """ Derivative of objective function """
        diff = []
        for i in range(num_layers):
            temp1 = 0.
            for j in range(i, num_layers):
                temp1 += theta[j]
            diff.append(sign * temp1 / (num_layers - i))

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

    bitcfg = np.around(result.x * 8)
    
    return bitcfg
