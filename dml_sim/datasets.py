import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.datasets import make_spd_matrix
from doubleml import DoubleMLData, DoubleMLClusterData
from scipy.special import expit

_array_alias = ['array', 'np.ndarray', 'np.array', np.ndarray]
_data_frame_alias = ['DataFrame', 'pd.DataFrame', pd.DataFrame]
_dml_data_alias = ['DoubleMLData', DoubleMLData]

def make_irm_friedman(n_obs=500, dim_x=20, alpha=None, return_type='DoubleMLData', **kwargs):
    """
    add comments
    """
    x = np.random.uniform(0, 1, (n_obs, dim_x))
    px = expit(x[:, 0] + x[:, 1] - 0.5)
    d = np.random.binomial(n=1, p=px, size = n_obs)
    
    g = 10*np.sin(np.pi*x[:, 1]*x[:, 2]) + 20*(x[:, 3]-0.5)**2+10*x[:, 4] + 5*x[:, 5]
    
    theta = x[:, 3]*np.cos(np.pi*x[:, 1]*x[:, 2])
    
    y = theta * d + g + np.random.standard_normal(size=[n_obs, ])
    

    if return_type in _array_alias:
        return x, y, d, theta
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')



from functools import reduce
from operator import mul

def make_plr_fingerhut2018(n_obs=500, dim_x=20, theta=1, return_type='DoubleMLData', **kwargs):
    """
    add comments
    """
    rrho = kwargs.get('rho', 0.8)
    majority_s = kwargs.get('majority_s', 0.75)

    mu = np.zeros(dim_x,)
    rho = np.ones(dim_x,)*rrho
    sigma = np.zeros(shape=(dim_x, dim_x))
    for i in range(dim_x):
        for j in range(i, dim_x):
            sigma[i][j] = reduce(mul, [rho[k] for k in range(i, j)], 1)
            
    sigma = np.triu(sigma) + np.triu(sigma).T - np.diag(np.diag(sigma))
    x = np.random.multivariate_normal(mu, sigma, n_obs)
    z = np.argsort(x[:, 0])
    threshold_idx = z[int(n_obs * majority_s)]
    threshold_val = x[threshold_idx, 0]
    majority = x[:, 0] < threshold_val
    minority = x[:, 0] >= threshold_val
    m0 = np.zeros(x.shape[0], dtype=np.float64)
    m0[majority] = x[majority, 1] + 10 * x[majority, 3] + 5 * x[majority, 6]
    m0[minority] = 10 * x[minority, 1] + x[minority, 3] + 5 * x[minority, 6]

    g0 = np.zeros(x.shape[0], dtype=np.float64)
    g0[majority] = x[majority, 0] + 10 * x[majority, 2] + 5 * x[majority, 5]
    g0[minority] = 10 * x[minority, 0] + x[minority, 2] + 5 * x[minority, 5]
    
    d = m0 + np.random.standard_normal(size=[n_obs,])
    y = d * theta + g0 + np.random.standard_normal(size=[n_obs,])

    if return_type in _array_alias:
        return x, y, d
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')




def make_irm_farell2021(n_obs=500, dim_x=20, alpha=None, return_type='DoubleMLData', **kwargs):
    """
    add comments
    """
    x = np.random.uniform(0, 1, (n_obs, dim_x))
    px = expit(x[:, 0] + x[:, 1] - 0.5)
    d = np.random.binomial(n=1, p=px, size = n_obs)
    
    g = 10*np.sin(np.pi*x[:, 1]*x[:, 2]) + 20*(x[:, 3]-0.5)**2+10*x[:, 4] + 5*x[:, 5]
    
    theta = x[:, 3]*np.cos(np.pi*x[:, 1]*x[:, 2])
    
    y = theta * d + g + np.random.standard_normal(size=[n_obs, ])
    

    if return_type in _array_alias:
        return x, y, d, theta
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'X{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return data
        else:
            return DoubleMLData(data, 'y', 'd', x_cols)
    else:
        raise ValueError('Invalid return_type.')
