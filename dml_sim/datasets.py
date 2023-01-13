import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.datasets import make_spd_matrix
from doubleml import DoubleMLData, DoubleMLClusterData
from scipy.special import expit

#for make_plr_fingerhut2018
from functools import reduce
from operator import mul

_array_alias = ['array', 'np.ndarray', 'np.array', np.ndarray]
_data_frame_alias = ['DataFrame', 'pd.DataFrame', pd.DataFrame]
_dml_data_alias = ['DoubleMLData', DoubleMLData]

def make_irm_friedman(n_obs=500, dim_x=20, return_type='DoubleMLData', **kwargs):
    """  
    Generates data from a five dimensional test function which was proposed to assess 
    predictive performance of methods and then extended to evaluate causal inference 
    methods for some highly non-linear outcome (Parikh et al. 2022)
    
    References:
        - Friedman, Jerome H., Eric Grosse, and Werner Stuetzle (June 1983). “Multidimensional
          Additive Spline Approximation”. In: SIAM Journal on Scientific and Statistical
          Computing 4.2, pp. 291–301. doi: 10.1137/0904023. 
        - Friedman, Jerome H. (1991). “Multivariate Adaptive Regression Splines”. In: The
          Annals of Statistics 19.1. Publisher: Institute of Mathematical Statistics, pp. 1–67. doi:
          10.1214/aos/1176347963.
        - Parikh, Harsh et al. (June 2022). “Validating Causal Inference Methods”. In: Proceedings
          of the 39th International Conference on Machine Learning. ISSN: 2640-3498. PMLR,
          pp. 17346–17358. url: https://proceedings.mlr.press/v162/parikh22a.html
    
    Arguments
    ----------
    n_obs : int
        The number of observations to generate.
    dim_x : int
        The number of features in the input data.
    return_type : str
        The format in which to return the data. Can be one of 'DoubleMLData', 'dataframe', or 'array'.
    **kwargs : dict
        Additional keyword arguments to pass to the function.
        
    Returns
    -------
    x, y, d, theta : numpy array
        The generated input data, output variable, treatment variable, and variable for the heterogeneous treatment effect, respectively.
        Returned if return_type='array'.
    (data, theta) : tuple
        A tuple containing pandas DataFrame of the generated data with columns for input variables, output variable, and treatment variable
        and variable for the heterogeneous treatment effect.
        Returned if return_type='dataframe'.
    (data, theta) : tuple
        A tuple containing DoubleMLData object containing the generated data with columns for input variables,output variable, treatment variable
        and variable for the heterogeneous treatment effect.
        Returned if return_type='DoubleMLData'.
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
            return (data, theta)
        else:
            return (DoubleMLData(data, 'y', 'd', x_cols), theta)
    else:
        raise ValueError('Invalid return_type.')

        




def make_irm_farell2021(n_obs=500, dim_x=20, return_type='DoubleMLData', **kwargs):
    
    """
    Generates data based on a non-linear model for treatment assignment dependent on the covariates as used
    in Farrell, Liang, and Misra 2021.
   
    References:
        - Farrell, Max H., Tengyuan Liang, and Sanjog Misra (2021). “Deep Neural Networks for Esti-
          mation and Inference”. en. In: Econometrica 89.1. doi: 10.3982/ECTA16901. 
        - Code for data generation available online.

    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    dim_x : int
        The number of features in the input data.
    return_type : str
        The format in which to return the data. Can be one of 'DoubleMLData', 'dataframe', or 'array'.
    **kwargs : dict
        Additional keyword arguments to pass to the function.
        
    Returns
    -------
    x, y, d, theta : numpy array
        The generated input data, output variable, treatment variable, and variable for the heterogeneous treatment effect, respectively.
        Returned if return_type='array'.
    (data, theta) : tuple
        A tuple containing pandas DataFrame of the generated data with columns for input variables, output variable, and treatment variable
        and variable for the heterogeneous treatment effect.
        Returned if return_type='dataframe'.
    (data, theta) : tuple
        A tuple containing DoubleMLData object containing the generated data with columns for input variables,output variable, treatment variable
        and variable for the heterogeneous treatment effect.
        Returned if return_type='DoubleMLData'.
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
            return (data, theta)
        else:
            return (DoubleMLData(data, 'y', 'd', x_cols), theta) 
    else:
        raise ValueError('Invalid return_type.')
