import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.datasets import make_spd_matrix
from doubleml import DoubleMLData, DoubleMLClusterData
from scipy.special import expit

# for make_irm_farell2021
from scipy.special import comb
from itertools import combinations_with_replacement

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

        
def make_plr_fingerhut2018(n_obs=500, dim_x=20, theta=1, return_type='DoubleMLData', **kwargs):
    """ 
    Generates data from a PLR model as used in Fingerhut, Sesia, and Romano 2022.
    
    References: 
        - Fingerhut, Nitai, Matteo Sesia, and Yaniv Romano (June 2022). "Coordinated Dou-
          ble Machine Learning".  doi: 10.48550/arXiv.2206.00885. 
        - Code for data generation available online. Link to GitHub Repository: 
          https://github.com/nitaifingerhut/C-DML/tree/9855dd8c7f6fcff0b082822f0d9dd355573715a7
    
    Parameters
    ----------
    n_obs : int
        The number of observations to generate.
    dim_x : int
        The number of features in the input data.
    theta : float
        The true value of the treatment effect. If None, a default value will be used.
    return_type : str
        The format in which to return the data. Can be one of 'DoubleMLData', 'dataframe', or 'array'.
    **kwargs : dict
        Additional keyword arguments to pass to the function.
        - rho : float
            correlation parameter
        - majority_s : float
            threshold to divide samples into two disjoint groups 
        
    Returns
    -------
    x, y, d : numpy array
        The generated input data, output variable, and treatment variable, respectively.
        Returned if return_type='array'.
    data : pandas DataFrame
        The generated data with columns for input variables, output variable, and treatment variable.
        Returned if return_type='dataframe'.
    data : DoubleMLData
        A wrapper object containing the generated data with columns for input variables, output variable, and treatment variable.
        Returned if return_type='DoubleMLData'.
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

    i = 0
    beta_theta_phi_X = 0
    beta_g_phi_X = 0
    bias_theta = -0.05 #alpha_{theta,1}
    bias_g = 0.09 # alpha_{g,1}
    bias_p = 0.09 # alpha_{p,1}
    count = comb(dim_x, 2, True, True)

    # Following vectors are generated with a fixed Random State for each simulation run
    # Resulting in same vectors for each repetition according to Farell et al.
    r = np.random.RandomState(1234)
    alpha_theta = r.uniform(low=0.1, high=0.22, size=[dim_x, 1])
    alpha_g = r.normal(loc=0.3, scale=0.7, size=[1, dim_x])
    beta_theta = r.uniform(low=-0.05, high=0.06, size=count)
    beta_g = r.normal(loc=0.01, scale=0.3, size=count)

    alpha_p = np.random.uniform(low=-0.55, high=0.55, size=[20, 1])
    x = np.random.uniform(low=0, high=1, size=[n_obs, dim_x])
    normal_errors = np.random.normal(size=[n_obs, 1], loc=0.0, scale=1.0)
    alpha_theta_X = np.dot(x, alpha_theta) + bias_theta
    # Second Degree polynomial (Phi score)
    polynomial_indices = combinations_with_replacement(list(range(dim_x)), 2)
    for p in polynomial_indices:
        beta_theta_phi_X += beta_theta[i]*np.multiply(x[:, p[0]], x[:, p[1]])
        beta_g_phi_X += beta_g[i]*np.multiply(x[:, p[0]], x[:, p[1]])
        i += 1
    beta_theta_phi_X = beta_theta_phi_X.reshape(-1, 1)
    beta_g_phi_X = beta_g_phi_X.reshape(-1,1)
    theta = alpha_theta_X + beta_theta_phi_X
    g = np.dot(x, alpha_g.T) + bias_g +  beta_g_phi_X
    alpha_p_X = np.dot(x[:, :20], alpha_p) + bias_p
    alpha_p_X = alpha_p_X.reshape(-1)
    prob_of_d = expit(alpha_p_X)
    d = np.random.binomial(size=n_obs, n=1, p=prob_of_d).reshape(n_obs,)
    theta = theta.reshape(-1)
    y = g.reshape(-1) + theta*d + normal_errors.reshape(-1)

    if return_type in _array_alias:
        return x, y, d, theta
    elif return_type in _data_frame_alias + _dml_data_alias:
        x_cols = [f'x{i + 1}' for i in np.arange(dim_x)]
        data = pd.DataFrame(np.column_stack((x, y, d)),   
                            columns=x_cols + ['y', 'd'])
        if return_type in _data_frame_alias:
            return (data, theta)
        else:
            return (DoubleMLData(data, 'y', 'd', x_cols), theta) 
    else:
        raise ValueError('Invalid return_type.')