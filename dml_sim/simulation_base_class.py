import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openpyxl
import doubleml
import sklearn
import tqdm

class simulation_study:
  """
  Initialize a simulation study object.
        
  Arguments
  ----------
      model (:class:`DoubleML` object): A DoubleML model (DoubleMLPLR and DoubleMLIRM are supported).
      is_heterogenous (bool): A boolean indicating whether the treatment effect is
          heterogenous (True, for IRM) or homogenous (False, for PLR).
      score (string): A string correspondending to the DoubleML model.
          For PLR use 'partialling out' or 'IV-type' and for IRM use 'ATE' oder 'ATTE'.
      DGP (callable): A callable data generating process. The callable should take at least 'n_obs', 
          'dim_x' as arguments. 'alpha' / 'theta' is necessary for some DGPs with non heterogenous
          treatment effect. The callable should return numpy arrays:
              X, dim(X) = (n_obs, dim_x)
              y, dim(y) = (n_obs,)
              d, dim(d) = (n_obs,)
            and
              theta, dim(theta) = (n_obs,)
            in order to calculate the average treatment effect if the treatment effect is heterogenous.
            In some cases, the DGP (doubleml.datasets.make_irm_data for example) generates a heterogenous
            from an argument with fixed value for theta. Initialize your instance with alpha choosen as None.
      n_rep (int): The number of replications to be run in the simulation study.
      np_dict (Dict[str, List[int]]): A dictionary mapping values for 'n_obs' and 'dim_x'. 
          Simulation will run over all permutations of 'n_obs' x 'dim_x'.
      lrn_dict (Dict[str, object]): A dictionary mapping learner names to objects
          representing the learners to be used in the simulation study. Each objects
          should have at least two methods: A .fit method that takes 
          X (NumPy array of shape (n_obs, dim_x)) and y (NumPy array of shape (n_obs,)).
          And a .predict method that takes X (NumPy array of shape (n_obs, dim_x)).
          The following structure is expected:
          {'learner_group1': {
              'ml_l' : learner,
              'ml_m' : learner,
              'ml_g' : learner},
           'learner_group2': {
              'ml_l' : learner,
              'ml_m' : learner,
              'ml_g' : learner}}
           Where 'learner' is an object with a fit and predict method
           in style of scikit-learn and 'learner_group' are keywords (str)
           to identify the type of the grouped 'learner'.
           If working with torch neural networks / skorch as sklearn learner,
           'learner_groupX' have to be defined as 'neural_net'.
       alpha (float or None): The value of the true treatment effect to be used in the study.
        
  Attributes
  ----------
      theta_0 (float or Dict[Tuple[int, int], float]): The true value of the parameter
          being estimated. If the treatment effect is heterogenous and alpha is not
          specified, this will be a dictionary mapping settings (as tuples of ints) to
          the true value of the parameter for that setting.
      n_folds (int): The number of folds to be used in DoubleMLs cross fitting. Defaults to 3.
      _data (Tuple[NumPy array, NumPy array, NumPy array] or None): The data ((x, y, d)) 
          generated by the DGP for the actual setting. This is initialized as None.
      _i_rep (int): The current replication number. This is initialized as 0.
      _theta_dml_i (NumPy array of shape (n_rep,)): An array to store the estimates
          of the parameter being estimated for each replication. This is initialized as
          a NumPy array of zeros.
      _se_dml_i (NumPy array of shape (n_rep,)): An array to store the standard errors
          of the estimates of the parameter being estimated for each replication. This
          is initialized as a NumPy array of zeros.
      _lowCI_i (NumPy array of shape (n_rep,)): An array to store the lower bounds of
          the confidence intervals for the estimates of the parameter being estimated
          for each replication. This is initialized as a NumPy array of zeros.
      _upCI_i (NumPy array of shape (n_rep,)): An array to store the upper bounds of
          the confidence intervals for the estimates of the parameter being estimated
          for each replication. This is initialized as a NumPy array of zeros.
      theta_dml (Dict[str, Dict[Tuple[int, int], NumPy array of shape (n_rep,)]]): 
          A dictionary mapping learner names to dictionaries that map settings (as tuples of ints)
          to arrays of estimates of the parameter being estimated for each replication.
          This is initialized as an empty dictionary with keys for each learner.        
      se_dml (Dict[str, Dict[Tuple[int, int], NumPy array of shape (n_rep,)]]): A
          dictionary mapping learner names to dictionaries that map settings (as tuples
          of ints) to arrays of standard errors for the estimates of the parameter being
          estimated for each replication. This is initialized as an empty dictionary
          with keys for each learner.
      lowCI (Dict[str, Dict[Tuple[int, int], NumPy array of shape (n_rep,)]]): A
          dictionary mapping learner names to dictionaries that map settings (as tuples
          of ints) to arrays of lower bounds of the confidence intervals for the
          estimates of the parameter being estimated for each replication. This is
          initialized as an empty dictionary with keys for each learner.
      upCI (Dict[str, Dict[Tuple[int, int], NumPy array of shape (n_rep,)]]): A
          dictionary mapping learner names to dictionaries that map settings (as tuples
          of ints) to arrays of upper bounds of the confidence intervals for the estimates of
          the parameter being estimated for each replication. This is initialized as an
          empty dictionary with keys for each learner.
      _all_permutations (List[Tuple[int, int]]): A list of all possible settings (as
          tuples of ints) to be used in the simulation study. This is initialized as an
          empty list.
      _n_obs_act (int or None): The number of observations being used in the current
          replication. This is initialized as None.
      _dim_x_act (int or None): The number of features / dimension of X being used in the current
          replication. This is initialized as None.
      _lrn_act (str or None): The name of the learner being used in the current
          session. This is initialized as None.
      _seed (int): The seed to be used for the NumPy random number generator. This is
          initialized as 1234.
      abs_bias (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner
          names to dictionaries that map settings (as tuples of ints) to the average absolute
          bias of the estimates of the parameter being estimated for each replication.
          This is initialized as an empty dictionary with keys for each learner.
      rel_bias (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner
          names to dictionaries that map settings (as tuples of ints) to the average relative
          bias of the estimates of the parameter being estimated for each replication.
          This is initialized as an empty dictionary with keys for each learner.
      std_bias (Dict[str (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner names
          to dictionaries that map settings (as tuples of ints) to the average standardized
          bias of the estimates of the parameter being estimated for each replication.
          This is initialized as an empty dictionary with keys for each learner.
      rmse (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner
          names to dictionaries that map settings (as tuples of ints) to the root mean
          squared error of the estimates of the parameter being estimated for each
          replication. This is initialized as an empty dictionary with keys for each
          learner.
      avg_se (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner
          names to dictionaries that map settings (as tuples of ints) to the average
          standard error of the estimates of the parameter being estimated for each
          replication. This is initialized as an empty dictionary with keys for each
          learner.
      empdev (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping learner
          names to dictionaries that map settings (as tuples of ints) to the empirical
          deviation of the estimates of the parameter being estimated for each
          replication. This is initialized as an empty dictionary with keys for each
          learner. If the number of replications is less than 2, this attribute will
          not be calculated.
      coverage (Dict[str, Dict[Tuple[int, int], float]]): A dictionary mapping
          learner names to dictionaries that map settings (as tuples of ints) to the
          coverage of the confidence intervals of the estimates of the parameter being
          estimated for each replication. This is initialized as an empty dictionary
          with keys for each learner.
      performance_df (Pandas DataFrame or None): A DataFrame containing the results of
          the performance measures for each learner and setting (as tuples of ints). 
          The rows are indexed by a tuple containing the learner
          name and setting, and the columns are the names of the performance measures.
          This is initialized as None.
      histograms (Dict[str, Dict[Tuple[int, int], Matplotlib Axes]]): A dictionary
          mapping learner names to dictionaries that map settings (as tuples of ints) to
          histograms of the estimates of the parameter being estimated for each
          replication. This is initialized as an empty dictionary with keys for each
          learner.
      boxplots (Dict[str, Dict[Tuple[int, int], Matplotlib Axes]]): A dictionary
          mapping learner names to dictionaries that map settings (as tuples of ints) to
          boxplots of the estimates of the parameter being estimated for each
          replication. This is initialized as an empty dictionary with keys for each
          learner.

  Methods
  -------
      _prepare_data(setting)
        Prepare the n_rep datasets for a given n and dim_x setting.

      _create_dml_data_obj()
        Create and return a :class: `DoubleMLData` object from arrays.
        Returns obj_dml_data

      _create_dml_object(obj_dml_data)
        Create and return a :class: `DoubleML` object with the given 
        :class: `DoubleMLData` obj_dml_data object.
        Returns obj_dml_model

      _run_fit()
        Performs a fit of a :class: `DoubleML` object and saves the results.

      run_simulation()
        Performs the full simulation for all learners and settings.

      [...]
  """

  def __init__(self, model, is_heterogenous, score, DGP, n_rep, np_dict, lrn_dict, alpha):
    self.model = model
    self.score = score
    self.n_folds = 3 #default
    self.DGP = DGP
    self.n_rep = n_rep
    self.np_dict = np_dict
    self.lrn_dict = lrn_dict
    self.is_heterogenous = is_heterogenous
    self.alpha = alpha
    self._data = None
    self._i_rep = 0 #initial
    self._theta_dml_i = np.zeros(shape=(n_rep,))
    self._se_dml_i = np.zeros(shape=(n_rep,))
    self._lowCI_i = np.zeros(shape=(n_rep,))
    self._upCI_i = np.zeros(shape=(n_rep,))
    self.theta_dml = {key: {} for key in self.lrn_dict.keys()}
    self.se_dml = {key: {} for key in self.lrn_dict.keys()}
    self.lowCI = {key: {} for key in self.lrn_dict.keys()}
    self.upCI = {key: {} for key in self.lrn_dict.keys()}
    self._all_permutations = []
    self._n_obs_act = None
    self._dim_x_act = None
    self._lrn_act = None
    self._seed = 1234
    self.abs_bias = {key: {} for key in self.lrn_dict.keys()}
    self.rel_bias = {key: {} for key in self.lrn_dict.keys()}
    self.std_bias = {key: {} for key in self.lrn_dict.keys()}
    self.rmse = {key: {} for key in self.lrn_dict.keys()}
    self.avg_se = {key: {} for key in self.lrn_dict.keys()}
    self.empdev = {key: {} for key in self.lrn_dict.keys()}
    self.coverage = {key: {} for key in self.lrn_dict.keys()}
    self.performance_df = None
    self.histograms = {}
    self.boxplots = {}

    np.random.seed(self._seed)

    if self.alpha is not None and self.is_heterogenous:
      print('Warning: Instance is initialized with a specified\nvalue for alpha and heterogenous treatment effect.')
      print('Please make sure that this it intended')
      print('\n')

    if self.is_heterogenous and self.alpha is not None:
      self.theta_0 = alpha
    elif self.is_heterogenous and self.alpha is None:
      self.theta_0 = {}
    elif not self.is_heterogenous and self.alpha is not None:
      self.theta_0 = alpha
    else:
      raise ValueError('Setting for instance is selected as not heterogenous. Please select a value for the treatment effect.')


  def _prepare_data(self, setting):
    """
    Prepare the n_rep datasets for a given n and dim_x setting.

    Parameters
    ----------
    setting : str
    """
    self._data = list() #reset
    if not self.is_heterogenous or (self.is_heterogenous and self.alpha is not None):
      causal_param = 'alpha' if 'alpha' in self.DGP.__code__.co_varnames else 'theta'
      DGP_kwargs = {  causal_param : self.alpha, 
                      'n_obs': self._n_obs_act, 
                      'dim_x': self._dim_x_act, 
                      'return_type': 'array'}

      for _ in range(self.n_rep):
        (x, y, d) = self.DGP(**DGP_kwargs)
        self._data.append((x, y, d))
    

    elif self.is_heterogenous and self.alpha is None:
      DGP_kwargs = {  'n_obs': self._n_obs_act, 
                      'dim_x': self._dim_x_act, 
                      'return_type': 'array'}
      self.theta_0[setting] = list()
      for _ in range(self.n_rep):
        (x, y, d, treatment_eff) = self.DGP(**DGP_kwargs)
        self._data.append((x, y, d))
        self.theta_0[setting].append(np.mean(treatment_eff))

  def _create_dml_data_obj(self):
    """
    Create and return a :class: `DoubleMLData` object from arrays.

    Returns
    ----------
    obj_dml_data :class:`DoubleMLData` object
    """

    (x, y, d) = self._data[self._i_rep]
    if 'neural_net' in self._lrn_act:
      x = x.astype('float32')
      y = y.astype('float32')
      d = d.astype('float32')
    obj_dml_data = doubleml.DoubleMLData.from_arrays(x, y, d)
    return obj_dml_data


  def _create_dml_object(self, obj_dml_data):
    """
    Create and return a :class: `DoubleML` object with the given 
    :class: `DoubleMLData` obj_dml_data object.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object

    Returns
    ----------
    obj_dml_model :class:`DoubleML` object
    """
    mll = sklearn.base.clone(self.lrn_dict[self._lrn_act]['ml_l']) if self.model == doubleml.double_ml_plr.DoubleMLPLR else None
    mlm = sklearn.base.clone(self.lrn_dict[self._lrn_act]['ml_m'])
    mlg = sklearn.base.clone(self.lrn_dict[self._lrn_act]['ml_g']) if not (self.model == doubleml.double_ml_plr.DoubleMLPLR and self.score == 'partialling out') else None
    """
    Banane: Frage: Ist "clone" hier noch notwendig, oder kopiert doubleml die learner sowieso?
    """
    model_kwargs = {'obj_dml_data': obj_dml_data,
                    'ml_m': mlm, 
                    'ml_g': mlg,
                    'n_folds': self.n_folds,
                    'score': self.score}

    if self.model == doubleml.double_ml_plr.DoubleMLPLR: model_kwargs['ml_l'] = mll

    obj_dml_model = self.model(**model_kwargs)

    return obj_dml_model


  def _run_fit(self):
    """
    Performs a fit of a :class: `DoubleML` object and saves the results.
    """
    data_obj = self._create_dml_data_obj()
    obj_dml = self._create_dml_object(data_obj)
    obj_dml.fit()
    self._theta_dml_i[self._i_rep] = obj_dml.coef[0]
    self._se_dml_i[self._i_rep] = obj_dml.se[0]
    self._lowCI_i[self._i_rep] = obj_dml.confint()['2.5 %'][0]
    self._upCI_i[self._i_rep] = obj_dml.confint()['97.5 %'][0]


  def run_simulation(self):
    """
    Performs the full simulation for all learners and settings.
    """
    for self._n_obs_act in self.np_dict['n_obs']:
      for self._dim_x_act in self.np_dict['dim_x']:
        indx = str(self._n_obs_act) + '_' + str(self._dim_x_act)
        self._all_permutations.append(indx)
        print(f'Generating dataset via given DGP with n = {str(self._n_obs_act)} and dim_x = {str(self._dim_x_act)}...')
        self._prepare_data(indx)
        print(f'Starting with simulation sessions:')
        print('\n')
        for lvl_lrn_dict in self.lrn_dict.keys():
          self._lrn_act = lvl_lrn_dict
          self._theta_dml_i = np.zeros(shape=(self.n_rep,))
          self._se_dml_i = np.zeros(shape=(self.n_rep,))
          self._lowCI_i = np.zeros(shape=(self.n_rep,))
          self._upCI_i = np.zeros(shape=(self.n_rep,))

          for self._i_rep in tqdm.tqdm(range(self.n_rep), 
                                       bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}',
                                       ascii=False,
                                       desc = f'{lvl_lrn_dict}'.ljust(20, '-')):
            self._run_fit()

          self.theta_dml[self._lrn_act][indx] = self._theta_dml_i
          self.se_dml[self._lrn_act][indx] = self._se_dml_i
          self.lowCI[self._lrn_act][indx] = self._lowCI_i
          self.upCI[self._lrn_act][indx] = self._upCI_i
          self._i_rep = 0
        print('\n')

  def histplot(self):
    """
    Plot histograms of the standardized bias for each learner and setting.
        
    Parameters:
    None
        
    Returns:
    None
    """
    for learner_i in self.lrn_dict.keys():
      for key in self.theta_dml[learner_i].keys():
        n, dim_x = key.split('_')
        theta_0 = self.theta_0[key] if self.is_heterogenous and self.alpha is None else self.theta_0
        self.histograms[learner_i + '_' + key] = plt.figure(constrained_layout=True);
        ax = sns.histplot((self.theta_dml[learner_i][key] - theta_0) / self.se_dml[learner_i][key],
                          color=sns.color_palette('pastel')[2], edgecolor = sns.color_palette('dark')[2],
                          stat='density', bins=30, label='DML estimation');

        ax.axvline(0., color='k');
        ax.set_title(f'{learner_i} for $n_{{obs}}$={n}, $dim_x$={dim_x}')
        xx = np.arange(-5, +5, 0.001)
        yy = stats.norm.pdf(xx)
        ax.plot(xx, yy, color='k', label='$\\mathcal{N}(0, 1)$');
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0));
        ax.set_xlim([-6., 6.]);
        ax.set_xlabel('$(\hat{\\theta}_0 - \\theta_0)/ SE$');
        ax.set_ylabel('Density')
    print('Show histograms? (Y/N)')
    while True:
      decision = input().upper()
      if decision == 'Y':
        break
      elif decision == 'N':
        break
      else:
        print("The input is invalid.\nPlease enter 'Y' or 'N'")
    if decision == 'Y':
      for key in self.histograms.keys():
        self.histograms[key].show()
      plt.show()
      plt.clf()


  def boxplot(self):
    """
    Plot boxplots of the standardized bias for each learner and setting.
        
    Parameters:
    None
        
    Returns:
    None
    """
    t1 = {key: {} for key in self.lrn_dict.keys()}
    t2 = {key: None for key in self.lrn_dict.keys()}
    for learner_i in self.lrn_dict.keys():
      for key in self.theta_dml[learner_i].keys():
        n, dim_x = key.split('_')
        theta_0 = self.theta_0[key] if self.is_heterogenous and self.alpha is None else self.theta_0
        t1[learner_i][f'$n_{{obs}}$={n}, $dim_x$={dim_x}'] = (self.theta_dml[learner_i][key] - theta_0) / self.se_dml[learner_i][key]
      t2[learner_i] = pd.DataFrame.from_dict(t1[learner_i])
      self.boxplots[learner_i] = plt.figure(constrained_layout=True);
      ax = sns.boxplot(data = t2[learner_i], color=sns.color_palette('pastel')[2])
      ax.set_title(learner_i)
      ax.set_xlabel('Data setting')
      ax.set_ylabel('$(\hat{\\theta}_0 - \\theta_0)/ SE$')
    print('Show boxplots? (Y/N)')
    while True:
      decision = input().upper()
      if decision == 'Y':
        break
      elif decision == 'N':
        break
      else:
        print("The input is invalid.\nPlease enter 'Y' or 'N'")
    if decision == 'Y':
      for key in self.boxplots.keys():
        self.boxplots[key].show()
      plt.show()
      plt.clf()

  def _absolute_bias(self, theta_dml_i, theta_0):
    """
    Calculate the average absolute bias.
    
    The absolute bias is defined as the mean of the predictions made by the learner
    minus the true value of the parameter being estimated.
    
    Args:
      theta_dml_i (numpy array): The estimates of the model parameter.
      theta_0 (float or numpy array): The true value of the model parameter.
    
    Returns:
        float: The absolute bias.
    """
    return np.mean(theta_dml_i - theta_0)

  def _relative_bias(self, theta_dml_i, theta_0):
    """
    Calculate the relative bias.
    
    The relative bias is defined as the mean of the difference between the predictions made
    by the learner and the true value of the parameter being estimated, divided by the
    true value of the parameter.
    
    Args:
      theta_dml_i (numpy array): The estimates of the model parameter.
      theta_0 (float or numpy array): The true value of the model parameter.
    
    Returns:
        float: The relative bias.
    """
    return np.mean((theta_dml_i - theta_0) / theta_0)

  def _standardized_bias(self, theta_dml_i, theta_0, se_dml_i):
    """
    Calculate the standardized bias.
    
    The standardized bias is defined as the mean of the difference between the predictions
    made by the learner and the true value of the parameter being estimated, divided by the
    standard error of the predictions made by the learner.

    Args:
      theta_dml_i (numpy array): The estimates of the model parameter.
      theta_0 (float or numpy array): The true value of the model parameter.
      se_dml_i (numpy array): Standard error of the predictions

    Returns:
      float: the standardized bias.
    """
    return np.mean((theta_dml_i) - theta_0) / np.mean(se_dml_i)

  def _rmse(self, theta_dml_i, theta_0):
    """
    Calculate the root mean squared error (RMSE).
    
    The RMSE is defined as the square root of the mean squared error between the predictions
    made by the learner and the true value of the parameter being estimated.

    Args:
      theta_dml_i (numpy array): The estimates of the model parameter.
      theta_0 (float or numpy array): The true value of the model parameter.

    Returns:
      float: the root mean squared error
    """
    return np.sqrt(np.mean((theta_dml_i - theta_0)**2))

  def _average_se(self, se_dml_i):
    """Calculate the average standard error.

    The average standard error is defined as the mean of the standard errors of the predictions
    made by the learner.

    Args:
        se_dml_i (numpy array): the standard errors of the predictions made by the learner.

    Returns:
        float: the average standard error.

    """
    return np.mean(se_dml_i)

  def _empirical_deviation(self, theta_dml_i):
    """
    Calculate the empirical deviation.
    
    The empirical deviation is defined as the square root of the empirical variance between the 
    individual predictions and mean of the predictions made by the learner.

    Args:
        theta_dml_i (numpy array): the predictions made by the learner.
    
    Returns:
        float: the empirical deviation.

    """
    return np.sqrt((1 / (self.n_rep - 1)) * np.sum((theta_dml_i - np.mean(theta_dml_i))**2))

  def _coverage(self, lowCI_i, upCI_i, theta_0):
    """
    Calculate the coverage of the CIs.

    Coverage is defined as the proportion of times that the confidence interval for the
    predictions made by the learner contains the true value of the parameter being estimated.

    Args:
        lowCI_i (numpy array): the lower bound of the confidence interval.
        upCI_i (numpy array): the upper bound of the confidence interval.
        theta_0 (float or numpy array): the true value of the parameter being estimated.

    Returns:
        float: the coverage.
    """
    return np.sum(np.greater_equal(theta_0, lowCI_i) & np.less_equal(theta_0, upCI_i)) / self.n_rep


  def iterate_performance_measures(self):
    """
    Calculate the performance measures for each learner and each setting.

    Calculate the absolute bias, relative bias, standardized bias, root mean squared error, 
    average standard error, empirical deviation, and coverage of the confidence interval for 
    each learner and each setting.

    Returns:
        Dict[str, Dict[str, Dict[Tuple[int, int], float]]]: A dictionary mapping performance measure names to
            dictionaries mapping learner names to dictionaries mapping settings (n_obs and dim_x as tuples of ints) 
            to the performance measure value for that learner and setting.
    """
    for learner_i in self.lrn_dict.keys():
      for setting in self._all_permutations:
          theta_0 = self.theta_0[setting] if self.is_heterogenous and self.alpha is None else self.theta_0
          self.abs_bias[learner_i][setting] = self._absolute_bias(self.theta_dml[learner_i][setting], theta_0)
          self.rel_bias[learner_i][setting] = self._relative_bias(self.theta_dml[learner_i][setting], theta_0)
          self.std_bias[learner_i][setting] = self._standardized_bias(self.theta_dml[learner_i][setting], theta_0, self.se_dml[learner_i][setting])
          self.rmse[learner_i][setting] = self._rmse(self.theta_dml[learner_i][setting], theta_0)
          self.avg_se[learner_i][setting] = self._average_se(self.se_dml[learner_i][setting])
          self.empdev[learner_i][setting] = self._empirical_deviation(self.theta_dml[learner_i][setting]) if self.n_rep > 1 else None
          self.coverage[learner_i][setting] = self._coverage(self.lowCI[learner_i][setting], self.upCI[learner_i][setting], theta_0)
        
    performance_dict = {'Absolute Bias': self.abs_bias, 
                        'Relative Bias': self.rel_bias,
                        'Standardized Bias': self.std_bias,
                        'Root Mean Squared Error (RMSE)': self.rmse,
                        'Average Standard Error (SE)':  self.avg_se,
                        'Empirical Deviation (ED)': self.empdev,
                        'Coverage of CI': self.coverage}
    self.performance_dict = performance_dict


  def measure_performance(self):
    """
    Measure the performance of each learner for each setting.

    This method calculates a number of performance measures for each learner and each setting,
    including absolute bias, relative bias, standardized bias, root mean squared error (RMSE),
    average standard error (SE), empirical deviation (ED), and coverage of the confidence intervals.

    The results are stored in a Pandas DataFrame, which is also printed to the console.

    Returns:
        self: Returns an instance of the current object, with the results stored in the 
            `performance_df` attribute. 
    """
    
    self.iterate_performance_measures()
    self.performance_df = pd.DataFrame.from_dict({(i,j): self.performance_dict[i][j] 
                           for i in self.performance_dict.keys() 
                           for j in self.performance_dict[i].keys()},
                           orient='columns').T

    print(self.performance_df)

  def save(self, pth: str):
    """
    Save the simulation study's results to the specified path.

    The results that are saved including the model properties, DGP, number of replications, 
    the list of number of observations and dimension of X, the list of learning algorithms, 
    the true value of the parameter, the seed used, the estimated parameters of the model 
    and standard errors, and the performance measures as well as Histograms and Boxplots of the results. 
    The data is saved in json, excel and png format.
        
    Parameters:
      pth (str): the path to save the simulation study results.
        
    Returns:
      None
    """
    save_dict = {'model': self.__dict__['model'],
                 'DGP': self.__dict__['DGP'],
                 'n_rep': self.__dict__['n_rep'],
                 'np_dict': self.__dict__['np_dict'],
                 'lrn_dict': self.__dict__['lrn_dict'],
                 'alpha': self.__dict__['alpha'],
                 'seed' : self.__dict__['_seed'],
                 'theta_dml': self.__dict__['theta_dml'],
                 'se_dml': self.__dict__['se_dml'],
                 'performance_measures': self.__dict__['performance_df']}
    save_flatten = pd.json_normalize(save_dict, sep='_')
    if not pth.endswith('/'): pth += '/'
    try:
      os.makedirs(pth)
    except FileExistsError:
      pass
    save_flatten.to_json(pth + 'log.json')
    pd.DataFrame(self.theta_dml).to_excel(pth + 'theta.xlsx')
    pd.DataFrame(self.se_dml).to_excel(pth + 'se.xlsx')
    self.performance_df.to_excel(pth + 'performance.xlsx')
    try:
      os.makedirs(pth + 'Histograms/')
      os.makedirs(pth + 'Boxplots/')
    except FileExistsError:
      pass
    for label, fig in self.histograms.items():
      fig.savefig(pth + f'Histograms/Histogram_{label}.png')
    for label, fig in self.boxplots.items():
      fig.savefig(pth + f'Boxplots/Boxplot_{label}.png')