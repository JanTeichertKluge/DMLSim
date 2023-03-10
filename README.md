# DMLSim - Simulation studies for Double Machine Learning in Python

The Python package **DMLSim** provides a simulation study framework using [DoubleML](https://github.com/DoubleML/doubleml-for-py)s Implementation of the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).

## Main Features

Simulation studies with Double / debiased machine learning for 

- Partially linear regression models (PLR)      
- Interactive regression models (IRM)

Instances of the main class 'simulation_study' can be used with all learners from sklearn. 
The learners need a fit() and a predict() method.
The module 'dml_nn' can be used to create a dictionary with the corresponding models. 
You are able to pass layer- and hyper-parameters when initializing the class 'network_builder'.

The DGP (data generating process) should take at least 'n_obs', 'dim_x' as arguments. 
'alpha' / 'theta' is necessary for all DGPs with non heterogenous treatment effect. 
The callable should return numpy arrays:
- X, dim(X) = (n_obs, dim_x)
- y, dim(y) = (n_obs,)
- d, dim(d) = (n_obs,)
- and theta, dim(theta) = (n_obs,) in order to calculate the average treatment effect if the treatment effect is heterogenous.

In some cases, the DGP (i.e. doubleml.datasets.make_irm_data) generates a heterogenous treatment effect from an argument with fixed value for theta. In these cases, initialize your instance with specific value for alpha.

The currently supported DGPs:
- all DGPs from doubleml.datasets
- all DGPs from dml_sim.datasets
- DGPs from the [opossum package](https://github.com/jgitr/opossum)


## Installation

**DMLSim** requires

- Python
- sklearn
- numpy
- scipy
- pandas
- statsmodels
- joblib
- DoubleML
- matplotlib
- seaborn
- tqdm
- etc.

To install DMLSim use

```
pip install git+https://github.com/JanTeichertKluge/DMLSim.git
```

## Usage
- [Find an example for PLR in Colab](https://colab.research.google.com/drive/1UT1cSmyz-c8ASKeLavoDv8yoe2wxwnaf?usp=sharing)
- [Find an example for IRM in Colab](https://colab.research.google.com/drive/1LHdHTFZSDweR6jgA7EXoZ1l-vdzAcrj5?usp=sharing)


Example Code for DMLSim (IRM):
```python
# Make imports
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from doubleml import DoubleMLIRM

from dml_sim.simulation_base_class import simulation_study as ssession # import simulation session object
from dml_sim.datasets import make_irm_farell2021 # import a data generating process
from dml_nn.simulation_learner import network_builder # import network builder to build torch networks wrapped in sklearn syntax using skorch

builder = network_builder() #Init builder with default settings
builder.__dict__ #Show default settings

# Define ML learner
lasso_reg = LassoCV()
lasso_cls = LogisticRegressionCV(penalty='l1', solver='liblinear')
rf_reg = RandomForestRegressor()
rf_cls = RandomForestClassifier()

# Create a structured learner dict for machine learning algorithms
learner_dict_irm_ml = {
        'lasso': {
            'ml_m' : clone(lasso_cls),
            'ml_g' : clone(lasso_reg)},
        'random_forests': {
            'ml_m' : clone(rf_cls),
            'ml_g' : clone(rf_reg)}
        }
  
 # Create a structured learner dict for neural network algorithms
learner_dict_irm_nn = builder.get_irm_nn_learners()      

# Combine dicts
learner_dict_irm = {**learner_dict_irm_ml, **learner_dict_irm_nn}        

# Create a dict with n_obs and dim_x for the DGP data setting
np_dict = {'n_obs': [500], 'dim_x': [10, 20]}

# Init a dml simulation session
scenario_A = ssession(model = DoubleMLIRM, 
                    score = 'ATE',
                    DGP = make_irm_farell2021, 
                    n_rep = 100,
                    np_dict =  np_dict, 
                    lrn_dict = learner_dict_irm, 
                    alpha = None,
                    is_heterogenous=True)
                    
# Perform full simulation
scenario_A.run_simulation()

# Generate boxplots
scenario_A.boxplot()

# Generate histograms
scenario_A.histplot()

# Measure performances
scenario_A.measure_performance()

# Save measures and plots to NEW_FOLDER
scenario_A.save('/content/NEW_FOLDER')
```
