# DMLSim - Library with packages on DoubleML and Neural Networks in Python

This library includes three packages:
- dml_sim
- dml_nn
- dml_emb

The Python package **dml_sim** from the library **DMLSim** provides a simulation study framework using [DoubleML](https://github.com/DoubleML/doubleml-for-py)s Implementation of the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).

The package **dml_nn** includes an API to use PyTorch Neural Networks together with the [skorch](https://github.com/skorch-dev/skorch) library inside the [DoubleML](https://github.com/DoubleML/doubleml-for-py). There are also several tools for creating Feed Forwards Neural Networks for use in PLR or IRM models.

The modules from the **dml_emb** package including several methods and tools to use embeddings from [transformer](https://github.com/huggingface/transformers) models as covariates in the [DoubleML](https://github.com/DoubleML/doubleml-for-py) framework. This package contains torch based architectures like a multimodal ensemble of transformer models, solutions to create embeddings like the cross-fitting method and modified APIs to use these models in a high-level way with [skorch](https://github.com/skorch-dev/skorch).

## About simulation studies with dml_sim

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

### Example Simulation
Example Code for DMLSim.dml_sim (IRM):

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

## About the use of embeddings with dml_emb
To include image and text data as confounders within the DoubleML framework, transformer models can be used to create embeddings from this unstructured data. Assume that there is a pandas DataFrame with columns for the output variable, the treatment variable and a column for text and image data. The *TransformerEnsemble* module contains model classes that can process this multimodal input. To deal with this in the high-level syntax of skorch, the modified classes from *FeatureRegressor* are used. The embeddings can then be generated from the DataFrame. In order to avoid data loss, the class *CrossEmbeddings* can be used, which creates the embeddings according to the cross-fitting approach.

### Example Embedding Generation
```python
from dml_emb.CrossEmbeddings import CrossEmbeddings
from dml_emb.FeatureRegressor import NeuralNetRegressorDoubleOut
from dml_emb.TransformerEnsemble import FineTuned_TransformerEnsemble

IMG = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
TXT = "bert-base-uncased"

module_p = FineTuned_TransformerEnsemble(image_model=IMG, 
                                         text_model=TXT,
                                         num_labels=1)
module_q = FineTuned_TransformerEnsemble(image_model=IMG, 
                                         text_model=TXT,
                                         num_labels=1)

def r2(net, X, y):
    return r2_score(y, net.predict(X))

model_p = NeuralNetRegressorDoubleOut( 
    module_p,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    #optimizer__amsgrad=True,
    lr=3e-5,
    max_epochs=1,
    batch_size=16, #try 16
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[ProgressBar(),
               EpochScoring(r2, use_caching=False, lower_is_better=False),
               EarlyStopping(patience=3, threshold=0.01,
                             threshold_mode='rel', lower_is_better=True,
                             load_best=True)]
)

model_q = NeuralNetRegressorDoubleOut(
    module_q,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.AdamW,
    #optimizer__amsgrad=True,
    lr=3e-5,
    max_epochs=1,
    batch_size=16, #try 16
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    callbacks=[ProgressBar(),
               EpochScoring(r2, use_caching=False, lower_is_better=False),
               EarlyStopping(patience=3, threshold=0.01,
                             threshold_mode='rel', lower_is_better=True,
                             load_best=True)]
)

ce = CrossEmbeddings(dataset = smpl, 
                     text_col = 'text', 
                     image_col = 'img_get', 
                     d_col = 'ln_p', 
                     y_col = 'ln_q', 
                     n_folds = 3,
                     aux_d = model_p, 
                     aux_y = model_q, 
                     txt_str = TXT, 
                     img_str = IMG)

ce.fit_and_predict_embeddings()

emb_df = ce.get_embedded_df()
emb_ar = ce.get_embeddings()
```



## Installation

**DMLSim** requires Python 3 with the following packages:
- DoubleML
- joblib
- matplotlib
- numpy
- openpyxl
- pandas
- Pillow
- python- dateutil
- scikit- learn
- scipy==1.7.3
- seaborn
- skorch
- statsmodels
- torch
- tqdm
- transformers

To install DMLSim use

```
pip install git+https://github.com/JanTeichertKluge/DMLSim.git
```
