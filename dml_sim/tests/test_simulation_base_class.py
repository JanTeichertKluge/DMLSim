import pytest
import doubleml
import numpy as np
import pandas as pd
import doubleml
from doubleml.datasets import make_plr_CCDDHNR2018
from doubleml.datasets import make_irm_data
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
import sys
sys.path.append(r'C:\Users\Nutzer\Source\Repos\DMLSim\dml_sim')
from simulation_base_class import simulation_study

learner_dict_irm_ml = {
    'lasso': {
        'ml_m' : clone(LogisticRegressionCV(penalty='l1', solver='liblinear')),
        'ml_g' : clone(LassoCV())},
    'random_forests': {
        'ml_m' : clone(RandomForestClassifier()),
        'ml_g' : clone(RandomForestRegressor())}}

learner_dict_plr_ml = {
    'lasso': {
        'ml_l' : clone(LassoCV()),
        'ml_m' : clone(LassoCV()),
        'ml_g' : clone(LassoCV())},
    'random_forests': {
        'ml_l' : clone(RandomForestRegressor()),
        'ml_m' : clone(RandomForestRegressor()),
        'ml_g' : clone(RandomForestRegressor())}}

n_obs_dim_x = {'n_obs': [500], 'dim_x': [10, 20]}
n_rep_test = 5

@pytest.mark.parametrize("model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha", 
    [
    (doubleml.DoubleMLPLR, False, 'IV-type', make_plr_CCDDHNR2018, n_rep_test, n_obs_dim_x, learner_dict_plr_ml, 1),
    (doubleml.DoubleMLPLR, False, 'partialling out', make_plr_CCDDHNR2018, n_rep_test, n_obs_dim_x, learner_dict_plr_ml, 1),
    (doubleml.DoubleMLIRM, True, 'ATE', make_irm_data, n_rep_test, n_obs_dim_x, learner_dict_irm_ml, 0.5),
    (doubleml.DoubleMLIRM, True, 'ATTE', make_irm_data, n_rep_test, n_obs_dim_x, learner_dict_irm_ml, 0.5)
])
def test_simulation_study_expected(model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha):
    sim_study = simulation_study(model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha)
    sim_study.run_simulation()
    assert sim_study.model_attr is not None
    sim_study.boxplot()
    assert sim_study.boxplots is not None
    sim_study.histplot()
    assert sim_study.histograms is not None
    sim_study.measure_performance()
    assert sim_study.performance_df is not None