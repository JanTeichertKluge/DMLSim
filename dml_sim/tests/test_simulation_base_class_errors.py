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

sys.path.append(r"C:\Users\Nutzer\Source\Repos\DMLSim\dml_sim")
from simulation_base_class import simulation_study

n_obs_dim_x = {"n_obs": [500], "dim_x": [10, 20]}
n_rep_test = 5


@pytest.mark.parametrize(
    "model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha",
    [
        # Case that IV-Type and no ml_g
        (
            doubleml.DoubleMLPLR,
            False,
            "IV-type",
            make_plr_CCDDHNR2018,
            n_rep_test,
            n_obs_dim_x,
            {
                "lasso": {"ml_l": clone(LassoCV()), "ml_m": clone(LassoCV())},
                "random_forests": {
                    "ml_l": clone(RandomForestRegressor()),
                    "ml_m": clone(RandomForestRegressor()),
                },
            },
            1,
        ),
        # Case if wrong score
        (
            doubleml.DoubleMLPLR,
            False,
            "ATE",
            make_plr_CCDDHNR2018,
            n_rep_test,
            n_obs_dim_x,
            {
                "lasso": {
                    "ml_l": clone(LassoCV()),
                    "ml_m": clone(LassoCV()),
                    "ml_g": clone(LassoCV()),
                },
                "random_forests": {
                    "ml_l": clone(RandomForestRegressor()),
                    "ml_m": clone(RandomForestRegressor()),
                    "ml_g": clone(RandomForestRegressor()),
                },
            },
            1,
        ),
        (
            doubleml.DoubleMLIRM,
            True,
            "partialling out",
            make_irm_data,
            n_rep_test,
            n_obs_dim_x,
            {
                "lasso": {
                    "ml_m": clone(
                        LogisticRegressionCV(penalty="l1", solver="liblinear")
                    ),
                    "ml_g": clone(LassoCV()),
                },
                "random_forests": {
                    "ml_m": clone(RandomForestClassifier()),
                    "ml_g": clone(RandomForestRegressor()),
                },
            },
            0.5,
        ),
        # Case wrong learner for irm model
        (
            doubleml.DoubleMLIRM,
            True,
            "ATTE",
            make_irm_data,
            n_rep_test,
            n_obs_dim_x,
            {
                "lasso": {
                    "ml_m": clone(
                        LogisticRegressionCV(penalty="l1", solver="liblinear")
                    ),
                    "ml_l": clone(LassoCV()),
                },
                "random_forests": {
                    "ml_m": clone(RandomForestClassifier()),
                    "ml_l": clone(RandomForestRegressor()),
                },
            },
            0.5,
        ),
    ],
)
def test_wrong_learners_init(
    model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha
):
    with pytest.raises(ValueError):
        sim = simulation_study(
            model, is_heterogenous, score, DGP, n_rep, np_dict, learner_dict, alpha
        )
        sim.run_simulation()
