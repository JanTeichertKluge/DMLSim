from itertools import dropwhile
from sklearn.base import clone

import torch
from torch import nn
from torch import optim

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping

import sys

try:
    from .doubleml_skorch_api import NeuralNetRegressorDoubleML

    # sys.path.append(r'C:\Users\Nutzer\source\repos\doubleml-neuralnets\API\torch_api')
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Could not import NeuralNetRegressorDoubleML API.")

# Define Classes for Neural Network Models

# Regressor
class RegressorModel(nn.Module):
    """
    Neural network model for regression tasks.

    This class defines a PyTorch neural network model for regression tasks. The model
    consists of a number of fully-connected layers with ReLU activation and dropout
    regularization. The number and size of the layers are specified by the `sizes`
    parameter.

    Args:
        sizes (list): A list of integers specifying the number of neurons in each layer.

    Attributes:
        model (nn.Module): The PyTorch neural network model.
    """

    def __init__(self, sizes, dropout):
        super(RegressorModel, self).__init__()
        # Create a list of layers
        layers = []
        layers.append(nn.LazyLinear(sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            # Add a dropout layer
            layers.append(nn.Dropout(p=dropout))
        # Add the output layer
        layers.append(nn.Linear(sizes[-1], 1))
        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.
        This method performs the forward pass of the model, i.e., it computes the
        output of the model given an input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)


# Classifier
class ClassifierModel(nn.Module):
    """
    Neural network model for classification tasks.

    This class defines a PyTorch neural network model for classification tasks. The model
    consists of a number of fully-connected layers with ReLU activation and dropout
    regularization. The number and size of the layers are specified by the `sizes`
    parameter.

    Args:
        sizes (list): A list of integers specifying the number of neurons in each layer.

    Attributes:
        model (nn.Module): The PyTorch neural network model.
    """

    def __init__(self, sizes, dropout):
        super(ClassifierModel, self).__init__()
        # Create a list of layers
        layers = []
        layers.append(nn.LazyLinear(sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            # Add a dropout layer
            layers.append(nn.Dropout(p=dropout))
        # Add the output layer
        layers.append(nn.Linear(sizes[-1], 2))
        layers.append(nn.Softmax(dim=1))
        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.
        This method performs the forward pass of the model, i.e., it computes the
        output of the model given an input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)


class network_builder:
    def __init__(
        self,
        # Literature based parameter settings for training neural networks
        layer_sizes={
            "model0": [1],
            "model1": [5],
            "model2": [10],
            "model3": [10, 5],
            "model4": [10, 5, 3],
            "model5": [20, 15, 5],
            "model6": [80, 80, 80],
            "model7": [60, 30, 20, 10],
            "model8": [20, 15, 15, 10, 10, 5],
            "model9": [60, 30, 20, 20, 10, 5],
        },
        dropout=0.2,
        lr=0.001,
        batch_size=128,
        n_epochs=100,
        early_stopping=10,
        seed=1234,
        weight_decay=0,
        optimizer=optim.Adam,
    ):
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.seed = seed
        self.weight_decay = weight_decay
        self.optimizer = optimizer

    # Method to build Neural Network models
    def _prepare_neural_networks(self):
        """
        Prepare a dictionary of Neural Network models using skorch.

        The function creates a dictionary of skorch Neural Net models using different architectures
        specified by the layer_sizes parameter. The models are trained using the specified learning rate
        and number of epochs, and are equipped with early stopping using the specified patience.

        Returns:
            dict: A dictionary of skorch Neural Net models.
        """

        # Create a dictionary to store the models
        nn_reg_skorch_models = {}
        nn_cls_skorch_models = {}
        # Iterate over the layer sizes
        for name, size in self.layer_sizes.items():
            # Create the model
            torch.manual_seed(self.seed)
            model_reg = RegressorModel(sizes=size, dropout=self.dropout)
            model_cls = ClassifierModel(sizes=size, dropout=self.dropout)
            skorch_regressor = NeuralNetRegressorDoubleML(
                module=model_reg,
                criterion=nn.MSELoss,
                optimizer=self.optimizer,
                optimizer__weight_decay=self.weight_decay,
                lr=self.lr,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=self.batch_size,
                verbose=0,
                max_epochs=self.n_epochs,
                callbacks=[EarlyStopping(patience=self.early_stopping)],
            )

            skorch_classifier = NeuralNetClassifier(
                module=model_cls,
                criterion=nn.CrossEntropyLoss,
                optimizer=self.optimizer,
                optimizer__weight_decay=self.weight_decay,
                lr=self.lr,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=self.batch_size,
                verbose=0,
                max_epochs=self.n_epochs,
                callbacks=[EarlyStopping(patience=self.early_stopping)],
            )

            nn_cls_skorch_models[name] = skorch_classifier
            nn_reg_skorch_models[name] = skorch_regressor

        return nn_reg_skorch_models, nn_cls_skorch_models

    # Creates a learner dict ready for PLR
    def get_plr_nn_learners(self, discrete_treatment=False):
        """
        This function creates a dictionary of Learner models for a partial linear model
        with continuous output. The torch models in the dictionary are trained using
        the specified learning rate and number of epochs, and are equipped with early stopping using
        the specified patience. The models in the dictionary use either neural network or traditional
        machine learning models as probabilistic learners.

        Args:
            discrete_treatment (bool, optional): A flag indicating whether the treatment is discrete
            or continuous. Defaults to False.

        Returns:
            dict: A dictionary of doubleML Double Learner models.
        """
        nn_reg_skorch_models, nn_cls_skorch_models = self._prepare_neural_networks()

        learner_dict_plr_nn = {
            f"neural_net_{nn_name}": {
                "ml_l": clone(nn_reg_skorch_models[nn_name]),
                "ml_m": clone(
                    nn_cls_skorch_models[nn_name]
                    if discrete_treatment
                    else nn_reg_skorch_models[nn_name]
                ),
                "ml_g": clone(nn_reg_skorch_models[nn_name]),
            }
            for nn_name in nn_reg_skorch_models.keys()
        }

        return learner_dict_plr_nn

    # Creates a learner dict ready for IRM
    def get_irm_nn_learners(self):
        """
        This function creates a dictionary of Learner models for an interactive regression model.
        The torch models in the dictionary are trained using the specified learning rate and
        number of epochs, and are equipped with early stopping using the specified patience.
        The models in the dictionary use either neural network or traditional
        machine learning models as probabilistic learners.

        Returns:
            dict: A dictionary of doubleML Double Learner models.
        """

        nn_reg_skorch_models, nn_cls_skorch_models = self._prepare_neural_networks()

        learner_dict_irm_nn = {
            f"neural_net_{nn_name}": {
                "ml_m": clone(nn_cls_skorch_models[nn_name]),
                "ml_g": clone(nn_reg_skorch_models[nn_name]),
            }
            for nn_name in nn_reg_skorch_models.keys()
        }

        return learner_dict_irm_nn
