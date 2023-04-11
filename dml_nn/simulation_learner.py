import sys
import torch
from torch import nn
from torch import optim
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.base import clone


try:
    from .doubleml_skorch_api import NeuralNetRegressorDoubleML
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Could not import NeuralNetRegressorDoubleML API.")


class RegressorModel(nn.Module):
    """
    Neural network model for regression tasks.

    This class defines a PyTorch neural network model for regression tasks. The model
    consists of a number of fully-connected layers with ReLU activation and dropout
    regularization. The number and size of the layers are specified by the `sizes`
    parameter.

    Args:
    ----------
        sizes (list): A list of integers specifying the number of neurons in each layer.

    Attributes:
    ----------
        model (nn.Module): The PyTorch neural network model.

    Methods:
    ----------
        forward(x)
            Forward pass of the model.
            This method performs the forward pass of the model, i.e., it computes the
            output of the model given an input tensor.
    """

    def __init__(self, sizes, dropout):
        super(RegressorModel, self).__init__()
        layers = []
        layers.append(nn.LazyLinear(sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
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
    ----------
        sizes (list): A list of integers specifying the number of neurons in each layer.

    Attributes:
    ----------
        model (nn.Module): The PyTorch neural network model.

    Methods:
    ----------
        forward(x)
            Forward pass of the model.
            This method performs the forward pass of the model, i.e., it computes the
            output of the model given an input tensor.
    """

    def __init__(self, sizes, dropout):
        super(ClassifierModel, self).__init__()
        layers = []
        layers.append(nn.LazyLinear(sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-1], 2))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class network_builder:
    """
    This class provides methods to build neural network models for partial linear regression and
    interactive regression models using different architectures specified by the layer_sizes parameter.

    Args:
    ----------
        layer_sizes (dict, optional): A dictionary of model names and their corresponding hidden layer sizes.
        dropout (float, optional): The probability of dropout. Defaults to 0.2.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        batch_size (int, optional): The batch size used for training the models. Defaults to 128.
        n_epochs (int, optional): The maximum number of epochs for training the models. Defaults to 100.
        early_stopping (int, optional): The number of epochs to wait for the validation loss to improve
            before early stopping. Defaults to 10.
        seed (int, optional): The seed used to initialize the models. Defaults to 1234.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.
        optimizer (torch.optim, optional): The optimizer used for training the models. Defaults to optim.Adam.

    Attributes:
    ----------
        layer_sizes (dict): A dictionary of model names and their corresponding hidden layer sizes.
        dropout (float): The probability of dropout.
        lr (float): The learning rate for the optimizer.
        batch_size (int): The batch size used for training the models.
        n_epochs (int): The maximum number of epochs for training the models.
        early_stopping (int): The number of epochs to wait for the validation loss to improve before early stopping.
        seed (int): The seed used to initialize the models.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        optimizer (torch.optim): The optimizer used for training the models.

    Methods:
    ----------
        _prepare_neural_networks():
            Prepare a dictionary of Neural Network models using skorch.

            Returns:
                dict: A dictionary of skorch Neural Net models.

        get_plr_nn_learners(discrete_treatment=False):
            This function creates a dictionary of Learner models for a partial linear model
            with continuous output. The torch models in the dictionary are trained using
            the specified learning rate and number of epochs, and are equipped with early stopping using
            the specified patience.

            Args:
                discrete_treatment (bool, optional): A flag indicating whether the treatment is discrete
                    or continuous. Defaults to False.

            Returns:
                dict: A dictionary of skorch models.

        get_irm_nn_learners():
            This function creates a dictionary of Learner models for an interactive regression model.
            The torch models in the dictionary are trained using the specified learning rate and
            number of epochs, and are equipped with early stopping using the specified patience.

            Returns:
                dict: A dictionary of skorch models.
    """

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

    def _prepare_neural_networks(self):
        nn_reg_skorch_models = {}
        nn_cls_skorch_models = {}
        for name, size in self.layer_sizes.items():
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

    def get_plr_nn_learners(self, discrete_treatment=False):
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

    def get_irm_nn_learners(self):
        nn_reg_skorch_models, nn_cls_skorch_models = self._prepare_neural_networks()
        learner_dict_irm_nn = {
            f"neural_net_{nn_name}": {
                "ml_m": clone(nn_cls_skorch_models[nn_name]),
                "ml_g": clone(nn_reg_skorch_models[nn_name]),
            }
            for nn_name in nn_reg_skorch_models.keys()
        }
        return learner_dict_irm_nn
