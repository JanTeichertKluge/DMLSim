from skorch import NeuralNetRegressor


class NeuralNetRegressorDoubleML:
    """
    Custom regressor compatible with the DoubleML package.
    This class contains an underlying neural network regressor implemented with PyTorch and skorch.

    Parameters
    ----------
    *args : list
        Positional arguments to pass to the NeuralNetRegressor constructor.
    **kwargs : dict
        Keyword arguments to pass to the NeuralNetRegressor constructor.

    Attributes
    ----------
    _estimator_type : str
        The type of estimator, set to "regressor".
    torch_nn : NeuralNetRegressor
        The underlying neural network regressor implemented with PyTorch and skorch.

    Methods
    -------
    set_params(**params)
        Set the parameters of the underlying NeuralNetRegressor instance using the set_params method of that instance.

    get_params(deep=True)
        Get the parameters of the underlying NeuralNetRegressor instance using the get_params method of that instance.

    fit(X, y)
        Fit the neural network regressor to the input features X and the target variable y.
        Reshapes the target variable y into a two-dimensional array with one column.
        Calls the fit method of the torch_nn attribute using the input features X and the reshaped target variable y.

    predict(X)
        Generate predictions for the input features X using the fitted neural network model.
        Calls the predict_proba method of the torch_nn attribute to generate predictions.
        Reshapes the predictions to a one-dimensional array using the reshape method of NumPy.

    """

    _estimator_type = "regressor"

    def __init__(self, *args, **kwargs):
        self.torch_nn = NeuralNetRegressor(*args, **kwargs)

    def set_params(self, **params):
        self.torch_nn.set_params(**params)
        return self

    def get_params(self, deep=True):
        dict = self.torch_nn.get_params(deep)
        return dict

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.torch_nn.fit(X, y)
        return self

    def predict(self, X):
        preds = self.torch_nn.predict_proba(X)
        return preds.reshape(-1)


class NeuralNetRegressorXdoubleML(NeuralNetRegressor):
    """
    Subclass of NeuralNetRegressor that is compatible with the DoubleML package.
    This class reshapes the target variable to a two-dimensional array and the predicted
    values to a one-dimensional array to be compatible with DoubleML's interface.

    Parameters
    ----------
    *args : list
        Positional arguments to pass to NeuralNetRegressor constructor.
    **kwargs : dict
        Keyword arguments to pass to NeuralNetRegressor constructor.

    Attributes
    ----------
    Inherits all attributes from NeuralNetRegressor.

    Methods
    -------
    fit(X, y)
        Fits the neural network model to the input features X and the target variable y.
        Reshapes the target variable y into a two-dimensional array with one column.
        Calls the fit method of the parent class using the super() function.

    predict(X)
        Generates predictions for the input features X using the fitted neural network model.
        Calls the predict method of the parent class using the super() function.
        Reshapes the predicted values into a one-dimensional array.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        super().fit(X, y)

    def predict(self, X):
        pred = super().predict(X)
        return pred.reshape(-1)
