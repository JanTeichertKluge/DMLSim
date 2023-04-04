import numpy as np
import torch
from skorch import NeuralNetRegressor


class NeuralNetRegressorDoubleOut(NeuralNetRegressor):
    """A neural network regressor class that supports multiple outputs.

    This class extends `NeuralNetRegressor` to handle models with multiple
    outputs. It creates the `predict_features` and overwrites the `get_loss` methods
    to handle the multiple outputs.

    Parameters
    ----------
    NeuralNetRegressor : class
        The parent class that this class extends.

    Methods
    -------
    predict_features(X):
        Predicts the features from the model.

    get_loss(y_pred, y_true, *args, **kwargs):
        Returns the loss function for the model.
    """

    def predict_features(self, X):
        """Predicts the features from the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data for which to predict features.

        Returns
        -------
        y_proba : array-like, shape (n_samples, n_features)
            The predicted features.
        """
        y_probas = []
        for yp in super().forward_iter(X, training=False):
            yp = yp[1] if isinstance(yp, tuple) else yp
            y_probas.append(yp)
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Returns the loss function for the model.

        Overrides the `get_loss` method of the parent class to handle
        models with multiple outputs.

        Parameters
        ----------
        y_pred : array-like
            The predicted values from the model.

        y_true : array-like
            The true values for the input data.

        *args : list
            Additional positional arguments to pass to the parent method.

        **kwargs : dict
            Additional keyword arguments to pass to the parent method.

        Returns
        -------
        loss_reconstruction : Tensor
            The loss function for the model.
        """
        if isinstance(y_pred, tuple):
            logits, _ = y_pred
        else:
            logits = y_pred
        loss_reconstruction = super().get_loss(logits, y_true, *args, **kwargs)
        return loss_reconstruction
