import numpy as np
from skorch import NeuralNetRegressor


class NeuralNetRegressorDoubleOut(NeuralNetRegressor):
    """A neural network regressor class that supports multiple outputs.

    This class extends `NeuralNetRegressor` to handle models with multiple
    outputs. It creates the `predict_features` and overwrites the `get_loss` methods
    to handle the multiple outputs.

    Methods:
    -------
    predict_features(X):
        Predicts the features from the model.

    get_loss(y_pred, y_true, *args, **kwargs):
        Returns the loss function for the model.
    """

    def predict_features(self, X):
        y_features = []
        for yf in super().forward_iter(X, training=False):
            yf = yf[1] if isinstance(yf, tuple) else yf
            y_features.append(yf)
        y_feature = np.concatenate(y_features, 0)
        return y_feature

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(y_pred, tuple):
            logits, _ = y_pred
        else:
            logits = y_pred
        loss_reconstruction = super().get_loss(logits, y_true, *args, **kwargs)
        return loss_reconstruction
