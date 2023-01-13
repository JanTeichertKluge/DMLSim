from skorch import NeuralNetRegressor

'''
    It is necessary to use this modified class. 
    Otherwise doubleML raises an error 
    in the .fit() method.
'''

class NeuralNetRegressorXdoubleML(NeuralNetRegressor):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


  def fit(self, X, y):
    y = y.reshape(-1,1)
    super().fit(X, y)

  def predict(self, X):
    pred = super().predict(X) #banane: testen!
    return pred.reshape(-1)





class NeuralNetRegressorDoubleML:
    _estimator_type = 'regressor'

    def __init__(self, *args, **kwargs):
        self.torch_nn = NeuralNetRegressor(*args, **kwargs)


    def set_params(self, **params):
        self.torch_nn.set_params(**params)
        return self

    def get_params(self, deep=True):
        dict = self.torch_nn.get_params(deep)
        return dict

    def fit(self, X, y):
        y = y.reshape(-1,1)
        self.torch_nn.fit(X, y)
        return self

    def predict(self, X):
        preds = self.torch_nn.predict(X)
        return preds.reshape(-1)
