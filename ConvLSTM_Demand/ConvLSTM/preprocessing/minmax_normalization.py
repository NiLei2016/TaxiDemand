from __future__ import print_function
import numpy as np
np.random.seed(1337)
class MinMaxNormalization(object):
    """
    MinMax Normalization-->[-1,1]
      x=(x-min)/(max-min)
      x=x*2-1
    """
    def __int__(self):
        pass
    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print('min', self._min, 'max', self._max)
    def transform(self,X):
        X = 1.*(X-self._min)/(self._max-self._min)
        X = X*2.-1
        return X
    def fit_transform( self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        X = (X+1.)/2.
        X = 1.*X*(self._max-self._min)+self._min
        return X