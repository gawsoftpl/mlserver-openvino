from sklearn.base import TransformerMixin

class PrepareTextForNN(TransformerMixin):
    '''
    Convert text and prepare to Neural Network input
    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
