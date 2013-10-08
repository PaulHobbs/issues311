__author__ = 'phobbs'

from operator import itemgetter

from sklearn import feature_extraction

class Transformer(object):
    def fit_transform(self, X, y=None):
        # Default.  Override if there is a more efficient way of doing this.
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class DictTransformer(Transformer):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)

    def transform(self, X):
        self.transformer.transform(X)


class FunctionDictTransformer(Transformer):
    def __init__(self, f):
        self.f = f

    def fit(self, X, y=None):
        return

    def transform(self, X):
        return self.f(X)


def set_params(thing, **params):
    if hasattr(thing, 'set_params'):
        return thing.set_params(**params)
    for k, v in params.iteritems():
        setattr(thing, k, v)
    return thing


def get_params(thing):
    return getattr(thing, 'get_params', lambda: {})()


class DictPipeline(Transformer):
    def __init__(self, steps):
        self.steps = steps

    def set_params(self, **kwargs):
        steps = dict(self.steps)
        for k, v in kwargs.iteritems():
            key, param = k.lsplit('__')
            steps[key].set_params({param: v})
        return self

    def get_params(self):
        return dict(
            (k, v) for name, thing in self.steps
            for k, v in get_params(thing)
        )

    def fit(self, X, y=None):
        for _, thing in self.steps[:-1]:
            X = thing.fit_transform(X, y)
        return self.steps[-1].fit(X, y)

    def transform(self, X, y=None):
        for _, thing in self.steps:
            X = thing.transform(X, y)
        return X

    def predict(self, X):
        for _, thing in self.steps[:-1]:
            X = thing.transform(X)
        return self.steps[-1].predict(X)


