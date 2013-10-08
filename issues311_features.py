__author__ = 'phobbs'

from features import *


FLOAT_FEATURES = ('latitude', 'longitude')
STRING_FEATURES = ('summary', 'description', 'source')

OTHER_FEATURES = []


def feature(f):
    OTHER_FEATURES.append(f)
    return f


def vectorizer_features(name, vec=feature_extraction.CountVectorizer):
    return DictPipeline([
        ('accessor', FunctionDictTransformer(itemgetter(name))),
        ('vec', DictTransformer(vec()))
    ])
