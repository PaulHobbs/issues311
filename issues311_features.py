import calendar

__author__ = 'phobbs'

from operator import itemgetter
import dateutil.parser

from sklearn import feature_extraction

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

def datetime_to_unix_timestamp(date_time):
    return calendar.timegm(date_time.utctimetuple())


@feature
@FunctionDictTransformer
def time_utc(X):
    return [datetime_to_unix_timestamp(dateutil.parser.parse(row['created_time'])) for row in X]


@feature
@FunctionDictTransformer
def time_of_day(X):
    for row in X:
      created_time = dateutil.parser.parse(row['created_time'])
      yield datetime_to_unix_timestamp(created_time.time())


# TODO handle space differently (with a RBF svm)

# TODO wire these (and others) together

# TODO hyperoptimization