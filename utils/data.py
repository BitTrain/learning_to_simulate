import collections

import tensorflow as tf
import numpy as np


SQRT_EPS = {
    typename: tf.constant(np.sqrt(np.finfo(np.dtype(typename)).eps))
    for typename in ("float16", "float32", "float64")
}

StatsPair = collections.namedtuple("StatsPair", ["mean", "std"])