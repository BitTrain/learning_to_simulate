import collections

import tensorflow as tf
import numpy as np


TF_NUMERIC_TO_INT = {
    "int8":  tf.int8,
    "uint8": tf.int8,

    "int16":   tf.int16,
    "uint16":  tf.int16,
    "float16": tf.int16,

    "int32":   tf.int32,
    "uint32":  tf.int32,
    "float32": tf.int32,

    "int64":     tf.int64,
    "uint64":    tf.int64,
    "float64":   tf.int64,
    "complex64": tf.int64
}

SQRT_EPS = {
    typename: tf.constant(np.sqrt(np.finfo(np.dtype(typename)).eps))
    for typename in ("float16", "float32", "float64")
}

StatsPair = collections.namedtuple("StatsPair", ["mean", "std"])