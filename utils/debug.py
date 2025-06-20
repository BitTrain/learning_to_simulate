from itertools import accumulate
from math import ceil, log10
from struct import pack
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf


def print_serialized_tf_tensor(
    tensor:        tf.Tensor,
    bitwave_sizes: Optional[Iterable]=None
):
    """Print the big-endian binary representation of each tensor element, optionally split into bitwaves."""
    dtype = tensor.dtype.as_numpy_dtype
    shape = tensor.shape.as_list()
    fmt = {np.uint8: 'B', np.uint16:  'H', np.uint32:  'I', np.uint64:  'Q',
           np.int8:  'b', np.int16:   'h', np.int32:   'i', np.int64:   'q',
                          np.float16: 'e', np.float32: 'f', np.float64: 'd'}[dtype]
    
    decimal_digits = lambda x: ceil(log10(x)) if x > 0 else 1
    index_width = sum(decimal_digits(dim) + 1 if dim > 0 else 2 for dim in shape) + 2  # +2 for parenthesis
    if np.issubdtype(dtype, np.integer):
        elem_fmt = "{}"
        min_val = np.iinfo(dtype).min
        elem_width = len(f"{min_val}")
    else:  # np.floating
        precision = np.finfo(dtype).precision
        elem_fmt = f"{{:.{precision}g}}"
        min_val = np.finfo(dtype).min
        elem_width = len(elem_fmt.format(min_val))

    for index, elem in np.ndenumerate(tensor.numpy()):
        index_str = f"{index}".ljust(index_width)
        elem_str = elem_fmt.format(elem).rjust(elem_width)
        print(index_str + ' ' + elem_str, end=" : ")
        bit_str = ''.join(f'{byte:08b}' for byte in pack('>' + fmt, elem))
        if bitwave_sizes is not None:  # Split printout into bitwaves
            starts = accumulate(bitwave_sizes, initial=0)
            ends = accumulate(bitwave_sizes)
            slices = (bit_str[start:end] for start, end in zip(starts, ends))
            print(' '.join(slices))
        else:
            print(bit_str)