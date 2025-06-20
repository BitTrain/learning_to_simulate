from math import pi
from typing import Optional

import numpy as np
import tensorflow as tf

from .data import TF_NUMERIC_TO_INT
from .debug import print_serialized_tf_tensor


@tf.function
def sample_from_logits(
    logits:      tf.Tensor,
    dtype:       tf.DType=tf.int32,
    *,
    method:      str="categorical",
    temperature: float=1.0,
) -> tf.Tensor:
    """
    Sample indexes from logits using different methods.

    Args:
        logits: A 'tf.Tensor' of shape '(*batch_dims, num_classes)' with input logits.
        dtype: Signed integer 'tf.DType' for output indexes.
        method: Sampling method. One of "categorical" and "greedy".
        temperature: Temperature scaling for categorical sampling.
    Returns:
        A 'tf.Tensor' of shape '(*batch_dims,)' with the index of each sampled class.
    """
    def _categorical(logits, temperature):
        logits = logits / tf.cast(temperature, logits.dtype)
        flat_logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])  # For `categorical`
        indexes = tf.random.categorical(flat_logits, num_samples=1, dtype=dtype)
        indexes = tf.reshape(indexes[..., 0], tf.shape(logits)[:-1])
        return indexes
    def _greedy(logits):
        return tf.argmax(logits, axis=-1, output_type=dtype)

    indexes = tf.case(
        [
            (tf.equal(method, "categorical"), lambda: _categorical(logits, temperature)),
            (tf.equal(method, "greedy"), lambda: _greedy(logits)),
        ],
        default=lambda: _categorical(logits, temperature),
        exclusive=True
    )
    return indexes

@tf.function
def from_numeric(
    tensor:        tf.Tensor,
    bitwave_sizes: tf.Tensor,
    bitqueue_size: int
) -> tf.Tensor:
    int_type = TF_NUMERIC_TO_INT[tensor.dtype.name]

    tensor = tf.expand_dims(tensor, axis=-1)  # For broadcasting with bitwave sizes tensor
    tensor = tf.bitcast(tensor, int_type)  # For bitwise ops   

    bitqueue_shifts = tf.reduce_sum(bitwave_sizes) - bitqueue_size - tf.cumsum(bitwave_sizes, exclusive=True)
    bitqueue_mask   = tf.bitwise.left_shift(tf.constant(1, dtype=int_type), bitqueue_size) - 1
    bitqueue_tensor = tf.bitwise.bitwise_and(
        tf.where(
            bitqueue_shifts > 0,
            tf.bitwise.right_shift(tensor, bitqueue_shifts),
            tf.bitwise.left_shift(tensor, -bitqueue_shifts)
        ),
        bitqueue_mask
    )

    return bitqueue_tensor

@tf.function
def to_numeric(
    bits:        tf.Tensor,
    num_bits:    int,
    lead_bitpos: int,
    dtype:       tf.DType,
) -> tf.Tensor:
    shift = lead_bitpos - num_bits
    shifted_bits = tf.bitwise.left_shift(bits, shift)

    numeric = tf.bitcast(shifted_bits, dtype)

    return numeric

@tf.function
def is_zero(
    bits:  tf.Tensor,
    start: int,
    end:   int,
) -> bool:
    dtype = bits.dtype
    bit_length = 8 * dtype.size

    mask = tf.constant(1, dtype)
    mask = tf.bitwise.left_shift(mask, end - start) - mask
    mask = tf.bitwise.left_shift(mask, bit_length - end)

    is_zero = tf.equal(tf.bitwise.bitwise_and(bits, mask), 0)

    return is_zero

@tf.py_function(Tout=[])
def validate_bitwave_sizes(
    num_typename:  str,
    bitwave_sizes: tf.Tensor,
    bitqueue_size: Optional[int]=None
) -> None:
    """Check consistency of bitwave sizes with a numeric datatype and to an optional bitqueue size."""
    num_typename  = num_typename.numpy().decode("utf-8")
    num_type      = np.dtype(num_typename)
    int_type      = TF_NUMERIC_TO_INT[num_typename].as_numpy_dtype
    bitwave_sizes = bitwave_sizes.numpy()
    bitqueue_size = bitqueue_size.numpy()

    # Check compatibility with datatype bit length (e.g. 32 for float32)
    bit_length = np.iinfo(int_type).bits
    total_bits = sum(bitwave_sizes)
    if total_bits != bit_length:
        raise ValueError(f"Total bit count in bitwaves ({total_bits}) does not match that of output type {num_type} "
                         f"({bit_length})")

    # Check compatibility with bitqueue size
    if bitqueue_size:
        num_bitwaves = len(bitwave_sizes)
        num_bits = 0  # Bits in queue
        trail_id = 0  # Next bitwave outside queue
        for lead_id, _ in enumerate(bitwave_sizes):
            while num_bits < bitqueue_size and trail_id < num_bitwaves:
                num_bits += bitwave_sizes[trail_id]  # Enqueue next bitwave
                if num_bits > bitqueue_size:
                    raise ValueError(f"Bitwave nr {trail_id+1} of size {bitwave_sizes[trail_id]} overshoots the bitqueue "
                                    f"after {lead_id} shifts.")
                trail_id += 1
            num_bits -= bitwave_sizes[lead_id]  # Dequeue leading bitwave


if __name__ == "__main__":
    print("≈≈≈ [utils.bitqueue] ≈≈≈")
    print('')

    print("~~~ [sample_from_logits] ~~~")
    print("Example:")
    logits = tf.random.uniform((1, 4))
    np.set_printoptions(precision=4)
    print(logits.numpy()[0])
    num_samples = 100
    results = [0, 0, 0, 0]
    for _ in range(num_samples):
        bits = sample_from_logits(logits, temperature=0.1)
        i = int(bits[0])
        results[i] += 1
    print(f"{num_samples} samples: {results}")

    print("~~~ [validate_bitwave_sizes] ~~~")
    print("Example:")
    num_type=tf.float32
    bitwave_sizes = tf.constant([9, 5, 4, 3, 2, 2, 2, 2, 1, 1, 1], dtype=TF_NUMERIC_TO_INT[num_type.name])
    bitqueue_size = 9
    print(f">>> Validating bitwave sizes {bitwave_sizes} on 32-bits and bitqueue size {bitqueue_size}")
    validate_bitwave_sizes(num_type.name, bitwave_sizes, bitqueue_size)
    print(f">>> Sizes valid.")
    print('')

    print("~~~ [from_numeric] ~~~")
    print("Example:")
    tensor = tf.constant(pi, dtype=num_type)
    print(">>> Serializing FP32(π) ≈ 3.14159")
    print_serialized_tf_tensor(tf.expand_dims(tensor, axis=-1), bitwave_sizes)
    print(f">>> Extracting bitqueues of size {bitqueue_size}")
    bitqueue_tensor = from_numeric(tensor, bitwave_sizes, bitqueue_size)
    print_serialized_tf_tensor(bitqueue_tensor, bitwave_sizes)
    print('')