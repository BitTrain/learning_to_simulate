import tensorflow as tf

from learning_to_simulate.utils.data import TF_NUMERIC_TO_INT
from learning_to_simulate.utils.bitqueue import validate_bitwave_sizes

    
@tf.function
def bitwise_regression_loss(
    target_bits: tf.Tensor,
    bit_probs:   tf.Tensor,
    bit_combos:  tf.Tensor,
    num_bits:    int,
    loss_blend:  float=0.5,
) -> tf.Tensor:
    """
    Differentiable weighted sum of cross-entropy and mean squared loss.
    """
    loss_dtype = bit_probs.dtype

    cce = -tf.math.log(
                tf.gather(bit_probs, target_bits, batch_dims=target_bits.shape.rank)
            ) / tf.math.log(tf.cast(2.0, loss_dtype))
    
    err = tf.expand_dims(target_bits, axis=-1) - bit_combos
    mse = tf.reduce_sum(bit_probs * tf.cast(tf.square(err), loss_dtype), axis=-1)

    expected_cce = tf.cast(num_bits, loss_dtype)
    expected_mse = tf.cast(tf.bitwise.left_shift(1, 2 * num_bits) - 1, loss_dtype) / 6.0
    loss_ratio = expected_cce / expected_mse  # NOTE Ideally pass pre-calculated

    loss_blend = tf.cast(loss_blend, loss_dtype)
    loss = loss_blend * cce + loss_ratio * (1 - loss_blend) * mse
    
    return tf.reduce_mean(loss)


if __name__ == "__main__":
    print("≈≈≈ [core.losses] ≈≈≈")
    print('')

    print("~~~ [bitwise_regression_loss] ~~~")
    print("Examples:")
    bitqueue_size  = 9
    bitqueue_range = 1 << bitqueue_size
    bitwave_sizes  = (9, 5, 4, 3, 2, 2, 2, 2, 1, 1, 1)
    print(f"bitqueue size: {bitqueue_size}\nbitwave sizes: {bitwave_sizes}")
    num_bitwaves   = len(bitwave_sizes)

    float_typename = "float32"
    validate_bitwave_sizes(float_typename,
                           tf.constant(bitwave_sizes),
                           bitqueue_size)
    int_type       = TF_NUMERIC_TO_INT[float_typename]

    batch_size     = 32
    num_features   = 128
    input_shape    = (batch_size, num_features)
    output_shape   = (3,)
    target_bits    = tf.random.uniform(shape=(*input_shape, *output_shape, num_bitwaves),
                                       minval=0,
                                       maxval=bitqueue_range,
                                       dtype=int_type)
    float_type     = tf.as_dtype(float_typename)
    bit_probs      = tf.ones(shape=(*target_bits.shape, bitqueue_range), dtype=float_type) / bitqueue_range

    for i in range(num_bitwaves):
        loss = bitwise_regression_loss(tf.gather(target_bits, i, axis=-1),
                                       tf.gather(bit_probs, i, axis=-2),
                                       tf.range(1 << bitqueue_size, dtype=int_type),
                                       tf.constant(bitqueue_size))
        print(f"Bitwave {i:<{len(str(num_bitwaves - 1))}} loss: {loss.numpy():.6f}")
    
    print('')

    bitqueue_size  = 12
    bitqueue_range = 1 << bitqueue_size
    bitwave_sizes  = (12, 7, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    print(f"bitqueue size: {bitqueue_size}\nbitwave sizes: {bitwave_sizes}")
    num_bitwaves   = len(bitwave_sizes)

    float_typename = "float64"
    float_type     = tf.as_dtype(float_type)
    validate_bitwave_sizes(float_typename,
                           tf.constant(bitwave_sizes),
                           bitqueue_size)
    int_type       = TF_NUMERIC_TO_INT[float_typename]

    batch_size     = 32
    num_features   = 128
    input_shape    = (batch_size, num_features)
    output_shape   = (3,)
    target_bits    = tf.random.uniform(shape=(*input_shape, *output_shape, num_bitwaves),
                                       minval=0,
                                       maxval=bitqueue_range,
                                       dtype=int_type)
    bit_probs      = tf.ones(shape=(*target_bits.shape, bitqueue_range), dtype=float_type) / bitqueue_range

    for i in range(num_bitwaves):
        loss = bitwise_regression_loss(tf.gather(target_bits, i, axis=-1),
                                       tf.gather(bit_probs, i, axis=-2),
                                       tf.range(1 << bitqueue_size, dtype=int_type),
                                       tf.constant(bitqueue_size))
        print(f"Bitwave {i:<{len(str(num_bitwaves - 1))}} loss: {loss.numpy():.6f}")
    
    print('')
