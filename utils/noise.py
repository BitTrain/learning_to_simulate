from typing import Optional

import tensorflow as tf


@tf.function
def random_walk_noise(
    positions:  tf.RaggedTensor,
    *,
    target_std: float=1.0,
    mask:       Optional[tf.RaggedTensor]=None
) -> tf.RaggedTensor:
    """
    @Sanchez-Gonzalez et al. (2020)
        Generates a random walk in velocity space on unmasked particles such that the cumulative noise on the final
        velocity has standard deviation 'target_std'. Returns the corresponding time-correlated noise in position space.
    """
    dtype = positions.dtype
    num_particles = positions.row_splits
    
    positions = positions.flat_values
    mask = mask.flat_values

    shape = tf.shape(positions)
    num_steps = tf.cast(shape[-2] - 1, dtype)

    if mask is None:
        mask = tf.ones_like(positions)  # No masking
    else:
        mask = tf.cast(mask, dtype)

    # Gaussian noise at each time step and unmasked particle index
    stddev = tf.cast(target_std, dtype) / tf.math.sqrt(num_steps)
    noise = mask * tf.random.normal(shape, stddev=stddev, dtype=dtype)

    # Random walk in velocity space
    random_walk = tf.cumsum(noise, axis=-2)

    # Integrated noise in position space
    position_noise = tf.cumsum(random_walk, exclusive=True, axis=-2)  # No noise on first positions
    position_noise = tf.RaggedTensor.from_row_splits(position_noise, num_particles)

    return position_noise