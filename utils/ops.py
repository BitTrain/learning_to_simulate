import tensorflow as tf


@tf.function
def swap_dims(tensor: tf.Tensor, i: int, j: int) -> tf.Tensor:
    """Swaps two dimensions of a tensor with or without batching."""
    rank = tf.rank(tensor)
    perm = tf.range(rank)
    
    # Convert negative indexes like -1, -2 to positive equivalents
    i = tf.where(i < 0, rank + i, i)
    j = tf.where(j < 0, rank + j, j)
    
    # Swap i and j in permutation tensor
    perm = tf.tensor_scatter_nd_update(perm, [[i], [j]], [j, i])

    return tf.transpose(tensor, perm)