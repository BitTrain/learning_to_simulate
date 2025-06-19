import tensorflow as tf

def mlp(
    hidden_units, 
    depth, 
    output_units,
    *, 
    hidden_activation="relu", 
    use_layer_normalization=False,
    name="mlp"
) -> tf.keras.layers.Layer:
    layers = tf.keras.Sequential(name=name)
    for _ in range(depth - 1):
        layers.add(tf.keras.layers.Dense(hidden_units, activation=hidden_activation))
    layers.add(tf.keras.layers.Dense(output_units, activation="linear"))
    if use_layer_normalization:
        layers.add(tf.keras.layers.LayerNormalization(epsilon=1e-7))
    return layers