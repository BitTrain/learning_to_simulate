from typing import Collection, Optional

import tensorflow as tf
import tensorflow_gnn as tfgnn


def CustomVanillaMPNNGraphUpdate(
    *,
    update_dim:              int,
    message_dim:             int,
    update_depth:            int,
    message_depth:           int,
    receiver_tag:            tfgnn.IncidentNodeTag,
    node_set_names:          Optional[Collection[tfgnn.NodeSetName]]=None,
    edge_feature:            Optional[tfgnn.FieldName]=None,
    reduce_type:             str="sum",
    use_layer_normalization: bool=False,
    next_state_layer:        tf.keras.layers.Layer=tfgnn.keras.layers.NextStateFromConcat,
    **dense_layer_kwargs
) -> tf.keras.layers.Layer:
    """
    Builds a customizable tfgnn.GraphUpdate layer based on the vanilla message-passing neural network (MPNN).

    Args:
        update_dim: Output dimension of the updated node states.
        message_dim: Dimension of message vectors.
        update_depth: Number of Dense layers in the node update MLP.
        message_depth: Number of Dense layers in the message MLP.
        receiver_tag: Which node of each edge receives the message. One of
            `tfgnn.TARGET` or `tfgnn.SOURCE`.
        node_set_names: The names of node sets to update. If unset, updates all
            that are on the receiving end of any edge set.
        edge_feature: Can be set to a feature name of the edge set to select
            it as an input feature. By default, this set to `None`, which disables
            this input.
        reduce_type: How to pool the messages from edges to receiver nodes; defaults
            to `"sum"`. Can be any reduce_type understood by `tfgnn.pool()`, including
            concatenations like `"sum|mean"`.  
        use_layer_normalization: Flag to determine whether to apply layer
            normalization after the fully connected layers.
        next_state_layer: Wraps around a node state update transformation on the
            combined graph inputs.

    Returns:
        A GraphUpdate layer built from the tfgnn.keras.ConvGNNBuilder factory for use 
        on a scalar GraphTensor with `tfgnn.HIDDEN_STATE` features on the node sets.
    """
    def mlp(units, depth, *, use_layer_normalization=False, name="mlp", **dense_layer_kwargs):
        layers = tf.keras.Sequential(name=name)
        for _ in range(depth):
            layers.add(tf.keras.layers.Dense(units, **dense_layer_kwargs))
        if use_layer_normalization:
            layers.add(tf.keras.layers.LayerNormalization())
        return layers

    gnn_builder = tfgnn.keras.ConvGNNBuilder(
        lambda edge_set_name, receiver_tag: tfgnn.keras.layers.SimpleConv(
            mlp(message_dim, 
                message_depth,
                use_layer_normalization=use_layer_normalization,
                name="mlp_message",
                **dense_layer_kwargs),
            reduce_type,
            receiver_tag=receiver_tag,
            sender_edge_feature=edge_feature
        ),
        lambda node_set_name: next_state_layer(
            mlp(update_dim,
                update_depth,
                use_layer_normalization=use_layer_normalization,
                name="mlp_update",
                **dense_layer_kwargs
            )
        ),
        receiver_tag=receiver_tag
    )

    return gnn_builder.Convolve(node_set_names)