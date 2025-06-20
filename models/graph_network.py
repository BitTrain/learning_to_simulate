import tensorflow as tf
import tensorflow_gnn as tfgnn

from learning_to_simulate.layers.mlp import mlp
from learning_to_simulate.layers.message_passing import CustomVanillaMPNNGraphUpdate


class EncodeProcessDecode(tf.keras.Model):
    """
    @Sanchez-Gonzalez et al. (2020)
        Model ported to TensorFlow/Keras from TensorFlow/Sonnet.
    """
    def __init__(
        self,
        *,
        output_dim:         int,
        latent_dim:         int=128,  # @ S-G, p. 4-5
        message_dim:        int=128,  # @ S-G, p. 4-5
        mlp_depth:          int=2,    # @ S-G, p. 4
        num_message_passes: int=10,
        shared_params:      bool=False,
        reduce_type:        str="sum",
        node_set_name:      str="particles",
        name:               str="EncodeProcessDecode",
    ):
        super().__init__(name=name)

        self._context_embedding = mlp(latent_dim,
                                      mlp_depth,
                                      latent_dim,
                                      use_layer_normalization=True,
                                      name="context_embedding")
        self._node_embedding = mlp(latent_dim,
                                   mlp_depth,
                                   latent_dim,
                                   use_layer_normalization=True,
                                   name="node_embedding")
        self._edge_embedding = mlp(latent_dim,
                                   mlp_depth,
                                   latent_dim,
                                   use_layer_normalization=True,
                                   name="edge_embedding")

        def embedding_fn(embedding_layer):
            def wrapper(graph_piece, **kwargs):
                """Embedding on concatenated features."""
                features = graph_piece.get_features_dict()
                if features:
                    return embedding_layer(
                        tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())])
                    )
                else:
                    tf.get_logger().warning(f"No features found in graph piece: {graph_piece}. "
                                            "Falling back on MakeEmptyFeature.")
                    return tfgnn.keras.layers.MakeEmptyFeature()(graph_piece)
            return wrapper
            
        # MLP encoders
        self._encoder = tfgnn.keras.layers.MapFeatures(
            context_fn=embedding_fn(self._context_embedding),
            node_sets_fn=embedding_fn(self._node_embedding),
            edge_sets_fn=embedding_fn(self._edge_embedding),
            name="input_embedding"
        )

        # Vanilla MPNN processors with skip connections
        if shared_params:
            self._processors = num_message_passes * [
               CustomVanillaMPNNGraphUpdate(
                    update_dim=latent_dim,
                    message_dim=message_dim,
                    update_depth=mlp_depth,
                    message_depth=mlp_depth,
                    hidden_activation="relu",
                    receiver_tag=tfgnn.TARGET,
                    edge_feature=tfgnn.HIDDEN_STATE,
                    reduce_type=reduce_type,
                    use_layer_normalization=True,
                    next_state_layer=tfgnn.keras.layers.ResidualNextState
                )
            ]
        else:
            self._processors = [
                CustomVanillaMPNNGraphUpdate(
                    update_dim=latent_dim,
                    message_dim=message_dim,
                    update_depth=mlp_depth,
                    message_depth=mlp_depth,
                    hidden_activation="relu",
                    receiver_tag=tfgnn.TARGET,
                    edge_feature=tfgnn.HIDDEN_STATE,
                    reduce_type=reduce_type,
                    use_layer_normalization=True,
                    next_state_layer=tfgnn.keras.layers.ResidualNextState
                )
                for _ in range(num_message_passes)
            ]

        # MLP decoder on nodes' updated hidden states
        self._decoder = tf.keras.Sequential(
           [
                tfgnn.keras.layers.Readout(node_set_name=node_set_name),
                mlp(latent_dim, mlp_depth, output_dim)
           ],
           name="prediction_head"
        )

    def call(self, input_graph: tfgnn.GraphTensor, training: bool=False):
        latent_graph = self._encoder(input_graph, training=training)
        for processor in self._processors:
            latent_graph = processor(latent_graph, training=training)
        outputs = self._decoder(latent_graph, training=training)
        
        return outputs