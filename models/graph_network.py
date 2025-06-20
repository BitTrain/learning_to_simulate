import tensorflow as tf
import tensorflow_gnn as tfgnn

from learning_to_simulate.layers.message_passing import CustomVanillaMPNNGraphUpdate
from learning_to_simulate.utils import bitqueue
from learning_to_simulate.utils.data import TF_NUMERIC_TO_INT


class EncodeProcessDecode(tf.keras.Model):
    """
    @Sanchez-Gonzalez et al. (2020)
        Model ported to TensorFlow/Keras from TensorFlow/Sonnet.
    """
    def __init__(
        self,
        *,
        output_dim:    int,
        bitwave_sizes: tuple[int, ...],
        bitqueue_size: int,
        latent_dim:    int=128,  # @ S-G, p. 4-5
        message_dim:   int=128,  # @ S-G, p. 4-5
        mlp_depth:     int=2,    # @ S-G, p. 4
        shared_params: bool=False,
        reduce_type:   str="sum",
        node_set_name: str="particles",
        name:          str="EncodeProcessDecode",
    ):
        super().__init__(name=name)

        self._bitwave_sizes = bitwave_sizes
        self._num_bitwaves = len(bitwave_sizes)
        self._bitqueue_size = bitqueue_size
        self._bitqueue_range = 1 << bitqueue_size

        self._node_set_name = node_set_name

        self._acc_embed = tf.keras.layers.Dense(latent_dim)
        self._node_embed = tf.keras.layers.Dense(latent_dim)
        self._edge_embed = tf.keras.layers.Dense(latent_dim)

        def node_embedding_fn(node_set, **kwargs):
            features = node_set.get_features_dict()
            acc = features.pop("acceleration")
            base = tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())])
            return {
                tfgnn.HIDDEN_STATE: self._node_embed(base),
                "acceleration": self._acc_embed(acc),
            }

        def edge_embedding_fn(edge_set, **kwargs):
            features = edge_set.get_features_dict()
            return self._edge_embed(
                tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())])
            )
            
        self._encoder = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=node_embedding_fn,
            edge_sets_fn=edge_embedding_fn,
            name="input_embedding"
        )

        # Vanilla MPNN processors with skip connections
        if shared_params:
            self._processors = self._num_bitwaves * [
               CustomVanillaMPNNGraphUpdate(
                    update_dim=latent_dim,
                    message_dim=message_dim,
                    update_depth=mlp_depth,
                    message_depth=mlp_depth,
                    hidden_activation="relu",
                    receiver_tag=tfgnn.TARGET,
                    edge_feature=tfgnn.HIDDEN_STATE,
                    extra_node_feature="acceleration",
                    reduce_type=reduce_type,
                    use_layer_normalization=True,
                    # next_state_layer=tfgnn.keras.layers.ResidualNextState
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
                    extra_node_feature="acceleration",
                    reduce_type=reduce_type,
                    use_layer_normalization=True,
                    # next_state_layer=tfgnn.keras.layers.ResidualNextState
                )
                for _ in range(self._num_bitwaves)
            ]

        # Linear decoder on nodes' updated hidden states
        self._decoders = [
            tf.keras.Sequential([
                tfgnn.keras.layers.Readout(node_set_name=node_set_name),
                tf.keras.layers.Dense(output_dim * self._bitqueue_range),
            ], name=f"prediction_head_{i}")
            for i in range(self._num_bitwaves)
        ]

    def call(
        self,
        input_graph:  tfgnn.GraphTensor,
        acceleration: tf.Tensor,
        training:     bool=False
    ):
        dtype = acceleration.dtype
        int_type = TF_NUMERIC_TO_INT[dtype.name]

        latent_graph = self._encoder(input_graph, training=training)

        logits_list = []
        rem = sum(self._bitwave_sizes)
        for bw_size, processor, decoder in zip(self._bitwave_sizes, self._processors, self._decoders):
            node_set = latent_graph.node_sets[self._node_set_name]
            latent_graph = latent_graph.replace_features(node_sets={
                self._node_set_name: {
                    tfgnn.HIDDEN_STATE: node_set[tfgnn.HIDDEN_STATE],
                    "acceleration": self._acc_embed(acceleration),
                }
            })
            latent_graph = processor(latent_graph, training=training)
            logits = decoder(latent_graph, training=training)
            logits = tf.reshape(logits, (*acceleration.shape, self._bitqueue_range))
            logits_list.append(logits)
            bits = bitqueue.sample_from_logits(logits, dtype=int_type)
            delta = bitqueue.to_numeric(bits, self._bitqueue_size, rem, dtype)
            acceleration += delta
            rem -= bw_size
        
        return acceleration, logits_list