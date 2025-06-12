import json
from typing import Any, Iterable, Mapping, Optional, Union, Tuple

import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np
from sklearn.neighbors import KDTree

from learning_to_simulate.models.graph_network import EncodeProcessDecode
from learning_to_simulate import utils, settings
from utils.data import SQRT_EPS, StatsPair


class LearnedSimulator(tf.keras.Model):
    """
    @Sanchez-Gonzalez et al. (2020)
        Core simulation model ported to TensorFlow/Keras from TensorFlow/Sonnet.
    """
    def __init__(
        self,
        *,
        dim:                         int,
        cutoff_radius:               float,
        boundaries:                  Iterable[Iterable[float]],
        noise_std:                   float,
        normalization_stats:         Mapping[str, StatsPair],
        num_particle_types:          int,

        static_particle_type_id:     int=3,
        particle_type_embedding_dim: int=16,
        velocity_context_size:       int=5,
        self_interaction:            bool=False,
        graph_network:               tf.keras.Model=EncodeProcessDecode,
        graph_network_kwargs:        Optional[Mapping[str, Any]]=None,
        dtype:                       Optional[Union[tf.DType, str]]=None,
        name:                        str="LearnedSimulator"
    ):
        super().__init__(dtype=dtype, name=name)

        self._dim = dim
        self._cutoff_radius = cutoff_radius
        self._boundaries = boundaries
        self._noise_std = noise_std
        self._normalization_stats = normalization_stats
        self._num_particle_types = num_particle_types
        self._static_particle_type_id = static_particle_type_id
        self._velocity_context_size = velocity_context_size
        self._self_interaction = self_interaction
        self._gnn = graph_network(output_dim=dim, **graph_network_kwargs if graph_network_kwargs else {})
        self._particle_type_embedding = tf.keras.layers.Embedding(
            input_dim=num_particle_types,
            output_dim=particle_type_embedding_dim
        )

        # Metrics
        self.acc_mse = tf.keras.metrics.MeanSquaredError(name="acc_mse")
        self.step_mse = tf.keras.metrics.MeanSquaredError(name="step_mse")
        self.rollout_mse = tf.keras.metrics.MeanSquaredError(name="rollout_mse")

    def get_config(self):
        return {
            "dim": self._dim,
            "cutoff_radius": self._cutoff_radius,
            "boundaries": json.dumps(self._boundaries),
            "noise_std": self._noise_std,
            "normalization_stats": json.dumps({
                key: {"mean": value.mean.numpy().tolist(), "std": value.std.numpy().tolist()}
                for key, value in self._normalization_stats.items()
            }),
            "num_particle_types": self._num_particle_types,
            "static_particle_type_id": self._static_particle_type_id,
            "velocity_context_size": self._velocity_context_size,
            "dtype": self.dtype
        }

    @classmethod
    def from_config(cls, config):
        config["boundaries"] = json.loads(config["boundaries"])
        config["normalization_stats"] = {
            key: StatsPair(tf.constant(value["mean"], config["dtype"]), tf.constant(value["std"], config["dtype"]))
            for key, value in json.loads(config["normalization_stats"]).items()
        }
        return cls(**config)

    @property
    def metrics(self):
        return [self.acc_mse, self.step_mse, self.rollout_mse]

    def call(
        self,
        inputs:   Mapping[str, tf.Tensor | tf.RaggedTensor],
        training: bool=False
    ) -> tf.Tensor:
        """Forward pass of the model, producing normalized accelerations."""
        positions = inputs["positions"]
        particle_type = inputs["particle_type"]
        global_context = inputs.get("global_context")

        if settings.TF_DEBUG_MODE:
            self._check_tensor_inputs(positions, particle_type, global_context)
            
        input_graph = self._build_graph_tensor(positions, particle_type, global_context)
        norm_acc = self._gnn(input_graph, training)

        return norm_acc
    
    def train_step(
        self,
        inputs: Mapping[str, tf.Tensor | tf.RaggedTensor]
    ) -> Mapping[str, tf.Tensor]:
        """One-step training."""
        positions = inputs["positions"]
        particle_type = inputs["particle_type"]

        # Corrupt input positions with random walk noise
        is_not_static = tf.not_equal(particle_type, self._static_particle_type_id)[..., tf.newaxis, tf.newaxis]
        position_noise = utils.noise.random_walk_noise(positions, target_std=self._noise_std, mask=is_not_static)
        positions += position_noise

        # Split off last step as target
        target_pos = positions[..., -1, :]
        positions = positions[..., :-1, :]

        # Get predicted and target normalized acceleration and compute loss
        with tf.GradientTape() as tape:
            pred_acc = self(inputs, training=True)
            target_acc = self._differentiate_position(
                positions, 
                target_pos
            )
            loss_acc = tf.reduce_mean(tf.square(target_acc.flat_values - pred_acc))

        # Backprop
        gradients = tape.gradient(loss_acc, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.acc_mse.update_state(target_acc.flat_values, pred_acc)

        return { "loss": self.acc_mse.result() }

    def test_step(
        self,
        inputs: Mapping[str, tf.Tensor | tf.RaggedTensor]
    ) -> Mapping[str, tf.Tensor]:
        """One-step evaluation."""
        positions = inputs["positions"]
        num_particles = positions.row_splits

        # Split off last step as target
        target_pos = positions[..., -1, :]
        positions = positions[..., :-1, :]

        # Get predicted acceleration and compute error
        pred_acc = self(inputs)
        target_acc = self._differentiate_position(
            positions, 
            target_pos
        )
        self.acc_mse.update_state(target_acc.flat_values, pred_acc)

        # Get predicted next position and compute error
        pred_acc = tf.RaggedTensor.from_row_splits(pred_acc, num_particles)
        pred_pos = self._integrate_acceleration(positions, pred_acc)
        self.step_mse.update_state(target_pos.flat_values, pred_pos.flat_values)

        return {
            "loss": self.acc_mse.result(),
            "err": self.step_mse.result()
        }

    def rollout(
        self,
        inputs:    Mapping[str, tf.Tensor],
        *,
        num_steps: int=1
    ) -> Mapping[str, tf.Tensor]:
        """Multi-step rollout evaluation."""
        positions = inputs["positions"]
        particle_type = inputs["particle_type"]
        global_contexts = inputs.get("global_contexts")

        num_seed = self._velocity_context_size + 1
        initial_pos, ground_truth_pos = tf.split(positions, [num_seed, -1], axis=-2)

        is_static = tf.equal(particle_type, self._static_particle_type_id)  # Kinematic mask
        is_static = tf.expand_dims(is_static, axis=-1)  # For broadcasting

        def step_fn(step, seed_pos, pred_pos):
            # Get global context at current time step
            if global_contexts is None:
                inputs["global_context"] = None
            else:
                inputs["global_context"] = global_contexts[step + num_seed - 1][tf.newaxis]

            # Update positions
            inputs["positions"] = seed_pos
            pred_acc = self(inputs)
            pred_acc = tf.reshape(pred_acc, tf.shape(seed_pos[..., -1, :]))  # Restore batch dims
            next_pos = tf.where(
                is_static,
                ground_truth_pos[..., step, :],  # Use frozen ground truth for static particles
                self._integrate_acceleration(seed_pos, pred_acc)
            )

            # Update loop variables
            pred_pos = pred_pos.write(step, next_pos)
            seed_pos = tf.concat([seed_pos[..., 1:, :], next_pos[..., tf.newaxis, :]], axis=-2)
            step = step + 1

            return step, seed_pos, pred_pos
        
        pred_pos = tf.TensorArray(size=num_steps, dtype=self.dtype)
        _, _, pred_pos = tf.while_loop(
            cond=lambda *loop_vars: tf.less(loop_vars[0], num_steps),
            body=step_fn,
            loop_vars=(0, initial_pos, pred_pos),
            parallel_iterations=1
        )

        # Trim down ground truth to num_steps
        ground_truth_pos = ground_truth_pos[:, :num_steps, :]

        # Align positions
        pred_pos = pred_pos.stack()
        initial_pos = utils.ops.swap_dims(initial_pos, -2, -3)
        ground_truth_pos = utils.ops.swap_dims(ground_truth_pos, -2, -3)

        # Compute error over full rollout and update metrics
        self.rollout_mse.update_state(ground_truth_pos, pred_pos)

        outputs = {
            "initial_positions": initial_pos,
            "predicted_rollout": pred_pos,
            "ground_truth_rollout": ground_truth_pos,
            "particle_types": particle_type,
            "rollout_mse": self.rollout_mse.result()
        }

        if global_contexts is not None:
             outputs["global_contexts"] = global_contexts
        
        return outputs
    
    def _check_tensor_inputs(
        self,
        positions:      tf.Tensor | tf.RaggedTensor,
        particle_type:  tf.Tensor | tf.RaggedTensor,
        global_context: Optional[tf.Tensor]=None,
        *,
        num_steps:      int=1
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # --- Shapes ---
        if isinstance(positions, tf.Tensor):
            positions_shape = tf.shape(positions)
            D = positions_shape[-1]
            T = positions_shape[-2]
            N = positions_shape[-3]
            B = positions_shape[-4]
        else:  # `tf.RaggedTensor`
            flat_positions_shape = tf.shape(positions.flat_values)
            D = flat_positions_shape[-1]
            T = flat_positions_shape[-2]
            N = positions.row_lengths()  # Non-uniform!
            B = positions.nrows()
        G = tf.shape(global_context)[-1] if global_context is not None else tf.constant([])
        # tf.print("Input shapes:", "\nB =", B, "\nN =", N, "\nT =", T, "\nD =", D, "\nG =", G)

        # --- Dimensions ---
        tf.debugging.assert_equal(
            D, self._dim,
            message="Input `positions` have incorrect spatial dimension, "
                f"expected {self._dim}"
        )
        tf.debugging.assert_greater_equal(
            T, num_steps + self._velocity_context_size,
            message="Not enough time steps in input `positions` to compute prior velocities, "
                f"expected at least {num_steps + self._velocity_context_size}"
        )

        # --- Types ---
        if not particle_type.dtype.is_integer:
            raise TypeError(f"`particle_type` must be an integer tensor, but got {particle_type.dtype}")
        tf.debugging.assert_same_float_dtype([positions, global_context], tf.as_dtype(self.dtype))

        return B, N, T, D, G

    def _build_graph_tensor(
        self,
        positions:      tf.Tensor | tf.RaggedTensor,
        particle_type:  tf.Tensor | tf.RaggedTensor,
        global_context: Optional[tf.Tensor]=None
    ) -> tfgnn.GraphTensor:
        # --- Global features ---
        context_stats = self._normalization_stats.get("context")
        global_features = {
            "global_context": (global_context - context_stats.mean) / 
                tf.math.maximum(context_stats.std, SQRT_EPS[self.dtype])
        } if global_context is not None else {}

        # --- Node features ---
        embedded_particle_type = self._particle_type_embedding(particle_type)

        vel_stats = self._normalization_stats["velocity"]
        velocities = positions[..., 1:, :] - positions[..., :-1, :]
        recent_velocities = velocities[..., -self._velocity_context_size:, :]
        recent_velocities = (recent_velocities - vel_stats.mean) / vel_stats.std
        # Merge temporal and spatial dim
        if isinstance(recent_velocities, tf.Tensor):
            recent_velocities = tf.reshape(recent_velocities, [-1, self._velocity_context_size * self._dim])
        else:  # `tf.RaggedTensor`
            recent_velocities = tf.RaggedTensor.from_row_splits(
                tf.reshape(recent_velocities.flat_values, [-1, self._velocity_context_size * self._dim]),
                recent_velocities.row_splits
            )

        last_position = positions[..., -1 , :]
        boundary_proximity = LearnedSimulator._compute_clipped_boundary_proximity(
            last_position,
            tf.convert_to_tensor(self._boundaries, dtype=self.dtype),
            self._cutoff_radius
        )

        node_features = {
            "position": last_position,
            "recent_velocities": recent_velocities,
            "boundary_proximity": boundary_proximity,
            "embedded_particle_type": embedded_particle_type,
        }

        # --- Edge features ---
        if isinstance(positions, tf.Tensor):
            (num_particles, num_neighbors, neighbor_displacements, neighbor_distances,
            senders, receivers) = tf.py_function(
                func=LearnedSimulator._compute_neighborhoods,
                inp=(last_position, self._cutoff_radius, self._self_interaction),
                Tout=(tf.int64, tf.int64, self.dtype, self.dtype, tf.int64, tf.int64)
            )
            # Assert static shapes
            num_particles = tf.ensure_shape(num_particles, [])
            num_neighbors = tf.ensure_shape(num_neighbors, [])
            # Set dynamic shapes for tracing
            neighbor_displacements.set_shape([None, self._dim])
            neighbor_distances.set_shape([None, 1])
            senders.set_shape([None])
            receivers.set_shape([None])
        else:  # `tf.RaggedTensor`
            (num_particles, num_neighbors, displacement_values, distance_values,
            senders_values, receivers_values) = tf.py_function(
                func=LearnedSimulator._compute_neighborhoods_flat_batch,
                inp=(last_position, self._cutoff_radius, self._self_interaction),
                Tout=(tf.int64, tf.int64, self.dtype, self.dtype, tf.int64, tf.int64)
            )
            # Assert static shapes
            num_particles = tf.ensure_shape(num_particles, [None])
            num_neighbors = tf.ensure_shape(num_neighbors, [None])
            # Set dynamic shapes for tracing
            displacement_values.set_shape([None, self._dim])
            distance_values.set_shape([None, 1])
            senders_values.set_shape([None])
            receivers_values.set_shape([None])
            # Compose ragged tensors from flat values
            neighbor_displacements = tf.RaggedTensor.from_row_lengths(displacement_values, num_neighbors)
            neighbor_distances     = tf.RaggedTensor.from_row_lengths(distance_values, num_neighbors)
            senders                = tf.RaggedTensor.from_row_lengths(senders_values, num_neighbors)
            receivers              = tf.RaggedTensor.from_row_lengths(receivers_values, num_neighbors)

        edge_features = {
            "neighbor_displacements": neighbor_displacements,
            "neighbor_distances": neighbor_distances,
        }

        # --- Graph tensor ---
        graph = tfgnn.GraphTensor.from_pieces(
            context=tfgnn.Context.from_fields(features=global_features),
            node_sets={
                "particles": tfgnn.NodeSet.from_fields(
                    features=node_features,
                    sizes=tf.expand_dims(num_particles, axis=-1),
                )
            },
            edge_sets={
                "neighbors": tfgnn.EdgeSet.from_fields(
                    features=edge_features,
                    sizes=tf.expand_dims(num_neighbors, axis=-1),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("particles", senders),
                        target=("particles", receivers)
                    )
                )
            }
        )

        return graph.merge_batch_to_components()        

    @tf.function
    def _integrate_acceleration(
        self,
        positions: tf.Tensor | tf.RaggedTensor,
        norm_acc:  tf.Tensor | tf.RaggedTensor
    ) -> tf.Tensor | tf.RaggedTensor:
        """Compute next position by semi-implicit forward Euler."""
        acc_stats = self._normalization_stats["acceleration"]
        acc = norm_acc * acc_stats.std + acc_stats.mean

        last_pos = positions[..., -1, :]
        last_vel = last_pos - positions[..., -2, :]  # * dt = 1
        next_vel = last_vel + acc                    # * dt = 1
        next_pos = last_pos + next_vel               # * dt = 1

        return next_pos

    @tf.function
    def _differentiate_position(
        self,
        positions:  tf.Tensor | tf.RaggedTensor,
        target_pos: tf.Tensor | tf.RaggedTensor
    ) -> tf.Tensor | tf.RaggedTensor:
        """Compute normalized acceleration by inverting the forward step."""
        prev_pos = positions[..., -1, :]
        prev_vel = prev_pos - positions[..., -2, :]  # / dt = 1
        next_vel = target_pos - prev_pos             # / dt = 1
        acc = next_vel - prev_vel                    # / dt = 1

        acc_stats = self._normalization_stats["acceleration"]
        norm_acc = (acc - acc_stats.mean) / acc_stats.std

        return norm_acc
    
    @staticmethod
    def _compute_neighborhoods(
        positions:        tf.Tensor,
        cutoff_radius:    float,
        self_interaction: bool=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        tf_dtype = positions.dtype
        @tf.numpy_function(Tout=(tf.int64, tf.int64, tf_dtype, tf_dtype, tf.int64, tf.int64))
        def _compute_neighborhoods_np(
            positions_np: np.ndarray
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            tree = KDTree(positions_np, metric="euclidean")
            receivers_list = tree.query_radius(positions_np, r=cutoff_radius)
            receivers = np.concatenate(receivers_list, axis=0)

            num_particles = len(positions)
            senders = np.repeat(np.arange(num_particles), [len(recvs) for recvs in receivers_list])

            if not self_interaction:
                mask = senders != receivers
                senders = senders[mask]
                receivers = receivers[mask]

            num_neighbors = len(senders)

            displacements = (positions_np[receivers] - positions_np[senders]) / cutoff_radius
            distances = np.linalg.norm(displacements, axis=1, keepdims=True)

            return (
                tf.constant(num_particles, dtype=tf.int64),
                tf.constant(num_neighbors, dtype=tf.int64),
                tf.constant(displacements, dtype=tf_dtype),
                tf.constant(distances, dtype=tf_dtype),
                tf.constant(senders, dtype=tf.int64),
                tf.constant(receivers, dtype=tf.int64),
            )
    
        return _compute_neighborhoods_np(np.asarray(positions))
    
    @staticmethod
    def _compute_neighborhoods_flat_batch(
        positions:        tf.RaggedTensor,
        cutoff_radius:    float,
        self_interaction: bool=False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        num_particles = []
        num_neighbors = []
        displacements = []
        distances     = []
        senders       = []
        receivers     = []

        for pos_b in positions:
            (npart_b, nnbrs_b,
            displs_b, dists_b,
            sends_b, recvs_b) = LearnedSimulator._compute_neighborhoods(
                pos_b, cutoff_radius, self_interaction)
            num_particles.append(npart_b)
            num_neighbors.append(nnbrs_b)
            displacements.append(displs_b)
            distances.append(dists_b)
            senders.append(sends_b)
            receivers.append(recvs_b)

        return (
                tf.stack(num_particles),
                tf.stack(num_neighbors),
                tf.concat(displacements, axis=0),
                tf.concat(distances, axis=0),
                tf.concat(senders, axis=0),
                tf.concat(receivers, axis=0)
        )

    @staticmethod
    @tf.function
    def _compute_clipped_boundary_proximity(
        positions:     tf.Tensor | tf.RaggedTensor,
        boundaries:    tf.Tensor,
        cutoff_radius: float
    ) -> tf.Tensor | tf.RaggedTensor:
        lower_dist = positions - boundaries[:, 0]
        upper_dist = boundaries[:, 1] - positions
        dist = tf.concat([lower_dist, upper_dist], axis=-1)

        boundary_proximity = tf.clip_by_value(dist / cutoff_radius, -1., 1.)

        return boundary_proximity
