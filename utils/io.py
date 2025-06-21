import argparse
import glob
import json
import os

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf

from .data import StatsPair


_FEATURE_DESCRIPTION = {
    "position": tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT["step_context"] = tf.io.VarLenFeature(tf.string)

_FEATURE_DTYPES = {
    "position": np.float32,
    "step_context": np.float32
}

_CONTEXT_FEATURES = {
    "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "particle_type": tf.io.VarLenFeature(tf.string)
}

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the learned simulator of Sanchez-Gonzalez et al.")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "eval_rollout"],
        default="train",
        help="One-step training or evaluation, or rollout evaluation.")
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=["train", "valid", "test"],
        default="test",
        help="Split to use when running evaluation.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The dataset directory.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20_000_000,
        help="Number of steps of training.")
    parser.add_argument(
        "--noise_std",
        type=float,
        default=6.7e-4,
        help="Standard deviation of the Gaussian noise.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path for saving checkpoints of the model.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path for saving outputs of the model.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )

    return parser.parse_args()


def load_dataset(
    data_path:           str,
    split:               str="train",  # or "valid" or "test"
    mode:                str="one_step",  # or "one_step_train" or "rollout"
    shuffle_buffer_size: int=10_000,  # @S-G, p. 13
    window_length:       int=7,  # @S-G, p. 4
    materialize_cache:   bool=False
) -> Tuple[tf.data.Dataset, Optional[int]]:
    # Load metadata.json
    metadata = load_metadata(data_path)

    # Create a tf.data.Dataset from the specified TFRecord
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f"{split}.tfrecord")])
    ds = ds.map(lambda ex: parse_serialized_simulation_example(ex, metadata))
    if mode.startswith("one_step"):
        # Split trajectories into windows: seed positions for velocities + a target position
        ds = ds.flat_map(lambda ctx, fts: split_trajectory(ctx, fts, window_length))
        # Split further into model inputs and target positions
        ds = ds.map(prepare_step_inputs)
        ds = ds.cache()
        ds_size = sum(1 for _ in ds) if materialize_cache else None
        # If in train mode, optionally repeat dataset and shuffle
        if mode == "one_step_train":
            ds = ds.repeat()
            ds = ds.shuffle(ds_size if materialize_cache else shuffle_buffer_size)
            options = tf.data.Options()
            options.deterministic = False
            ds = ds.with_options(options)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    elif mode == "rollout":
        ds = ds.map(lambda ctx, fts: prepare_rollout_inputs(ctx, fts), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds_size = sum(1 for _ in ds) if materialize_cache else None
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        raise ValueError(f"mode: {mode} not recognized")

    return ds, ds_size

def parse_serialized_simulation_example(
    serialized: bytes,
    metadata:   Mapping[str, Any]
) -> Tuple[Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]]:
    
    def _decode_values(values: tf.Tensor, dtype: np.dtype) -> tf.Tensor:
        tf_dtype = tf.as_dtype(dtype)
        @tf.py_function(Tout=tf_dtype)
        def _decode(values):
            if len(values) == 1:
                out = np.frombuffer(values[0].numpy(), dtype=dtype)
            else:
                out = np.array([np.frombuffer(elem.numpy(), dtype=dtype) for elem in values])
            return tf.convert_to_tensor(out, tf_dtype)
        return _decode(values)
    
    context, features = tf.io.parse_single_sequence_example(
        serialized,  
        context_features=_CONTEXT_FEATURES,
        sequence_features=_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT if "context_mean" in metadata else
                          _FEATURE_DESCRIPTION
    )

    # Decode sequence features
    for feature_key, item in features.items():
        features[feature_key] = _decode_values(item.values, _FEATURE_DTYPES[feature_key])

    # Reshape to account for extra frame at the beginning
    position_shape = [metadata["sequence_length"] + 1, -1, metadata["dim"]]
    features["position"] = tf.reshape(features["position"], position_shape)
    if "context_mean" in metadata:
        step_context_shape = [metadata["sequence_length"] + 1, len(metadata["context_mean"])]
        features["step_context"] = tf.reshape(features["step_context"], step_context_shape)
        
    # Decode particle type explicitly
    context["particle_type"] = _decode_values(context["particle_type"].values, np.int64)
    context["particle_type"] = tf.reshape(context["particle_type"], [-1])  # Flatten

    return context, features

def split_trajectory(
    context:       Mapping[str, tf.Tensor],
    features:      Mapping[str, tf.Tensor],
    window_length: int
) -> tf.data.Dataset:
    num_timesteps = tf.shape(features["position"])[0]
    num_splits = num_timesteps - window_length + 1

    def _slice_feature(feature: tf.Tensor) -> tf.Tensor:
        def _slice_fn(idx):
            return feature[idx : idx + window_length]

        return tf.map_fn(
            _slice_fn,
            tf.range(num_splits),
            fn_output_signature=tf.TensorSpec(
                shape=tf.TensorShape([window_length]).concatenate(feature.shape[1:]),
                dtype=feature.dtype
            )
        )

    positions = _slice_feature(features["position"])
    particle_type = tf.tile(
        tf.expand_dims(context["particle_type"], axis=0),
        multiples=[num_splits, 1]
    )
    inputs_dict = {
        "positions": positions,
        "particle_type": particle_type
    }
    if "step_context" in features:
        global_context = _slice_feature(features["step_context"])
        inputs_dict["global_context"] = global_context

    return tf.data.Dataset.from_tensor_slices(inputs_dict)  # for flat_map
    
def prepare_step_inputs(
    inputs_dict: Mapping[str, tf.Tensor]
) -> Mapping[str, tf.Tensor]:
    
    positions = tf.transpose(inputs_dict["positions"], perm=[1, 0, 2])  # [T, N, D] -> [N, T, D]
    inputs_dict["positions"] = positions

    if "step_context" in inputs_dict:
        inputs_dict["global_context"] = inputs_dict["global_context"][-2]  # Select penultimate context for step
        inputs_dict["global_context"] = inputs_dict["global_context"][tf.newaxis]  # Add context feature dimension

    return inputs_dict  # Dev NOTE: No split into target_pos, moved to model call

def prepare_rollout_inputs(
    context:  Mapping[str, tf.Tensor],
    features: Mapping[str, tf.Tensor]
) -> Mapping[str, tf.Tensor]:
    
    positions = tf.transpose(features["position"], perm=[1, 0, 2])  # [T, N, D] -> [N, T, D]
    inputs_dict = {
        "positions": positions,
        **context
    }
    if "step_context" in features:
        inputs_dict["global_contexts"] = features["step_context"]

    return inputs_dict

def load_metadata(data_path: str) -> Mapping[str, Any]:
    with open(os.path.join(data_path, "metadata.json"), "rt") as file:
        metadata = json.loads(file.read())
    return metadata

def get_normalization_stats(
    metadata:      Mapping[str, Any],
    acc_noise_std: float,
    vel_noise_std: float,
    tf_dtype:      tf.DType = tf.float32
) -> Mapping[str, StatsPair]:
    def _to_tensor(val: List[float]) -> tf.Tensor:
        return tf.convert_to_tensor(val, dtype=tf_dtype)

    def _combine_std(base_std: tf.Tensor, noise_std: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(base_std * base_std + noise_std * noise_std)

    normalization_stats = {
        "acceleration": StatsPair(
            mean=_to_tensor(metadata["acc_mean"]),
            std=_combine_std(_to_tensor(metadata["acc_std"]), _to_tensor(acc_noise_std))
        ),
        "velocity": StatsPair(
            mean=_to_tensor(metadata["vel_mean"]),
            std=_combine_std(_to_tensor(metadata["vel_std"]), _to_tensor(vel_noise_std))
        )
    }

    if "context_mean" in metadata:
        normalization_stats["context"] = StatsPair(
            mean=_to_tensor(metadata["context_mean"]),
            std=_to_tensor(metadata["context_std"])
        )

    return normalization_stats

def get_latest_checkpoint(model_path: str) -> str:
    checkpoints = sorted(glob.glob(os.path.join(model_path, "*.weights.h5")))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in model path.")
    return checkpoints[-1]
