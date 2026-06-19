import os
import pickle
import json
import dataclasses
from typing import Callable
import jax
import flax
from conformal_human_motion_prediction.models import RegressFlowFlax
from conformal_human_motion_prediction.models.dct_pose_transformer_pytorch_attn import DCTPoseTransformer
from conformal_human_motion_prediction.motion_prediction.h36m_settings import (
    N_JOINTS,
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
)


@dataclasses.dataclass
class Model:
    init: Callable  # init(x, params)
    apply_train: Callable  # apply(x, params)
    apply_test: Callable  # apply(x, params)
    has_batch_stats: bool
    has_dropout: bool
    has_attentionmask: bool


def wrap_model(model) -> Model:
    def init(key, x):
        params_dict = model.init(key, x, train=True)
        params_dict["batch_stats"] = None
        return params_dict

    def apply_train(params, x):
        return model.apply({"params": params}, x, train=True)

    def apply_test(params, x):
        return model.apply({"params": params}, x, train=False)

    return Model(
        init=init,
        apply_train=apply_train,
        apply_test=apply_test,
        has_batch_stats=False,
        has_dropout=False,
        has_attentionmask=False,
    )


def wrap_model_with_dropout(model) -> Model:
    def init(key, x):
        params_dict = model.init({"params": key}, x, deterministic=True)
        return params_dict

    def apply_train(params, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply(params, x, deterministic=False, rngs={"drop_path": key_dropout, "dropout": key_dropout2})

    def apply_test(params, x):
        return model.apply(params, x, deterministic=True)

    return Model(
        init=init,
        apply_train=apply_train,
        apply_test=apply_test,
        has_batch_stats=False,
        has_dropout=True,
        has_attentionmask=False,
    )


def wrap_model_with_batchstats(model) -> Model:
    def init(key, x):
        params_dict = model.init(key, x, train=True)
        return params_dict

    def apply_train(params, batch_stats, x):
        return model.apply({"params": params, "batch_stats": batch_stats}, x, train=True, mutable=["batch_stats"])

    def apply_test(params, batch_stats, x):
        return model.apply({"params": params, "batch_stats": batch_stats}, x, train=False, mutable=False)

    return Model(
        init=init,
        apply_train=apply_train,
        apply_test=apply_test,
        has_batch_stats=True,
        has_dropout=False,
        has_attentionmask=False,
    )


def wrap_model_with_batchstats_dropout(model) -> Model:
    def init(key, x):
        # key, drop = jax.random.split(key, 2)
        # params_dict = model.init({"params": key, "drop_path": drop}, x)
        params_dict = model.init({"params": key}, x, deterministic=True)
        return params_dict

    def apply_train(params, batch_stats, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            deterministic=False,
            # rngs={'drop_path': key_dropout},
            rngs={"drop_path": key_dropout, "dropout": key_dropout2},
            mutable=["batch_stats"],
        )

    def apply_test(params, batch_stats, x):
        return model.apply({"params": params, "batch_stats": batch_stats}, x, deterministic=True, mutable=False)

    return Model(
        init=init,
        apply_train=apply_train,
        apply_test=apply_test,
        has_batch_stats=True,
        has_dropout=True,
        has_attentionmask=False,
    )


def wrap_model_with_attentionmask(model) -> Model:
    def init(key, x):
        rng1, rng2, rng3 = jax.random.split(key, 3)
        params_dict = model.init({"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False)
        return params_dict

    def apply_train(params, attention_mask, relative_position_index, x, key_dropout):
        key_dropout, key_dropout2 = jax.random.split(key_dropout, 2)
        return model.apply(
            {"params": params, "attention_mask": attention_mask, "relative_position_index": relative_position_index},
            x,
            deterministic=False,
            rngs={"drop_path": key_dropout, "dropout": key_dropout2},
            mutable=["attention_mask", "relative_position_index"],
        )

    def apply_test(params, attention_mask, relative_position_index, x):
        return model.apply(
            {"params": params, "attention_mask": attention_mask, "relative_position_index": relative_position_index},
            x,
            deterministic=True,
            mutable=False,
        )

    return Model(
        init=init,
        apply_train=apply_train,
        apply_test=apply_test,
        has_batch_stats=False,
        has_dropout=True,
        has_attentionmask=True,
    )


def model_from_string(
    model_name: str,
    output_dim: int,
    activation_fun: str = "relu",
    mlp_num_layers: int = 1,
    mlp_hidden_dim: int = 20,
    architecture_str: str = "resnet50"
):
    act_fn = getattr(flax.linen, activation_fun)

    if model_name == "RegressFlow":
        num_joints = output_dim // 2
        model = RegressFlowFlax(
            num_joints=num_joints,
            fc_filters=[-1],
            architecture_str=architecture_str,
            accept_nchw=True,
            predict_aleatoric_uncertainty=False,
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "RegressFlowWithAleatoric":
        # Calculate number of joints from output_dim (output_dim = num_joints * 2)
        num_joints = output_dim // 2
        model = RegressFlowFlax(
            num_joints=num_joints,
            fc_filters=[-1],
            architecture_str=architecture_str,
            accept_nchw=True,
            predict_aleatoric_uncertainty=True,
        )
        wrapped_model = wrap_model_with_batchstats(model)
    elif model_name == "DCTPoseTransformer":
        model = DCTPoseTransformer(
            input_dim=(3 * N_JOINTS),  # 3D coordinates per joint
            seq_len=INPUT_HORIZON_LENGTH,
            seq_len_output=PREDICTION_HORIZON_LENGTH,
            reduced_size=False,
        )
        wrapped_model = wrap_model(model)
    elif model_name == "DCTPoseTransformerReducedOutput":
        model = DCTPoseTransformer(
            input_dim=(3 * N_JOINTS),  # 3D coordinates per joint
            seq_len=INPUT_HORIZON_LENGTH,
            seq_len_output=PREDICTION_HORIZON_LENGTH,
            reduced_size=True,
        )
        wrapped_model = wrap_model(model)
    else:
        raise ValueError(f"Model {model_name} is not implemented (yet)")

    return wrapped_model


def pretrained_model_from_string(
    model_name="LeNet", dataset_name="MNIST", run_name="example", seed=0, n_samples=None, save_path="../models"
):
    if n_samples is not None:
        dataset_name += f"_samples{n_samples}"
    base_path = f"{save_path}/{dataset_name}/{model_name}/seed_{seed}/"
    args_file_path = f"{base_path}/{run_name}_args.json"
    if not os.path.exists(args_file_path):
        # Fallback: pose checkpoints are consolidated under a single RegressFlow/ directory,
        # while model_name (e.g. RegressFlowResNet18_3Joints) is still used for the cache key.
        base_path = f"{save_path}/{dataset_name}/RegressFlow/seed_{seed}/"
        args_file_path = f"{base_path}/{run_name}_args.json"
    if not os.path.exists(args_file_path):
        base_path = save_path
        args_file_path = f"{base_path}/{run_name}_args.json"
        if not os.path.exists(args_file_path):
            raise FileNotFoundError(f"File {args_file_path} not found")

    args_dict = json.load(open(args_file_path, "r"))

    extra_args = {
        "activation_fun": args_dict.get("activation_fun", "relu"),
        "mlp_num_layers": args_dict.get("mlp_num_layers", 1),
        "mlp_hidden_dim": args_dict.get("mlp_hidden_dim", 64),
        "architecture_str": args_dict.get("architecture_str", "resnet50"),
    }

    model = model_from_string(args_dict["model"], args_dict["output_dim"], **extra_args)

    params_file_path = f"{base_path}/{run_name}_params.pickle"
    if not os.path.exists(params_file_path):
        params_file_path = f"{base_path}/{run_name}.pickle"
        if not os.path.exists(params_file_path):
            raise FileNotFoundError(f"File {params_file_path} not found")
    params_dict = pickle.load(open(params_file_path, "rb"))
    params_dict.pop("model")

    return model, params_dict, args_dict
