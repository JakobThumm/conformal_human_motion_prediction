from jax import flatten_util
import jax.numpy as jnp


def compute_num_params(params):
    vector_params = flatten_util.ravel_pytree(params)[0]
    return vector_params.shape[0]


def compute_norm_params(params):
    vector_params = flatten_util.ravel_pytree(params)[0]
    return jnp.linalg.norm(vector_params).item()
