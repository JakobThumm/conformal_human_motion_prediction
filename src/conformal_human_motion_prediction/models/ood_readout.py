"""
Readout head used to reduce the motion model's output for sketched-Lanczos OOD scoring.

Sketched-Lanczos OOD scoring builds a GGN/Laplace approximation whose cost scales with the
scored model's *output* dimension, so it cannot run on the motion model's full
10 x 13 x 3 = 390-dim future motion. It runs on a reduction of it instead.

The default reduction is a fixed random orthonormal projection to `OOD_PROJECTION_DIM`
directions:

    z = y_rel @ Q,    Q in R^{390 x d} with orthonormal columns

so every output coordinate enters the readout with non-zero weight and, in expectation, each
contributes equally -- as opposed to the legacy head, which selects 9 coordinates and discards
the other 381 by construction. `Q` is drawn once from `OOD_PROJECTION_SEED` at build time and
stored in the model pickle, so scoring is reproducible.

Orthonormal columns matter: they make this a genuine projection rather than an arbitrarily
scaled map, so no direction is double-counted and the `d` outputs share a common scale.

The head's parameters are stop-gradient-ed (`FrozenMotionRandomProjection`) so they stay out of
the GGN. This is not cosmetic: in reduced-output mode only ~412k of the model's 11.8M parameters
have a non-zero Jacobian at all (the covariance head sits behind `jax.lax.stop_gradient`), so an
un-frozen head sitting directly on the output would contribute a large share of the effective
Jacobian directions and the score would start measuring the head instead of the backbone.
"""

import jax
from flax import linen as nn

from conformal_human_motion_prediction.motion_prediction.h36m_settings import OOD_PROJECTION_DIM


class MotionRandomProjection(nn.Module):
    """Project flattened offset-relative future motion onto `projection_dim` fixed directions.

    Attributes:
        projection_dim: Number of output directions (the OOD model's output dimension).
    """

    projection_dim: int = OOD_PROJECTION_DIM

    @nn.compact
    def __call__(self, y_rel):
        """
        Args:
            y_rel: Offset-relative future motion, [batch_size, seq_len_output, n_joints*3] or
                [batch_size, output_dim]. Units are irrelevant to AUROC (the OOD score is
                homogeneous of degree 2 in the Jacobian) but fix the score's scale.

        Returns:
            Projected output [batch_size, projection_dim].
        """
        flat = y_rel.reshape(y_rel.shape[0], -1)
        kernel = self.param(
            "kernel", nn.initializers.orthogonal(), (flat.shape[-1], self.projection_dim)
        )
        return flat @ kernel


# Same module with its parameters stop-gradient-ed on the way in, so the projection contributes
# nothing to the GGN or to the per-datapoint Jacobian. `init=True, mutable=True` are required,
# otherwise `init` raises ScopeCollectionNotFound.
FrozenMotionRandomProjection = nn.map_variables(
    MotionRandomProjection,
    "params",
    trans_in_fn=lambda tree: jax.tree_util.tree_map(jax.lax.stop_gradient, tree),
    init=True,
    mutable=True,
)
