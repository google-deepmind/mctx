# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mixed values retain contributions from tiny-prior visited actions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
import numpy as np


class MixedValuePrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      negative=(False, True),
      tiny_value=(False, True),
  )
  def test_zero_prior_preserves_visited_value(
      self, dtype, negative, tiny_value
  ):
    qvalue = (1e-4 if tiny_value else 0.25) * (-1 if negative else 1)
    qvalues = jnp.array([qvalue, 1.0], dtype=dtype)
    prior_probs = jnp.array([0.0, 1.0], dtype=dtype)
    visits = jnp.array([1, 0])
    raw_value = jnp.array(0.0, dtype=dtype)
    expected = float(qvalues[0]) / 2
    for fn in (
        qtransforms._compute_mixed_value,
        jax.jit(qtransforms._compute_mixed_value),
    ):
      actual = fn(raw_value, qvalues, visits, prior_probs)
      self.assertEqual(actual.dtype, dtype)
      np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=0)

  @parameterized.parameters(jnp.bfloat16, jnp.float32)
  def test_multiple_zero_prior_visits_match_normalized_reference(self, dtype):
    probs = jax.nn.softmax(jnp.array([-1000.0, 0.0, -1000.0], dtype))
    visits = jnp.array([1, 0, 2])
    qvalues = jnp.array([0.25, 3.0, 0.75], dtype)
    raw = jnp.array(-0.5, dtype)
    # Both visited priors are floored to the same positive value, so their
    # conditional weights are 1/2. Total visits set the mixing coefficient.
    expected = (-0.5 + 3 * (0.25 + 0.75) / 2) / 4
    for fn in (
        qtransforms._compute_mixed_value,
        jax.jit(qtransforms._compute_mixed_value),
    ):
      np.testing.assert_allclose(fn(raw, qvalues, visits, probs), expected)

  def test_completed_qvalues_use_the_visited_action_value(self):
    root = base.RootFnOutput(
        prior_logits=jnp.array([[-1000.0, 0.0]]),
        value=jnp.array([0.5]),
        embedding=jnp.zeros((1, 1)),
    )
    tree = search.instantiate_tree_from_root(
        root,
        num_simulations=1,
        root_invalid_actions=jnp.zeros((1, 2)),
        extra_data=None,
    )
    tree = tree.replace(
        children_visits=tree.children_visits.at[0, 0, 0].set(1),
        children_rewards=tree.children_rewards.at[0, 0, 0].set(0.25),
    )
    tree = jax.tree.map(lambda x: x[0], tree)

    def transform(t):
      return qtransforms.qtransform_completed_by_mix_value(
          t,
          jnp.int32(0),
          value_scale=1.0,
          maxvisit_init=0.0,
          rescale_values=False,
      )

    for fn in (transform, jax.jit(transform)):
      np.testing.assert_allclose(fn(tree), [0.25, 0.375])

  def test_no_visits_and_ordinary_priors(self):
    raw = jnp.array(-0.5)
    qvalues = jnp.array([0.25, 0.75, 10.0])
    probs = jnp.array([0.2, 0.8, 0.0])
    for fn in (
        qtransforms._compute_mixed_value,
        jax.jit(qtransforms._compute_mixed_value),
    ):
      with jax.debug_nans(True):
        np.testing.assert_array_equal(
            fn(raw, qvalues, jnp.zeros(3), probs), raw
        )
        value = fn(raw, qvalues, jnp.array([2, 1, 0]), probs)
      np.testing.assert_allclose(value, (-0.5 + 3 * 0.65) / 4, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
