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
"""Range normalization should remain finite with low-precision tree values."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import qtransforms
from mctx._src import search
import numpy as np


def tree_for(values, visits=None):
  root = mctx.RootFnOutput(
      prior_logits=jnp.zeros((1, len(values)), jnp.float32),
      value=values[:1],
      embedding=jnp.zeros((1, 1)),
  )
  tree = search.instantiate_tree_from_root(
      root,
      num_simulations=1,
      root_invalid_actions=jnp.zeros((1, len(values))),
      extra_data=None,
  )
  counts = (
      jnp.ones_like(values, dtype=jnp.int32)
      if visits is None
      else jnp.asarray(visits, jnp.int32)
  )
  tree = tree.replace(
      children_rewards=tree.children_rewards.at[0, 0].set(values),
      children_visits=tree.children_visits.at[0, 0].set(counts),
  )
  return jax.tree.map(lambda x: x[0], tree)


class QtransformPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      values=[[1.0, 1.0, 1.0], [-60000.0, 0.0, 60000.0], [1.0, 3.0, 6.0]],
  )
  def test_automatic_normalization_matches_float64_reference(
      self, dtype, values
  ):
    qvalues = jnp.asarray(values, dtype=dtype)
    reference = np.asarray(qvalues, dtype=np.float64)
    reference = (reference - reference.min()) / max(
        reference.max() - reference.min(), 1e-8
    )
    tree = tree_for(qvalues)
    for fn in (
        lambda x: qtransforms._rescale_qvalues(x, 1e-8),
        lambda x: qtransforms.qtransform_by_parent_and_siblings(
            tree_for(x), jnp.array(0)
        ),
        lambda x: qtransforms.qtransform_completed_by_mix_value(
            tree_for(x),
            jnp.array(0),
            use_mixed_value=False,
            maxvisit_init=1.0,
            value_scale=1.0,
        )
        / 2,
    ):
      for calculate in (fn, jax.jit(fn)):
        actual = calculate(qvalues)
        self.assertEqual(actual.dtype, dtype)
        self.assertTrue(bool(jnp.isfinite(actual).all()))
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=0.01 if dtype == jnp.bfloat16 else 0.001,
            atol=1e-6,
        )

  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_given_bounds_do_not_overflow_for_representable_qvalues(self, dtype):
    values = jnp.array([-60000.0, 0.0, 60000.0], dtype=dtype)
    tree = tree_for(values)
    fn = lambda t: qtransforms.qtransform_by_min_max(
        t, jnp.array(0), min_value=-60000.0, max_value=60000.0
    )
    for calculate in (fn, jax.jit(fn)):
      actual = calculate(tree)
      self.assertEqual(actual.dtype, dtype)
      np.testing.assert_allclose(actual, [0.0, 0.5, 1.0], atol=0, rtol=0)

  def test_typed_half_precision_bounds_use_safe_range_arithmetic(self):
    values = jnp.array([-60000.0, 0.0, 60000.0], jnp.float16)
    tree = tree_for(values)
    actual = qtransforms.qtransform_by_min_max(
        tree, jnp.array(0), min_value=values[0], max_value=values[-1]
    )
    self.assertEqual(actual.dtype, jnp.float16)
    np.testing.assert_array_equal(actual, [0.0, 0.5, 1.0])

  def test_unvisited_actions_retain_zero_scores(self):
    values = jnp.array([2.0, 100.0, 4.0], jnp.float16)
    tree = tree_for(values, [1, 0, 1])
    actual = qtransforms.qtransform_by_parent_and_siblings(tree, jnp.array(0))
    np.testing.assert_array_equal(actual, [0.0, 0.0, 1.0])

  def test_explicit_wider_bounds_and_epsilon_keep_their_precision(self):
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
      values = jnp.array([1.0, 3.0, 6.0], jnp.float32)
      tree = tree_for(values)
      epsilon = jnp.array(1e-8, jnp.float64)
      scaled = qtransforms._rescale_qvalues(values, epsilon)
      parent = qtransforms.qtransform_by_parent_and_siblings(
          tree, jnp.array(0), epsilon=epsilon
      )
      bounded = qtransforms.qtransform_by_min_max(
          tree,
          jnp.array(0),
          min_value=jnp.array(1.0, jnp.float64),
          max_value=jnp.array(6.0, jnp.float64),
      )
      for actual in (scaled, parent, bounded):
        self.assertEqual(actual.dtype, jnp.float64)
        np.testing.assert_allclose(actual, [0.0, 0.4, 1.0], rtol=1e-14, atol=0)
    finally:
      jax.config.update('jax_enable_x64', previous)

  def test_half_precision_muzero_search_matches_float32_on_small_tree(self):
    def policy(dtype):
      root = mctx.RootFnOutput(
          prior_logits=jnp.array([[-8.0, 0.0, 8.0]]),
          value=jnp.zeros(1, dtype),
          embedding=jnp.zeros((1, 1)),
      )

      def recurrent(params, rng_key, action, embedding):
        del params, rng_key
        return (
            mctx.RecurrentFnOutput(
                reward=action.astype(dtype),
                discount=jnp.zeros(1, dtype),
                prior_logits=jnp.zeros((1, 3)),
                value=jnp.zeros(1, dtype),
            ),
            embedding,
        )

      return mctx.muzero_policy(
          None,
          jax.random.PRNGKey(0),
          root,
          recurrent,
          num_simulations=16,
          max_depth=1,
          dirichlet_fraction=0.0,
      )

    reference = policy(jnp.float32)
    actual = jax.jit(lambda: policy(jnp.float16))()
    np.testing.assert_allclose(
        actual.action_weights, reference.action_weights, rtol=0, atol=0
    )
    np.testing.assert_array_equal(actual.action, reference.action)


if __name__ == '__main__':
  absltest.main()
