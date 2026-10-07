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
"""Finite search averages for low-precision values and integer visit counts."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import search
import numpy as np


def one_edge_tree(dtype, old_value, reward, visits):
  root = mctx.RootFnOutput(
      prior_logits=jnp.zeros((1, 2)),
      value=jnp.array([old_value], dtype),
      embedding=jnp.zeros((1, 1)),
  )
  tree = search.instantiate_tree_from_root(
      root, 1, root_invalid_actions=jnp.zeros((1, 2)), extra_data=None
  )
  return tree.replace(
      parents=tree.parents.at[0, 1].set(0),
      action_from_parent=tree.action_from_parent.at[0, 1].set(0),
      children_index=tree.children_index.at[0, 0, 0].set(1),
      children_rewards=tree.children_rewards.at[0, 0, 0].set(reward),
      node_visits=tree.node_visits.at[0, 0]
      .set(visits)
      .at[0, 1]
      .set(visits - 1),
      children_visits=tree.children_visits.at[0, 0, 0].set(visits - 1),
  )


class BackwardPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32],
      use_jit=[False, True],
      case=[
          (60000.0, 60000.0, 1),
          (60000.0, -60000.0, 3),
          (100.0, 200.0, 70000),
      ],
  )
  def test_backup_matches_wide_reference_without_changing_storage(
      self, dtype, use_jit, case
  ):
    old, reward, count = case
    tree = one_edge_tree(dtype, old, reward, count)
    stored_old = float(tree.node_values[0, 0])
    stored_reward = float(tree.children_rewards[0, 0, 0])
    expected = (stored_old * count + stored_reward) / (count + 1)
    operation = jax.jit(search.backward) if use_jit else search.backward
    result = operation(tree, jnp.array([1]))
    self.assertEqual(result.node_values.dtype, dtype)
    self.assertTrue(bool(jnp.isfinite(result.node_values).all()))
    tolerance = (
        0.008
        if dtype == jnp.bfloat16
        else 0.001
        if dtype == jnp.float16
        else 1e-6
    )
    np.testing.assert_allclose(
        result.node_values[0, 0], expected, rtol=tolerance
    )
    self.assertEqual(int(result.node_visits[0, 0]), count + 1)
    self.assertEqual(int(result.children_visits[0, 0, 0]), count)
    np.testing.assert_array_equal(result.parents, tree.parents)
    np.testing.assert_array_equal(
        result.children_rewards, tree.children_rewards
    )

  @parameterized.product(use_jit=[False, True], initial=[0.0, 60000.0])
  def test_public_muzero_preserves_finite_high_reward_averages(
      self, use_jit, initial
  ):
    dtype = jnp.float16
    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros((1, 2)),
        value=jnp.array([initial], dtype),
        embedding=jnp.zeros((1, 1)),
    )

    def recurrent(params, key, action, embedding):
      del params, key, action
      return (
          mctx.RecurrentFnOutput(
              reward=jnp.full(1, 60000.0, dtype),
              discount=jnp.zeros(1, dtype),
              prior_logits=jnp.zeros((1, 2)),
              value=jnp.zeros(1, dtype),
          ),
          embedding,
      )

    def policy():
      return mctx.muzero_policy(
          None,
          jax.random.PRNGKey(0),
          root,
          recurrent,
          num_simulations=4,
          dirichlet_fraction=0.0,
          max_depth=1,
          qtransform=lambda tree, node: jnp.zeros_like(
              tree.children_prior_logits[node]
          ),
      )

    result = (jax.jit(policy) if use_jit else policy)()
    self.assertTrue(bool(jnp.isfinite(result.search_tree.node_values).all()))
    np.testing.assert_allclose(
        result.search_tree.node_values[0, 0],
        (initial + 4 * 60000.0) / 5,
        rtol=0.002,
    )
    np.testing.assert_allclose(jnp.sum(result.action_weights, axis=-1), 1.0)


if __name__ == '__main__':
  absltest.main()
