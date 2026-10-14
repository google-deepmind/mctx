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
"""Tests for `search.py`."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import search
import numpy as np


@jax.vmap
def _vmapped_backward(tree, leaf_index):
  """The unbatched backward of earlier versions, vmapped: the reference."""

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != search.Tree.ROOT_INDEX

  def body_fun(loop_state):
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1
    tree = tree.replace(
        node_values=search.update(tree.node_values, parent_value, parent),
        node_visits=search.update(tree.node_visits, count + 1, parent),
        children_values=search.update(
            tree.children_values, children_values, parent, action),
        children_visits=search.update(
            tree.children_visits, children_counts, parent, action))
    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)
  return tree


def _hashed_recurrent_fn(params, rng_key, action, embedding):
  """A deterministic environment whose outputs depend on the whole path."""
  del params, rng_key
  embedding = (embedding * 31 + action + 7) % 10007
  h = embedding.astype(jnp.float32)
  num_actions = 5
  return mctx.RecurrentFnOutput(
      reward=jnp.sin(h),
      discount=jnp.where(embedding % 11 == 0, 0.0, -0.9 + 0.1 * jnp.cos(h)),
      prior_logits=jnp.cos(h[:, None] * jnp.arange(1, num_actions + 1)),
      value=jnp.sin(2.0 * h)), embedding


def _run(policy_fn, batch_size, num_simulations, max_depth, seed):
  rng = np.random.default_rng(seed)
  num_actions = 5
  root = mctx.RootFnOutput(
      prior_logits=jnp.asarray(
          rng.normal(size=(batch_size, num_actions)), jnp.float32),
      value=jnp.asarray(rng.normal(size=batch_size), jnp.float32),
      embedding=jnp.asarray(rng.integers(0, 10007, batch_size), jnp.int32))
  invalid_actions = jnp.asarray(
      rng.random((batch_size, num_actions)) < 0.2, jnp.float32)
  invalid_actions = invalid_actions.at[:, 0].set(0.0)

  @jax.jit
  def run(root, invalid_actions):
    return policy_fn(
        params=(), rng_key=jax.random.PRNGKey(seed), root=root,
        recurrent_fn=_hashed_recurrent_fn, num_simulations=num_simulations,
        invalid_actions=invalid_actions, max_depth=max_depth)

  return run(root, invalid_actions)


class SearchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("muzero", mctx.muzero_policy, None),
      ("muzero_max_depth", mctx.muzero_policy, 3),
      ("gumbel_muzero", mctx.gumbel_muzero_policy, None),
      ("gumbel_muzero_max_depth", mctx.gumbel_muzero_policy, 4),
  )
  def test_batched_backward_matches_vmapped_backward(self, policy_fn,
                                                      max_depth):
    """The search builds the same tree as with the earlier vmapped backward."""
    out = _run(policy_fn, 16, 48, max_depth, seed=1)
    with mock.patch.object(search, "backward", _vmapped_backward):
      expected = _run(policy_fn, 16, 48, max_depth, seed=1)
    jax.tree.map(np.testing.assert_allclose, out, expected)

  def test_backward_on_rows_at_different_depths(self):
    """Rows whose leaves are at different depths, the root included."""
    tree = _run(mctx.muzero_policy, 8, 32, None, seed=2).search_tree
    leaves = jnp.asarray([0, 1, 5, 31, 32, 17, 0, 9], jnp.int32)
    got = search.backward(tree, leaves)
    expected = _vmapped_backward(tree, leaves)
    jax.tree.map(np.testing.assert_allclose, got, expected)


if __name__ == "__main__":
  absltest.main()
