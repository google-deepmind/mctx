# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
# ============================================================================

"""Parity tests for opt-in bounded-scan tree traversal."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import search


class BoundedLoopTest(parameterized.TestCase):

  def test_bounded_scan_terminates_without_extra_body_calls(self):
    with mock.patch.dict(os.environ, {
        "MCTX_USE_BOUNDED_SCAN": "true",
        "MCTX_BOUNDED_SCAN_UNROLL": "2",
    }):
      result = jax.jit(lambda x: search._tree_traversal_loop(
          lambda v: v < 5, lambda v: v + 1, x, max_iterations=12))(jnp.int32(0))
    self.assertEqual(int(result), 5)

  def test_bounded_scan_preserves_unentered_loop(self):
    with mock.patch.dict(os.environ, {"MCTX_USE_BOUNDED_SCAN": "true"}):
      result = search._tree_traversal_loop(
          lambda v: v < 5, lambda v: v + 1, jnp.int32(9),
          max_iterations=10)
    self.assertEqual(int(result), 9)

  def test_invalid_scan_unroll_is_rejected(self):
    with mock.patch.dict(os.environ, {
        "MCTX_USE_BOUNDED_SCAN": "true",
        "MCTX_BOUNDED_SCAN_UNROLL": "0",
    }):
      with self.assertRaisesRegex(ValueError, "at least 1"):
        search._tree_traversal_loop(
            lambda v: v < 5, lambda v: v + 1, jnp.int32(0), 8)

  @parameterized.named_parameters(
      ("depth_one", 1),
      ("depth_four", 4),
      ("unbounded_depth", None),
  )
  def test_muzero_search_matches_unbounded_while_loop(self, max_depth):
    prior_logits = jnp.array([[2., 0., -1.], [0., 3., -2.]])
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=jnp.zeros((2,)),
        embedding=(),
    )
    rewards = jnp.array([[0., 2., -1.], [1., 0., -2.]])

    def recurrent_fn(params, key, action, embedding):
      del params, key, embedding
      reward = rewards[jnp.arange(action.shape[0]), action]
      return mctx.RecurrentFnOutput(
          reward=reward,
          discount=jnp.full_like(reward, .9),
          prior_logits=prior_logits,
          value=jnp.zeros_like(reward),
      ), ()

    def run(scan_enabled):
      with mock.patch.dict(os.environ, {
          "MCTX_USE_BOUNDED_SCAN": "true" if scan_enabled else "false",
          "MCTX_BOUNDED_SCAN_UNROLL": "2",
      }):
        # Separate jitted functions ensure JAX traces the requested branch.
        policy = jax.jit(lambda key: mctx.muzero_policy(
            params=(), rng_key=key, root=root, recurrent_fn=recurrent_fn,
            num_simulations=7, max_depth=max_depth, dirichlet_fraction=0.0,
        ))
        return policy(jax.random.PRNGKey(42))

    expected = run(False)
    actual = run(True)
    self.assertTrue(jnp.array_equal(expected.action, actual.action))
    self.assertTrue(jnp.array_equal(
        expected.action_weights, actual.action_weights))
    for name in ("node_visits", "node_values", "children_index",
                 "children_visits", "children_values"):
      self.assertTrue(jnp.array_equal(
          getattr(expected.search_tree, name),
          getattr(actual.search_tree, name)), name)


if __name__ == "__main__":
  absltest.main()
