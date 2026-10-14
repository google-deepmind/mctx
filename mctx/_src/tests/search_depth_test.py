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
"""Tests of the depth limit of `search.search`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from mctx._src import base
from mctx._src import search
import numpy as np


def _root_action_selection(rng_key, tree, node_index):
  del rng_key, tree, node_index
  return jnp.array(0, dtype=jnp.int32)


def _interior_action_selection(rng_key, tree, node_index, depth):
  del rng_key, tree, node_index, depth
  return jnp.array(0, dtype=jnp.int32)


class SearchDepthTest(parameterized.TestCase):

  @parameterized.parameters(jnp.float32, jnp.bfloat16)
  def test_max_depth_holds_beyond_256_plies(self, prior_dtype):
    # One action: every simulation walks the same path, one node deeper each
    # time, until `max_depth`; afterwards the deepest node is expanded again.
    num_simulations = 384
    max_depth = 300
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((1, 1), prior_dtype),
        value=jnp.zeros((1,), jnp.float32),
        embedding=jnp.zeros((1,), jnp.int32))

    def recurrent_fn(params, rng_key, action, embedding):
      del params, rng_key, action
      output = base.RecurrentFnOutput(
          reward=jnp.zeros_like(embedding, dtype=jnp.float32),
          discount=jnp.ones_like(embedding, dtype=jnp.float32),
          prior_logits=jnp.zeros((1, 1), prior_dtype),
          value=jnp.zeros_like(embedding, dtype=jnp.float32))
      return output, embedding + 1

    tree = jax.jit(lambda key: search.search(
        params=(), rng_key=key, root=root, recurrent_fn=recurrent_fn,
        root_action_selection_fn=_root_action_selection,
        interior_action_selection_fn=_interior_action_selection,
        num_simulations=num_simulations, max_depth=max_depth))(
            jax.random.PRNGKey(0))
    expanded = int((np.asarray(tree.node_visits[0]) > 0).sum()) - 1
    self.assertEqual(expanded, max_depth)
    self.assertEqual(int(np.asarray(tree.embeddings[0]).max()), max_depth)
    self.assertEqual(int(tree.node_visits[0, 0]), num_simulations + 1)


if __name__ == "__main__":
  absltest.main()
