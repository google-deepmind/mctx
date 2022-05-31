"""A unit that verifies that multiaction-muzero-policy gives the expected sequence of actions."""
import functools
import json

import chex
from absl import logging
from absl.testing import parameterized
import jax
import mctx
import numpy as np

from mctx._src.tests.tree_test import _prepare_root, _prepare_recurrent_fn


class MuzeroForActionSequenceTest(parameterized.TestCase):
  def test_tree(self, draw_graph=False):
    with open("test_data/muzero_for_action_sequence_qtransform_tree.json", "rb") as fd:
      tree = json.load(fd)
    assert tree["algorithm"] == "muzero_for_action_sequence"

    env_config = tree["env_config"]
    root = tree["tree"]
    num_actions = len(root["child_stats"])
    num_simulations = root["visit"] - 1
    qtransform = functools.partial(
        getattr(mctx, tree["algorithm_config"].pop("qtransform")),
        **tree["algorithm_config"].pop("qtransform_kwargs", {}))

    batch_size = 3

    def run_policy():
      return mctx.muzero_policy_for_action_sequence(
        params=(),
        rng_key=jax.random.PRNGKey(1),
        root=_prepare_root(batch_size=batch_size, num_actions=num_actions),
        recurrent_fn=_prepare_recurrent_fn(num_actions, **env_config),
        num_simulations=num_simulations,
        qtransform=qtransform,
        **tree["algorithm_config"],
        num_actions_to_generate=3,
      )

    policy_output = jax.jit(run_policy)()
    logging.info("Done search.")

    if draw_graph:
      from examples.visualization_demo import convert_tree_to_graph
      graph = convert_tree_to_graph(policy_output.search_tree)
      graph.draw("/tmp/multiaction-muzero-dotgraph.png", prog="dot")

    expected_actions = np.array([[14, 14, 3],
                                 [0, 9, 14],
                                 [1, 7, 9]])
    chex.assert_trees_all_close(policy_output.action, expected_actions)
