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
"""A demonstration of the policy improvement by planning with Gumbel."""

import functools
from typing import Tuple

from absl import app
from absl import flags
import chex
import jax
import jax.numpy as jnp
import mctx

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("num_actions", 82, "Number of actions.")
flags.DEFINE_integer("num_simulations", 4, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("num_runs", 1, "Number of runs on random data.")


@chex.dataclass(frozen=True)
class DemoOutput:
  prior_policy_value: chex.Array
  prior_policy_action_value: chex.Array
  selected_action_value: chex.Array
  action_weights_policy_value: chex.Array


def _run_demo(rng_key: chex.PRNGKey) -> Tuple[chex.PRNGKey, DemoOutput]:
  """Runs a search algorithm on random data."""
  batch_size = FLAGS.batch_size
  rng_key, logits_rng, q_rng, search_rng = jax.random.split(rng_key, 4)
  # We will demonstrate the algorithm on random prior_logits.
  # Normally, the prior_logits would be produced by a policy network.
  prior_logits = jax.random.normal(
      logits_rng, shape=[batch_size, FLAGS.num_actions])
  # Defining a bandit with random Q-values. Only the Q-values of the visited
  # actions will be revealed to the search algorithm.
  qvalues = jax.random.uniform(q_rng, shape=prior_logits.shape)
  # If we know the value under the prior policy, we can use the value to
  # complete the missing Q-values. The completed Q-values will produce an
  # improved policy in `policy_output.action_weights`.
  raw_value = jnp.sum(jax.nn.softmax(prior_logits) * qvalues, axis=-1)
  use_mixed_value = False

  # The root output would be the output of MuZero representation network.
  root = mctx.RootFnOutput(
      prior_logits=prior_logits,
      value=raw_value,
      # The embedding is used only to implement the MuZero model.
      embedding=jnp.zeros([batch_size]),
  )
  # The recurrent_fn would be provided by MuZero dynamics network.
  recurrent_fn = _make_bandit_recurrent_fn(qvalues)

  # Running the search.
  policy_output = mctx.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=FLAGS.num_simulations,
      max_num_considered_actions=FLAGS.max_num_considered_actions,
      qtransform=functools.partial(
          mctx.qtransform_completed_by_mix_value,
          use_mixed_value=use_mixed_value),
  )

  # Collecting the Q-value of the selected action.
  selected_action_value = qvalues[jnp.arange(batch_size), policy_output.action]

  # We will compare the selected action to the action selected by the
  # prior policy, while using the same Gumbel random numbers.
  gumbel = policy_output.search_tree.extra_data.root_gumbel
  prior_policy_action = jnp.argmax(gumbel + prior_logits, axis=-1)
  prior_policy_action_value = qvalues[jnp.arange(batch_size),
                                      prior_policy_action]

  # Computing the policy value under the new action_weights.
  action_weights_policy_value = jnp.sum(
      policy_output.action_weights * qvalues, axis=-1)

  output = DemoOutput(
      prior_policy_value=raw_value,
      prior_policy_action_value=prior_policy_action_value,
      selected_action_value=selected_action_value,
      action_weights_policy_value=action_weights_policy_value,
  )
  return rng_key, output


def _make_bandit_recurrent_fn(qvalues):
  """Returns a recurrent_fn for a determistic bandit."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    # For the bandit, the reward will be non-zero only at the root.
    reward = jnp.where(embedding == 0,
                       qvalues[jnp.arange(action.shape[0]), action],
                       0.0)
    # On a single-player environment, use discount from [0, 1].
    # On a zero-sum self-play environment, use discount=-1.
    discount = jnp.ones_like(reward)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=jnp.zeros_like(qvalues),
        value=jnp.zeros_like(reward))
    next_embedding = embedding + 1
    return recurrent_fn_output, next_embedding

  return recurrent_fn


def main(_):
  rng_key = jax.random.PRNGKey(FLAGS.seed)
  jitted_run_demo = jax.jit(_run_demo)
  for _ in range(FLAGS.num_runs):
    rng_key, output = jitted_run_demo(rng_key)
    # Printing the obtained increase of the policy value.
    # The obtained increase should be non-negative.
    action_value_improvement = (
        output.selected_action_value - output.prior_policy_action_value)
    weights_value_improvement = (
        output.action_weights_policy_value - output.prior_policy_value)
    print("action value improvement:         %.3f (min=%.3f)" %
          (action_value_improvement.mean(), action_value_improvement.min()))
    print("action_weights value improvement: %.3f (min=%.3f)" %
          (weights_value_improvement.mean(), weights_value_improvement.min()))


if __name__ == "__main__":
  app.run(main)
