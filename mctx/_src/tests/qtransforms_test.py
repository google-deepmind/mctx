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
"""Tests for `qtransforms.py`."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from mctx._src import qtransforms
import numpy as np


class QtransformsTest(absltest.TestCase):

  def test_mix_value(self):
    """Tests the output of _compute_mixed_value()."""
    raw_value = jnp.array(-0.8)
    prior_logits = jnp.array([-jnp.inf, -1.0, 2.0, -jnp.inf])
    probs = jax.nn.softmax(prior_logits)
    visit_counts = jnp.array([0, 4.0, 4.0, 0])
    qvalues = 10.0 / 54 * jnp.array([20.0, 3.0, -1.0, 10.0])
    mix_value = qtransforms._compute_mixed_value(
        raw_value, qvalues, visit_counts, probs)

    num_simulations = jnp.sum(visit_counts)
    expected_mix_value = 1.0 / (num_simulations + 1) * (
        raw_value + num_simulations *
        (probs[1] * qvalues[1] + probs[2] * qvalues[2]))
    np.testing.assert_allclose(expected_mix_value, mix_value)

  def test_mix_value_with_zero_visits(self):
    """Tests that zero visit counts do not divide by zero."""
    raw_value = jnp.array(-0.8)
    prior_logits = jnp.array([-jnp.inf, -1.0, 2.0, -jnp.inf])
    probs = jax.nn.softmax(prior_logits)
    visit_counts = jnp.array([0, 0, 0, 0])
    qvalues = jnp.zeros_like(probs)
    with jax.debug_nans():
      mix_value = qtransforms._compute_mixed_value(
          raw_value, qvalues, visit_counts, probs)

    np.testing.assert_allclose(raw_value, mix_value)


if __name__ == "__main__":
  absltest.main()
