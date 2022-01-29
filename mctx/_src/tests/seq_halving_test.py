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
"""Tests for `seq_halving.py`."""
from absl.testing import absltest
from mctx._src import seq_halving


class SeqHalvingTest(absltest.TestCase):

  def _check_visits(self, expected_results, max_num_considered_actions,
                    num_simulations):
    """Compares the expected results to the returned considered visits."""
    self.assertLen(expected_results, num_simulations)
    results = seq_halving.get_sequence_of_considered_visits(
        max_num_considered_actions, num_simulations)
    self.assertEqual(tuple(expected_results), results)

  def test_considered_min_sims(self):
    # Using exactly `num_simulations = max_num_considered_actions *
    #   log2(max_num_considered_actions)`.
    num_sims = 24
    max_num_considered = 8
    expected_results = [
        0, 0, 0, 0, 0, 0, 0, 0,  # Considering 8 actions.
        1, 1, 1, 1,              # Considering 4 actions.
        2, 2, 2, 2,              # Considering 4 actions, round 2.
        3, 3, 4, 4, 5, 5, 6, 6,  # Considering 2 actions.
    ]  # pyformat: disable
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_extra_sims(self):
    # Using more simulations than `max_num_considered_actions *
    #   log2(max_num_considered_actions)`.
    num_sims = 47
    max_num_considered = 8
    expected_results = [
        0, 0, 0, 0, 0, 0, 0, 0,  # Considering 8 actions.
        1, 1, 1, 1,              # Considering 4 actions.
        2, 2, 2, 2,              # Considering 4 actions, round 2.
        3, 3, 3, 3,              # Considering 4 actions, round 3.
        4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
        11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17,
    ]  # pyformat: disable
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_less_sims(self):
    # Using a very small number of simulations.
    num_sims = 2
    max_num_considered = 8
    expected_results = [0, 0]
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_less_sims2(self):
    # Using `num_simulations < max_num_considered_actions *
    #   log2(max_num_considered_actions)`.
    num_sims = 13
    max_num_considered = 8
    expected_results = [
        0, 0, 0, 0, 0, 0, 0, 0,  # Considering 8 actions.
        1, 1, 1, 1,              # Considering 4 actions.
        2,
    ]  # pyformat: disable
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_not_power_of_2(self):
    # Using max_num_considered_actions that is not a power of 2.
    num_sims = 24
    max_num_considered = 7
    expected_results = [
        0, 0, 0, 0, 0, 0, 0,  # Considering 7 actions.
        1, 1, 1, 2, 2, 2,     # Considering 3 actions.
        3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    ]  # pyformat: disable
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_action0(self):
    num_sims = 16
    max_num_considered = 0
    expected_results = range(num_sims)
    self._check_visits(expected_results, max_num_considered, num_sims)

  def test_considered_action1(self):
    num_sims = 16
    max_num_considered = 1
    expected_results = range(num_sims)
    self._check_visits(expected_results, max_num_considered, num_sims)


if __name__ == "__main__":
  absltest.main()
