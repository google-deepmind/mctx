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
"""Tests for Mctx."""

from absl.testing import absltest
import mctx


class MctxTest(absltest.TestCase):
  """Test mctx can be imported correctly."""

  def test_import(self):
    self.assertTrue(hasattr(mctx, "gumbel_muzero_policy"))
    self.assertTrue(hasattr(mctx, "muzero_policy"))
    self.assertTrue(hasattr(mctx, "qtransform_by_min_max"))
    self.assertTrue(hasattr(mctx, "qtransform_by_parent_and_siblings"))
    self.assertTrue(hasattr(mctx, "qtransform_completed_by_mix_value"))
    self.assertTrue(hasattr(mctx, "PolicyOutput"))
    self.assertTrue(hasattr(mctx, "RootFnOutput"))
    self.assertTrue(hasattr(mctx, "RecurrentFnOutput"))


if __name__ == "__main__":
  absltest.main()
