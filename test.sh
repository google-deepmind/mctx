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

# Runs CI tests on a local machine.
set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/mctx-env
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

# Install dependencies.
python3 -m pip install --upgrade pip uv
python3 -m uv pip install --upgrade pip setuptools wheel
python3 -m uv pip install --upgrade flake8 pytest-xdist pylint pylint-exit
python3 -m uv pip install --editable ".[test]"

# Lint with flake8.
python3 -m flake8 `find mctx -name '*.py' | xargs` --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

# Lint with pylint.
# Fail on errors, warning, and conventions.
PYLINT_ARGS="-efail -wfail -cfail"
# Lint modules and tests separately.
pylint --rcfile=.pylintrc `find mctx -name '*.py' | grep -v 'test.py' | xargs` -d R0917 || pylint-exit $PYLINT_ARGS $?
# Disable `protected-access` warnings for tests.
pylint --rcfile=.pylintrc `find mctx -name '*_test.py' | xargs` -d W0212,R0917 || pylint-exit $PYLINT_ARGS $?

# Build the package.
python3 -m uv pip install build
python3 -m build
python3 -m pip wheel --no-deps dist/mctx-*.tar.gz
python3 -m pip install mctx-*.whl

# Check types with pytype.
# Note: pytype does not support 3.12 as of 23.11.23
# See https://github.com/google/pytype/issues/1308
if [ `python -c 'import sys; print(sys.version_info.minor)'` -lt 12 ];
then
  python3 -m pip install pytype
  python3 -m pytype "mctx" -j auto --keep-going --disable import-error
fi;

# Run tests using pytest.
# Change directory to avoid importing the package from repo root.
mkdir _testing && cd _testing

# Run tests using pytest.
python3 -m pytest --numprocesses auto --pyargs mctx
cd ..

# Cleanup.
rm -rf _testing mctx-*.whl dist/

set +u
deactivate
echo "All tests passed. Congrats!"
