#!/bin/sh
# Copyright 2020 Google LLC
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

# To use: Run './build_pip_package.sh' from this directory.
# Package output location will be printed to the console upon completion.
# Requires virtualenv and yarn to be installed.

set -ex
shopt -s extglob

command -v virtualenv >/dev/null
command -v yarn >/dev/null

dest="/tmp/lit-pip"
mkdir -p "$dest"

source_dir="$PWD/.."

pushd ../lit_nlp
yarn && yarn build
popd

cd "$dest"

mkdir -p release
pushd release

rm -rf lit_nlp
cp -LR "$source_dir/pip_package/setup.py" .
cp -LR "$source_dir/pip_package/README.md" .
cp -LR "$source_dir/lit_nlp" .
pushd lit_nlp/client
rm -rf !(build)
popd

virtualenv venv
export VIRTUAL_ENV=venv
export PATH="$PWD/venv/bin:${PATH}"
unset PYTHON_HOME

# # Require wheel for bdist_wheel command, and setuptools 36.2.0+ so that
# # env markers are handled (https://github.com/pypa/setuptools/pull/1081)
pip install -qU wheel 'setuptools>=36.2.0'

python setup.py bdist_wheel --python-tag py3 >/dev/null

ls -hal "$PWD/dist"
