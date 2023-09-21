#!/bin/bash
# This script is assumed to be run from the root of the repo.

# Ensure that python3 and yarn are installed.
command -v python3 > /dev/null
command -v yarn > /dev/null

# Build the LIT front-end with yarn.
(cd lit_nlp && yarn && yarn build)

# Build the latest source distribution and wheel, including the front-end.
python -m pip install --upgrade build
python -m build
