set -e

rm -rf www
mkdir -p www/documentation
# One-shot sphinx build
sphinx-build sphinx_src www/documentation
# Eleventy build with serving
eleventy --input=src --output=www --config=.eleventy.js --serve
