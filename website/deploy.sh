
rm -rf www && rm -rf ../docs && mkdir ../docs && \
sphinx-build sphinx_src www/documentation && \
eleventy --input=src --output=www --config=.eleventy.deploy.js && \
cp -r www/. ../docs && rm -rf www
touch ../docs/.nojekyll
