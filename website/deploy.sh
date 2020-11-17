
rm -rf www && rm -rf ../docs && mkdir ../docs && eleventy --input=src --output=www --config=.eleventy.deploy.js && cp -r www/. ../docs
