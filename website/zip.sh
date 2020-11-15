
DATE=`date +%m-%d-%s` 
rm -rf www && eleventy --input=src --output=www --config=.eleventy.js && cd www && zip -r ../site-$DATE.zip .
