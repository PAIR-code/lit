# Learning Interpretability Tool Website

This directory contains the code for the Learning Interpretability Tool website, which is hosted as a github page.

If you wish to help make changes to our website, you've come to the right place. **NOTE**, any updates should be made in a feature branch to the `/website/src/` content. You can then make a pull request to the "dev" branch, and it will be committed to the repo and deployed by a member of the LIT team.

## Local Testing of the Homepage and Demos

1. Install 11ty to your machine (globally if you have permissions), run `npm install -g @11ty/eleventy`. You may need to use sudo in front of this command, depending on the setup of npm on your computer of choice.
2. From this directory, install dependencies by running `npm install`
3. From this directory, run `./local.sh`
4. Navigate to `http://localhost:8080`

## Building the Site for Deployment to Github Pages

1. Make changes to the site inside the `/website/src/` directory, using local testing instructions above to validate changes.
2. Once you're ready to deploy, run `./deploy.sh` from the command-line while in the `/website` directory.
3. This will build the site, remove the old docs folder and replace it with the updates.
4. Commit the docs folder to the Github repo, and wait a few minutes for Github CDNs to reflect your changes.
