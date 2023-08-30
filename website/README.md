# Learning Interpretability Tool Website

This directory contains the code for the Learning Interpretability Tool website,
which is hosted on GitHub Pages at https://pair-code.github.io/lit.

The website consists of two parts: - A main site built with
[Eleventy](https://www.11ty.dev/), served at
`https://pair-code.github.io/lit/` - A documentation site built with
[Sphinx](https://www.sphinx-doc.org/), at
`https://pair-code.github.io/lit/documentation/`

If you wish to help make changes to our website, you've come to the right place.

**NOTE**, any updates should be made in a feature branch to the `/website/src/`
content. You can then make a pull request to the "dev" branch, and it will be
committed to the repo and deployed by a member of the LIT team.

## Dependencies

For Sphinx: `pip install sphinx myst-parser linkify-it-py furo`

For Eleventy (11ty): `npm install -g @11ty/eleventy`

Then from this directory, run `npm install`

## Directories

-   `website/src` - source for the Eleventy site
-   `website/sphinx_src` - source for the Sphinx site
-   `website/www` - source for compiled HTML site for local development
-   `docs/` compiled HTML site to be served by GitHub Pages

Don't edit `docs/` directly; rather edit the source and follow the instructions
below.

## Local Testing of the Homepage and Demos

Run `./local.sh` and navigate to the URL shown, or `http://localhost:8080`.

Note that this will auto-refresh if the 11ty source is updated, but only build
the Sphinx site once.

## Building the Site for Deployment to Github Pages

Run this any time `/website/src/` or `/website/sphinx_src/` has changed.

1.  Once you're ready to deploy, run `./deploy.sh` from the command-line while
    in the `/website` directory.
2.  This will build the site, remove the old `docs/` folder and replace it with
    the updates.
3.  Commit the docs folder to the Github repo, and wait a few minutes for Github
    CDNs to reflect your changes.
