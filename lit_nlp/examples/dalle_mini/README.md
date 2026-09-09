Dalle_Mini Demo for the Learning Interpretability Tool
=======================================================

This demo showcases how LIT can be used in text-to-image generation mode. It is
based on the mini-dalle Mini model
(https://www.piwheels.org/project/dalle-mini/).

You will need a standalone virtual environment for the Python libraries, which
you can set up using the following commands from the root of the LIT repo.

```sh
# Create the virtual environment. You may want to use python3 or python3.10
# depends on how many Python versions you have installed and their aliases.
python -m venv .dalle-mini
source .dalle-mini/bin/activate
# This requirements.txt file will also install the core LIT library deps.
pip install -r ./lit_nlp/examples/dalle_mini/requirements.txt
# The LIT web app still needs to be built in the usual way.
(cd ./lit_nlp && yarn && yarn build)
```

Once your virtual environment is setup, you can launch the demo with the
following command.

```sh
python -m lit_nlp.examples.dalle_mini.demo
```