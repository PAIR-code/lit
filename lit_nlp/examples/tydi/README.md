TyDi QA Demo for the Learning Interpretability Tool
=======================================================

This demo showcases how LIT can be used to a multilingual question-answering
model trained on the [TyDi QA dataset](https://doi.org/10.1162/tacl_a_00317)
using FLAX.

You will need a stand-alone virtual environment for the Python libraries, which you can set up using the following commands from the root of the LIT repo.

```sh
# Create the virtual environment. You may want to use python3 or python3.10
# depends on how many Python versions you have installed and their aliases.
python -m venv .tydi-venv
source .tydi-venv/bin/activate
# This requirements.txt file will also install the core LIT library deps.
pip install -r ./lit_nlp/examples/tydi/requirements.txt
# The LIT web app still needs to be built in the usual way.
(cd ./lit_nlp && yarn && yarn build)
```

Once your virtual environment is setup, you can launch the demo with the
following command.

```sh
python -m lit_nlp.examples.tydi.demo
```
