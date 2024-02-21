DALL·E Mini Demo for the Learning Interpretability Tool
=======================================================

This demo showcases how LIT can be used to visualize generative image models,
using [DALL·E Mini](https://github.com/borisdayma/dalle-mini) as a case study.

Due to the specific requirements of the `dalle-mini` library, this demo
necessitates some specific setup.

First, you will need to [create an account](https://wandb.ai/login) on Weights &
Biases in order to generate an API key to use some of the models included in
this demo.

Second, you will need a stand-alone virtual environment for the Python
libraries, which you can setup using the following commands from the root of the
LIT repo.

```sh
# Create the virtual environment. You may want to use python3 or python3.10
# depends on how many Python versions you have installed and their aliases.
python -m venv .dalle-venv
source .dalle-venv/bin/activate
# This requirements.txt file will also install the core LIT library deps.
pip install -r ./lit_nlp/examples/dalle/requirements.txt
# The LIT web app still needs to be built in the usual way.
(cd ./lit_nlp && yarn && yarn build)
```

Once your virtual environment is setup, you can launch the demo with the
following command. Note that you will be prompted to log into you Weights &
Biases account and provide an API key as the demo starts up.

```sh
python -m lit_nlp.examples.dalle.demo
```
