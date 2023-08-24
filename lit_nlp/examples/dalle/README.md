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
python -m venv .dalle-venv
source .dalle-venv/bin/activate
pip install -r ./lit_nlp/examples/dalle-mini/requirements.txt
```

Once your virtual environment is setup, you can launch the demo with the
following command. Note that you will be prompted to log into you Weights &
Biases account and provide an API key as the demo starts up.

```sh
python -m lit_nlp.examples.dalle.demo
```
