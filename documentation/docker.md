# Running LIT in a Docker container

<!--* freshness: { owner: 'lit-dev' reviewed: '2022-11-15' } *-->

Users might want to deploy LIT onto servers for public-facing, long-running
instances. This is how we host the LIT demos found on
https://pair-code.github.io/lit/demos/. This doc describes the basic usage of
LIT's built-in demos, how to integrate your custom demo into this

## Basic Usage

LIT can be run as a containerized app using [Docker](https://www.docker.com/) or
your preferred engine. This is how we run our
[hosted demos](https://pair-code.github.io/lit/demos/).

We provide a basic
[`Dockerfile`](../lit_nlp/Dockerfile) that you can
use to build and run any of the demos in the `lit_nlp/examples` directory. The
`Dockerfile` installs all necessary dependencies for LIT and builds the
front-end code from source. Then it runs [gunicorn](https://gunicorn.org/) as
the HTTP server, invoking the `get_wsgi_app()` method from our demo file to get
the WSGI app to serve. The options provided to gunicorn for our use-case can be
found in
[`gunicorn_config.py`](../lit_nlp/examples/gunicorn_config.py).
You can find a reference implementation in
[`glue_demo.py`](../lit_nlp/examples/glue_demo.py) or
[`lm_demo.py`](../lit_nlp/examples/lm_demo.py).

Use the following shell commands to build the default Docker image for LIT from
the provided `Dockerfile`, and then run a container from that image. Comments
are provided in-line to help explain what each step does.

```shell
# Build the docker image using the -t argument to name the image. Remember to
# include the trailing . so Docker knows where to look for the Dockerfile.
docker build -t lit-app .

# Now you can run LIT as a containerized app using the following command. Note
# that the last parameter to the run command is the value you passed to the -t
# argument in the build command above.
docker run --rm -p 5432:5432 lit-app
```

The image above defaults to launching the GLUE demo on port 5432, but you can
override this using the DEMO_NAME and DEMO_PORT environment variables, as shown
below.

```shell
# DEMO_NAME is used to complete the Python module path
#
#     "lit_nlp.examples.$DEMO_NAME"
#
# Therefore, valid values for DEMO_NAME are Python module paths in the
# lit_nlp/examples directory, such as
#
#   * direct children -- glue_demo, lm_demo, image_demo, t5_demo, etc.
#   * And nested children -- coref.coref_demo, is_eval.is_eval_demo, etc.
docker run --rm -p 5432:5432 -e DEMO_NAME=lm_demo lit-app

# Use the DEMO_PORT environment variable as to change the port that LIT uses in
# the container. Be sure to also change the -p option to map the container's
# DEMO_PORT to a port on the host system.
docker run --rm -p 2345:2345 -e DEMO_PORT=2345 lit-app

# Bringing this all together, you can run multiple LIT apps in separate
# containers on your machine using the combination of the DEMO_NAME and
# DEMO_PORT arguments, and docker run with the -d flag to run the container in
# the background.
docker run -d -p 5432:5432 -e DEMO_NAME=t5_demo lit-app
docker run -d -p 2345:2345 -e DEMO_NAME=lm_demo -e DEMO_PORT=2345 lit-app
```

## Integrating Custom LIT Instances with the Default Docker Image

Many LIT users create their own custom LIT server script to demo or serve, which
involves creating an executable Python module with a `main()` method, as
described in the [Python API docs](g3doc/api.md#adding-models-and-data).

These custom server scripts can be easily integrated with LIT's default image as
long as your server meets two requirements:

1.  Ensure your server script is located in the `lit_nlp/examples` directory (or
    in a nested directory under `lit_nlp/examples`).
2.  Ensure that your server script defines a `get_wsgi_app()` function similar
    to the minimal example shown below.

```python
def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Return WSGI app for container-hosted demos."""
  # Set any flag defaults for this LIT instance
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags before calling main()
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("get_wsgi_app() called with unused args: %s", unused)
  return main([])
```

Assuming your custom script meets the two requirements above, you can simply
rebuild the default Docker image and run a container using the steps above,
ensuring that you pass the `-e DEMO_NAME=your_server_script_path_here` to the
run command.

A more detailed description of the `get_wsgi_app()` code can be found below.

```python
def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Returns a WSGI app for gunicorn to consume in container-hosted demos."""
  # Start by setting any default values for flags your LIT instance requires.
  # Here we set:
  #
  #     server_type to "external" (required), and
  #     demo_mode to "True" (optional)
  #
  # You can add additional defaults as required for your use case.
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)

  # Parse any parameters from flags before calling main(). All flags should
  # defined using one of absl's flags.DEFINE methods.
  #
  # Note the use of the known_only=True parameter here. This ensures that only
  # those flags that have been define using one of absl's flags.DEFINE methods
  # will be parsed from the command line arguments in sys.argv. All unused
  # arguments will be returned as a Sequence[str].
  unused = flags.FLAGS(sys.argv, known_only=True)

  # Running a LIT instance in a container based on the default Dockerfile and
  # image will always produce unused arguments, because sys.argv contains the
  # command and parameters used to run the gunicorn sever. While not stricly
  # required, we recommend logging these to the console, e.g., in case you need
  # to verify the value of an environment variable.
  if unused:
    logging.info("get_wsgi_app() called with unused args: %s", unused)

  # Always pass an empty list to main() inside of get_wsgi_app() functions, as
  # absl apps are supposed to use absl.flags to define any and all flags
  # required to run the app.
  return main([])
```

## Building Your Own Image

Coming soon.
