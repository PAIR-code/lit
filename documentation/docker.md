# Running LIT in a Docker container

<!--* freshness: { owner: 'lit-dev' reviewed: '2021-12-16' } *-->

Users might want to deploy LIT onto servers for public-facing, long-running
instances. This is how we host the LIT demos found on
https://pair-code.github.io/lit/demos/. Specifically, we deploy containerized
LIT instances through Google Cloud's Google Kubernetes Engine (GKE).

The code required to deploy LIT as a containerized web app can be seen by
looking at our masked language model demo.

First, let's look at the relevant code in
[`lm_demo.py`](../lit_nlp/examples/lm_demo.py):

The `get_wsgi_app()` method is what is invoked by the Dockerfile. It sets the
`server_type` to `"external"`, constructs the LIT `Server` instance, and returns
the result of it's `serve()` method which is the underlying `LitApp` WSGI
application.

Now, let's explore the [`Dockerfile`](https://github.com/PAIR-code/lit/blob/main/Dockerfile):

The Dockerfile installs all necessary dependencies for LIT and builds the
front-end code from source. Then it runs [gunicorn](https://gunicorn.org/) as
the HTTP server, invoking the `get_wsgi_app()` method from our demo file to get
the WSGI app to serve. The options provided to gunicorn for our use-case can be
found in
[`gunicorn_config.py`](../lit_nlp/examples/gunicorn_config.py).

Then, our container is built and deployed following the basics of the
[GKE tutorial](https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app).

You can launch any of the built-in demos from the same Docker image. First,
build the image with `docker build -t lit:latest .`. Running a container from
this image, as with `docker run --rm -p 5432:5432 lit:latest`, will start
the GLUE demo and mount it on port 5432.You cna change the demo by setting the
`$DEMO_NAME` environment variable to one of the valid demo names, and you can
change the port by setting the `$DEMO_PORT` environment variable. Remember to
change the `-p` option to forward the container's port to the host.
