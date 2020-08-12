# Frequently Asked Questions

<!--* freshness: { owner: 'lit-dev' reviewed: '2020-08-04' } *-->

### Can LIT work with `<insert giant transformer model here>`?

Generally, yes! But you'll probably want to use `warm_start=1.0` (or pass
`--warm_start=1.0` as a flag) to pre-compute predictions when the server loads,
so you don't have to wait when you first visit the UI.

Also, beware of memory usage: since LIT keeps the models in memory to support
new queries, only so many can fit on a single GPU. If you want to load more
models than can fit in local memory, LIT has experimental support for
remotely-hosted models on another LIT server (see
[`remote_model.py`](../language/lit/components/remote_model.py)
for more details), and you can also write a [`lit.Model`](python_api.md#models)
class to interface with your favorite serving framework.

### How many datapoints / examples can LIT handle?

It depends on your model, and on your hardware. We've successfully tested with
10k examples (the entire MultiNLI `validation_matched` split), including
embeddings from the model. But, a couple caveats:

*   LIT expects predictions to be available on the whole dataset when the UI
    loads. This can take a while if you have a lot of examples or a larger model
    like BERT. In this case, you can pass `warm_start=1.0` to the server (or use
    `--warm_start=1.0`) to warm up the cache on server load.

*   If you're using the embedding projector - i.e. if your model returns any
    `Embeddings` fields to visualize - this runs in the browser using WebGL (via
    [ScatterGL](https://github.com/PAIR-code/scatter-gl)), and so may be slow on
    older machines if you have more than a few thousand points.
