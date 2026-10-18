# Labkit-SAI-server

A small HTTP server that runs **2D (instance) segmentation** of microscopy images
using deep-learning networks on a (GPU) machine. A client, typically
[Labkit](https://github.com/xulman/labkit-ui) in Fiji, sends a 2D image and receives
back a 16-bit label image of the same size. SAM2 also accepts a box prompt.

The server is a thin [FastAPI](https://fastapi.tiangolo.com/) wrapper around several
segmentation networks:

| network          | environment | original model | custom model |
|------------------|-------------|:--------------:|:------------:|
| `cellpose`       | pytorch     | yes            | yes          |
| `instanseg`      | pytorch     | yes            | yes          |
| `maskrcnnv1COCO` | pytorch     | yes            | yes          |
| `sam2`           | pytorch     | yes (tiny, small, base-plus, large) | no |
| `stardist`       | tensorflow  | yes            | yes          |

Plus a built-in `fakenet.square` that is always available and needs no model. It is
handy for testing the connection.

## Models: original and custom

The **original** models are the default, officially published weights. The network
library downloads them itself on first use, so their files in `server-MODELS/` are
just empty placeholders. If the file is present, the model is offered.

The **custom** models are your own (re)trained weights. Drop the weights file into
`server-MODELS/` under a name that follows the format below.

## Filename format in `server-MODELS/`

Every model is one file named as a dot-separated chain of exactly four parts:

```
<model>.<network>.<environment>.model
```

- `<model>`: a free-form name of the model. `original` is reserved for the
  official weights. For SAM2 it picks the variant, e.g. `original-tiny`.
  The name must not contain dots.
- `<network>`: which network/architecture the model is for. It must be one of the
  networks the environment understands (see the table above).
- `<environment>`: the "signature" of the environment that can run it
  (`torch` or `tensorflow`, see below).
- `model`: a mandatory suffix.

Examples:

```
original.cellpose.torch.model          # official Cellpose weights (empty placeholder)
original-large.sam2.torch.model        # official SAM2.1 hiera-large (empty placeholder)
myNuclei2025.cellpose.torch.model      # custom Cellpose weights
original.stardist.tensorflow.model     # official StarDist 2D_versatile_fluo
```

A client sees each model as `<network>.<model>`, e.g. `cellpose.myNuclei2025`. Each
server lists only the files that carry its own signature. The server rescans the
folder at startup, or when the `/segmentation_2D/update_available_methods_list`
endpoint is called.

## Environments: `server-pytorch` and `server-tensorflow`

PyTorch- and TensorFlow-based networks don't coexist well in one Python environment,
so each has its own folder. Each folder holds a separate [pixi](https://pixi.sh)
environment and its own copy of the server:

| folder               | signature in `server-MODELS/` | networks |
|----------------------|-------------------------------|----------|
| `server-pytorch/`    | `torch`                       | cellpose, instanseg, maskrcnnv1COCO, sam2 |
| `server-tensorflow/` | `tensorflow`                  | stardist |

Each folder contains:

- `pixi.toml` and `pixi.lock`: the environment definition.
- `server.py`: the FastAPI server, with the same HTTP API in every environment.
- `segmentation_methods.py`: discovers the models of its own signature and maps
  each `<network>.<model>` to the code that loads and applies it. The per-network
  code lives in `segmentation-wrappers/`.

Both servers share the `server-MODELS/` folder, and you can run both at once on
different ports.

**Adding a new environment** (e.g. for JAX or ONNX) means creating a new
`server-<name>/` folder with its own `pixi.toml`, `server.py` and
`segmentation_methods.py`. The new environment must define its own signature: it
passes that string to `utils.list_models_files(methods_folder, '<signature>')`, and
its model files are then named `<model>.<network>.<signature>.model`.

## Running the PyTorch server

You need [pixi](https://pixi.sh) and a Linux machine with an NVIDIA GPU. The
environment targets CUDA 12.6.

```bash
cd server-pytorch

# one-time: create the environment (downloads PyTorch, CUDA libs, ...)
pixi install

# create the config file from the template and adjust it (see below)
cp server.cfg.example server.cfg

# start the server on an explicit port
pixi run fastapi run server.py --port 8100
```

Start the server from inside `server-pytorch/`, because it finds `../server-MODELS`
and `server.cfg` relative to that folder.

`fastapi run` listens on all interfaces (`0.0.0.0`) by default. Add `--host 127.0.0.1`
to accept local connections only, which is what you want with the SSH tunnel below.

Alternatively, enter the environment once with `pixi shell` and then run just
`fastapi run server.py --port 8100`.

### `server.cfg`

The server always reads `server.cfg` at startup and refuses to start if the file is
missing. Start from the template `server.cfg.example`:

```
# If the key is empty, N/A, NA, or NONE (case ignoring), the key is not required for the communication
SECRET-KEY=NONE
# The docs (swagger UI) are only shown if the provided value is 'YES'
SHOW-DOCS=YES
```

- `SECRET-KEY`: when set to a real value, every request must carry the HTTP header
  `X-API-Token: <the key>`. Requests without it get `404 Not Found`, so the API can't
  be discovered. Generate a good key with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- `SHOW-DOCS`: with `YES`, the interactive API docs are at `http://<server>:<port>/docs`.
  Any other value disables `/docs`, `/redoc` and `/openapi.json` entirely.

`server.cfg` contains the key and is therefore git-ignored. Don't commit it.

The server prints its effective setup at startup, e.g.
`Server config: key REQUIRED, docs hidden`.

### Quick test

```bash
# the list of available models
curl http://127.0.0.1:8100/segmentation_2D/list_available_methods

# the same, when SECRET-KEY is set
curl -H "X-API-Token: $(sed -n 's/^SECRET-KEY=//p' server.cfg)" \
     http://127.0.0.1:8100/segmentation_2D/list_available_methods
```

## GPU machine behind SSH: tunneling

When the GPU machine is reachable only over SSH, forward a local port to the server's
port. On the GPU machine, start the server and keep it local-only:

```bash
cd server-pytorch
pixi run fastapi run server.py --host 127.0.0.1 --port 8100
```

On your computer, open the tunnel. `-N` means no remote shell, just forwarding:

```bash
ssh -N -L 8100:127.0.0.1:8100 user@gpu-machine
```

Leave the tunnel open, and point the client (Labkit) to `http://127.0.0.1:8100`. The
traffic travels encrypted inside the SSH connection, and the server port is never
exposed on the GPU machine's network. To keep the server running after you log out,
start it inside `tmux` or `screen` on the GPU machine.
