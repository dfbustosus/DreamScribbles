# DreamScribbles: The AI Art Transformer

Turn a simple sketch into a stunning image in your chosen style. Draw a scribble and let ControlNet + Stable Diffusion transform it.

## Highlights

- Simple canvas. Draw with your mouse or finger.
- Style presets. Pixar, Watercolor, Photorealistic, Anime, Oil Painting, Cyberpunk, and more.
- Fast prototype. Built with Python, Gradio UI, and Hugging Face Diffusers.
- Production-ready scaffold. CI, tests, linting, typing, Docker CPU/CUDA, and clear docs.

## Demo (Local)

1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install

```bash
pip install -U pip
pip install -e ".[dev]"
```

3) Optional: set your Hugging Face token (if needed for gated models)

```bash
cp .env.example .env
# edit .env and set DREAMSCRIBBLES_HF_TOKEN=hf_xxx
```

4) Run

```bash
python -m dreamscribbles
```

Open http://127.0.0.1:7860 and draw a scribble.

## Docker (CPU)

```bash
docker build -f docker/Dockerfile.cpu -t dreamscribbles:cpu .
docker run --rm -p 7860:7860 --env-file .env dreamscribbles:cpu
```

## Docker (CUDA)

Requires an NVIDIA GPU with recent drivers and nvidia-container-toolkit.

```bash
docker build -f docker/Dockerfile.cuda -t dreamscribbles:cuda .
docker run --rm --gpus all -p 7860:7860 --env-file .env dreamscribbles:cuda
```

## Config

`.env` (see `.env.example`) controls defaults. Key variables:

- `DREAMSCRIBBLES_HF_TOKEN`: Optional Hugging Face token
- `DREAMSCRIBBLES_MODEL_ID`: Base SD model (default: runwayml/stable-diffusion-v1-5)
- `DREAMSCRIBBLES_CONTROLNET_ID`: ControlNet model (default: lllyasviel/sd-controlnet-scribble)
- `DREAMSCRIBBLES_DEVICE_PREFERENCE`: One of `cuda`, `mps`, `cpu` (auto-detected if empty)
- `DREAMSCRIBBLES_SHARE`: `true` to enable Gradio share links

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker.yml
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.cuda
│   └── .dockerignore
├── scripts/
│   └── download_models.py
├── src/
│   └── dreamscribbles/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── styles.py
│       ├── utils/
│       │   └── image_ops.py
│       ├── inference/
│       │   └── pipeline.py
│       └── web/
│           └── app.py
├── tests/
│   ├── test_image_ops.py
│   └── test_styles.py
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pytest.ini
└── pyproject.toml
```

## Notes on Models and Performance

- Defaults use SD 1.5 + ControlNet Scribble, which runs on CPU (slow), MPS (Apple Silicon), or CUDA (fastest).
- First run downloads several GB of models. Use `scripts/download_models.py` to prefetch.
- Safety checker stays enabled by default; see `pipeline.py` for configuration.

## Testing & CI

- `pytest` for tests; heavy inference is not run in CI. Only fast utility tests execute.
- Lint and type checks via `ruff`, `black`, `isort`, and `mypy`.

## Security and Conduct

See `SECURITY.md` and `CODE_OF_CONDUCT.md`.

## License

MIT License. See `LICENSE`.
