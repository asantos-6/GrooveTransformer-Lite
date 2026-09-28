# tap2drum-groovetransformer_lite-gradio

Small local web UI for recording taps and feeding them through the BaseVAE model in this repo.

The app:

- records or uploads tap audio,
- detects tap onsets,
- quantizes them into a `1 x 32 x 3` model input groove,
- runs `BaseVAE.predict(...)`,
- renders the detected taps and generated 9-voice drum output as WAV files.

## Setup

Install the app dependencies in the same environment you use for the model:

```bash
python -m pip install -r apps/requirements-tap2drum-groovetransformer_lite-gradio.txt
```

## Run

From the repository root:

```bash
python apps/tap2drum-groovetransformer_lite-gradio.py
```

Open:

```text
http://127.0.0.1:7870
```

Use a different checkpoint or port:

```bash
python apps/tap2drum-groovetransformer_lite-gradio.py --model-path base_vae_beta_0_5.pth --server-port 7871
```

Generated WAV files are written to:

```text
artifacts/tap2drum-groovetransformer_lite-gradio/
```
