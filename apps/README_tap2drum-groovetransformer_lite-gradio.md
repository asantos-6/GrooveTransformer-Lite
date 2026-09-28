# tap2drum-groovetransformer_lite-gradio

Small local web UI for recording taps and feeding them through the BaseVAE model in this repo.

The app:

- records or uploads tap audio,
- detects tap onsets,
- quantizes them into a `1 x 32 x 3` model input groove,
- runs `BaseVAE.predict(...)`,
- renders the detected taps and generated 9-voice drum output as WAV files.

## Timing

The model input is one 2-bar window: 32 sixteenth-note steps. At 120 BPM, that musical window is 4 seconds, plus a short audio tail for rendered drum decay.

Use **BPM** when you know the tap tempo. Use **Fit 2 bars to detected taps** when the recording is intended to be exactly one 2-bar phrase and you want the app to infer a BPM from the detected tap span.

Longer recordings are currently reduced to one 2-bar model window. The mixed output aligns that window back to the original recording at the detected tap-window start time.

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
