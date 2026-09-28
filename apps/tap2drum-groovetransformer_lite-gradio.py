from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import BaseVAE


APP_NAME = "tap2drum-groovetransformer_lite-gradio"
APP_TITLE = "Tap2Drum GrooveTransformer Lite Gradio"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / APP_NAME
DEFAULT_MODEL_PATH = REPO_ROOT / "base_vae_beta_0_2.pth"

SR = 22050
STEPS_PER_BEAT = 4
MODEL_STEPS = 32
DEFAULT_BPM = 120
DEFAULT_THRESHOLD = 0.25
DEFAULT_MIN_TAP_GAP_MS = 70
DEFAULT_OUTPUT_THRESHOLD = 0.5

DRUM_NAMES = [
    "kick",
    "snare",
    "closed_hat",
    "open_hat",
    "low_tom",
    "mid_tom",
    "high_tom",
    "crash",
    "ride",
]

_MODEL: BaseVAE | None = None
_MODEL_PATH: Path | None = None
_DEVICE: torch.device | None = None


def load_model(
    model_path: Path,
    model_class,
    params_dict=None,
    is_evaluating: bool = True,
    device: torch.device | None = None,
):
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        loaded_dict = torch.load(model_path, map_location=device) if device is not None else torch.load(model_path)
    except Exception:
        loaded_dict = torch.load(model_path, map_location=torch.device("cpu"))

    if params_dict is None:
        if "params" not in loaded_dict:
            raise ValueError("Checkpoint does not contain params. Pass params_dict as a dict or JSON path.")
        params_dict = loaded_dict["params"]

    if isinstance(params_dict, str):
        with open(params_dict, "r", encoding="utf-8") as handle:
            params_dict = json.load(handle)

    model = model_class(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    model.to(device or torch.device("cpu"))
    if is_evaluating:
        model.eval()
    return model


def get_model(model_path: Path) -> tuple[BaseVAE, torch.device]:
    global _MODEL, _MODEL_PATH, _DEVICE
    model_path = model_path.resolve()
    if _MODEL is not None and _MODEL_PATH == model_path and _DEVICE is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _MODEL = load_model(model_path, BaseVAE, device=_DEVICE)
    _MODEL_PATH = model_path
    return _MODEL, _DEVICE


def read_audio(path: Path, target_sr: int = SR) -> tuple[np.ndarray, int]:
    sr, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(1, np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    if sr != target_sr:
        gcd = int(np.gcd(sr, target_sr))
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    wavfile.write(path, sr, (audio * 32767.0).astype(np.int16))
    return path


def envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    rectified = np.abs(audio).astype(np.float32)
    window = max(1, int(0.012 * sr))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(rectified, kernel, mode="same")
    peak = float(np.max(smoothed)) if smoothed.size else 0.0
    return smoothed / peak if peak > 0 else smoothed


def detect_taps(
    audio: np.ndarray,
    sr: int,
    threshold: float,
    min_tap_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    env = envelope(audio, sr)
    min_distance = max(1, int(sr * float(min_tap_gap_ms) / 1000.0))
    peaks, props = find_peaks(env, height=float(threshold), distance=min_distance)
    if peaks.size == 0:
        raise gr.Error("No taps were detected. Try tapping louder or lowering the detection threshold.")
    strengths = np.asarray(props["peak_heights"], dtype=np.float32)
    strengths = strengths / (float(np.max(strengths)) + 1e-8)
    return peaks.astype(np.int64), strengths


def taps_to_model_input(
    tap_samples: np.ndarray,
    tap_strengths: np.ndarray,
    sr: int,
    bpm: float,
    steps: int = MODEL_STEPS,
    steps_per_beat: int = STEPS_PER_BEAT,
) -> tuple[np.ndarray, dict[str, float]]:
    step_sec = 60.0 / float(bpm) / float(steps_per_beat)
    tap_times = tap_samples.astype(np.float32) / float(sr)
    start_time = max(0.0, float(tap_times[0] - 0.5 * step_sec))
    relative_times = tap_times - start_time
    grid_positions = np.rint(relative_times / step_sec).astype(np.int64)

    hvo = np.zeros((steps, 3), dtype=np.float32)
    used = 0
    for grid_pos, tap_time, strength in zip(grid_positions, relative_times, tap_strengths):
        if grid_pos < 0 or grid_pos >= steps:
            continue
        grid_time = float(grid_pos) * step_sec
        offset = np.clip((float(tap_time) - grid_time) / step_sec, -0.5, 0.5)
        velocity = float(np.clip(0.25 + 0.75 * strength, 0.05, 1.0))
        if hvo[grid_pos, 0] == 0 or velocity > hvo[grid_pos, 1]:
            hvo[grid_pos, 0] = 1.0
            hvo[grid_pos, 1] = velocity
            hvo[grid_pos, 2] = offset
        used += 1

    if hvo[:, 0].sum() == 0:
        raise gr.Error("Detected taps fell outside the 2-bar model window. Try a shorter recording.")

    info = {
        "detected_taps": float(len(tap_samples)),
        "used_taps": float(used),
        "start_time_sec": float(start_time),
        "step_sec": float(step_sec),
    }
    return hvo, info


def _exp_env(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.linspace(0.0, decay, n, endpoint=False))


def _noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, n)


def drum_voice(kind: str, velocity: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    velocity = float(np.clip(velocity, 0.0, 1.0))

    if kind == "kick":
        n = int(0.34 * sr)
        t = np.arange(n) / sr
        freq = 48 + 95 * np.exp(-t * 18)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return velocity * np.sin(phase) * _exp_env(n, 8.0)

    if kind == "snare":
        n = int(0.24 * sr)
        t = np.arange(n) / sr
        tone = 0.35 * np.sin(2 * np.pi * 190 * t) * _exp_env(n, 9.0)
        buzz = _noise(n, seed) * _exp_env(n, 12.0)
        return velocity * (tone + 0.75 * buzz)

    if kind in {"closed_hat", "ride"}:
        n = int((0.07 if kind == "closed_hat" else 0.18) * sr)
        x = np.concatenate([[0.0], np.diff(_noise(n, seed))])
        return velocity * 0.45 * x * _exp_env(n, 18.0 if kind == "closed_hat" else 8.0)

    if kind in {"open_hat", "crash"}:
        n = int((0.42 if kind == "open_hat" else 0.85) * sr)
        x = np.concatenate([[0.0], np.diff(_noise(n, seed))])
        return velocity * 0.35 * x * _exp_env(n, 5.5 if kind == "open_hat" else 3.0)

    tom_freq = {"low_tom": 95, "mid_tom": 130, "high_tom": 175}.get(kind, 120)
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    freq = tom_freq + 45 * np.exp(-t * 14)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return velocity * 0.75 * np.sin(phase) * _exp_env(n, 7.0)


def render_hvo_audio(
    hvo: np.ndarray,
    bpm: float,
    sr: int = SR,
    steps_per_beat: int = STEPS_PER_BEAT,
    one_voice_name: str = "closed_hat",
) -> np.ndarray:
    hvo = np.asarray(hvo, dtype=np.float32)
    if hvo.ndim == 3:
        hvo = hvo[0]

    n_steps = hvo.shape[0]
    step_sec = 60.0 / float(bpm) / float(steps_per_beat)
    length = int((n_steps * step_sec + 1.2) * sr)
    audio = np.zeros(length, dtype=np.float32)

    if hvo.shape[1] == 3:
        hits = hvo[:, 0:1]
        velocities = hvo[:, 1:2]
        offsets = hvo[:, 2:3]
        names = [one_voice_name]
    else:
        n_voices = hvo.shape[1] // 3
        hits = hvo[:, :n_voices]
        velocities = hvo[:, n_voices : 2 * n_voices]
        offsets = hvo[:, 2 * n_voices : 3 * n_voices]
        names = DRUM_NAMES[:n_voices]

    event_id = 0
    for step in range(n_steps):
        for voice, name in enumerate(names):
            if hits[step, voice] <= 0.5:
                continue
            offset_sec = float(np.clip(offsets[step, voice], -0.5, 0.5)) * step_sec
            start = int((step * step_sec + offset_sec) * sr)
            start = max(0, min(start, length - 1))
            sample = drum_voice(name, velocities[step, voice], sr=sr, seed=event_id)
            end = min(length, start + len(sample))
            audio[start:end] += sample[: end - start]
            event_id += 1

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = 0.92 * audio / peak
    return audio.astype(np.float32)


def mix_audio(original: np.ndarray, generated: np.ndarray, generated_gain: float = 0.8) -> np.ndarray:
    length = max(len(original), len(generated))
    left = np.pad(original, (0, length - len(original)))
    right = np.pad(generated * float(generated_gain), (0, length - len(generated)))
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo = stereo / peak
    return stereo


def process_audio(
    audio_path: str | None,
    model_path: str,
    bpm: float,
    tap_threshold: float,
    min_tap_gap_ms: float,
    output_threshold: float,
    output_dir: str,
):
    if audio_path is None:
        raise gr.Error("Record or upload tap audio first.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model, device = get_model(Path(model_path))

    original_audio, sr = read_audio(Path(audio_path), target_sr=SR)
    tap_samples, tap_strengths = detect_taps(original_audio, sr, tap_threshold, min_tap_gap_ms)
    input_hvo, info = taps_to_model_input(tap_samples, tap_strengths, sr, bpm)
    input_tensor = torch.tensor(input_hvo[None], dtype=torch.float32, device=device)

    with torch.no_grad():
        output_hvo, latent_z = model.predict(input_tensor, threshold=float(output_threshold))

    output_hvo_np = output_hvo.detach().cpu().numpy()
    detected_taps_audio = render_hvo_audio(input_hvo, bpm=bpm, sr=sr, one_voice_name="closed_hat")
    generated_audio = render_hvo_audio(output_hvo_np, bpm=bpm, sr=sr)
    mixed_audio = mix_audio(original_audio, generated_audio)

    taps_path = write_wav(output_root / "detected_taps.wav", detected_taps_audio, sr)
    output_path = write_wav(output_root / "groove_transformer_output.wav", generated_audio, sr)
    mix_path = write_wav(output_root / "recording_plus_output.wav", mixed_audio, sr)

    output_hits = output_hvo_np[0, :, :9]
    hit_counts = {
        name: int(count)
        for name, count in zip(DRUM_NAMES, output_hits.sum(axis=0).astype(int))
    }
    status = (
        f"Done. Detected taps: {int(info['detected_taps'])} | "
        f"Used in 2-bar window: {int(input_hvo[:, 0].sum())} | "
        f"Input shape: {tuple(input_tensor.shape)} | "
        f"Output shape: {tuple(output_hvo_np.shape)} | "
        f"Latent shape: {tuple(latent_z.shape)} | "
        f"Device: {device}\\n"
        f"Output hit counts: {hit_counts}"
    )
    return str(taps_path), str(output_path), str(mix_path), status


def build_app(default_model_path: Path, default_output_dir: Path) -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Record or upload taps, quantize them into the BaseVAE input groove, and generate a 9-voice drum pattern."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="Tap recording", sources=["microphone", "upload"], type="filepath")
                model_path = gr.Textbox(label="Model checkpoint", value=str(default_model_path))
                output_dir = gr.Textbox(label="Output directory", value=str(default_output_dir))
                bpm = gr.Slider(minimum=60, maximum=200, value=DEFAULT_BPM, step=1, label="BPM")
                tap_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=DEFAULT_THRESHOLD,
                    step=0.01,
                    label="Tap detection threshold",
                )
                min_tap_gap_ms = gr.Slider(
                    minimum=20,
                    maximum=250,
                    value=DEFAULT_MIN_TAP_GAP_MS,
                    step=5,
                    label="Minimum tap gap (ms)",
                )
                output_threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=DEFAULT_OUTPUT_THRESHOLD,
                    step=0.01,
                    label="Model hit threshold",
                )
                run_button = gr.Button("Generate drums", variant="primary")
            with gr.Column():
                tap_audio = gr.Audio(label="Detected taps rendered as hi-hat", type="filepath")
                drum_audio = gr.Audio(label="Generated drum output", type="filepath")
                mix_audio_output = gr.Audio(label="Original recording + generated output", type="filepath")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        run_button.click(
            fn=process_audio,
            inputs=[audio, model_path, bpm, tap_threshold, min_tap_gap_ms, output_threshold, output_dir],
            outputs=[tap_audio, drum_audio, mix_audio_output, status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run the {APP_NAME} app.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7870)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(default_model_path=args.model_path, default_output_dir=args.output_dir)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(args.output_dir.resolve())],
    )


if __name__ == "__main__":
    main()
