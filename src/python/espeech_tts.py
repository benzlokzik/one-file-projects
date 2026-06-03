#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "f5-tts",
#   "ruaccent",
#   "gradio",
#   "soundfile",
#   "huggingface-hub",
# ]
# ///
"""
ESpeech-TTS — Russian zero-shot voice cloning (F5-TTS) in a single file.

Model: https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2

CLI:
  uv run espeech_tts.py "Прив+ет, как дел+а?" \
      --ref-audio voice.wav --ref-text "Это мой голос." -o out.wav

GUI (Gradio web UI):
  uv run espeech_tts.py --gui

Notes:
- A reference audio (3-12 s, clean speech) is REQUIRED — the model clones THAT
  voice. There is no built-in default speaker.
- Russian stress is added automatically via RUAccent, UNLESS the text already
  contains '+' stress marks (e.g. 'прив+ет'). Pass --no-accent to disable.
- First run downloads the checkpoint (~2.7 GB) + vocoder; later runs are cached
  under ~/.cache/huggingface.
- For long-audio reference clipping, `--remove-silence` and pydub need ffmpeg in
  PATH; plain WAV output does not.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Static config (matches the official ESpeech-TTS Gradio space).
# ---------------------------------------------------------------------------
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

# CLI key -> (hf repo id, checkpoint filename)
MODELS: dict[str, tuple[str, str]] = {
    "RL-V2": ("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt"),
    "RL-V1": ("ESpeech/ESpeech-TTS-1_RL-V1", "espeech_tts_rlv1.pt"),
    "SFT-95K": ("ESpeech/ESpeech-TTS-1_SFT-95K", "espeech_tts_95k.pt"),
    "SFT-256K": ("ESpeech/ESpeech-TTS-1_SFT-256K", "espeech_tts_256k.pt"),
    "PODCASTER": ("ESpeech/ESpeech-TTS-1_podcaster", "espeech_tts_podcaster.pt"),
}
DEFAULT_MODEL = "RL-V2"

# vocab.txt is shared across all ESpeech checkpoints (identical tokenizer).
VOCAB_REPO = "ESpeech/ESpeech-TTS-1_podcaster"
VOCAB_FILENAME = "vocab.txt"


def fail(message: str, code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def pick_device(pref: str | None) -> str:
    """Resolve 'auto' to cuda > mps > cpu; otherwise honor the request."""
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _patch_accentizer_onnx(accentizer) -> None:
    """Work around a RUAccent/onnxruntime skew: some cached accent ONNX graphs
    declare `token_type_ids` as a required input, but RUAccent's CharTokenizer
    only emits `input_ids` + `attention_mask`. Inject all-zero token_type_ids
    (the standard single-segment default) for any session that needs them.
    """
    import numpy as np

    def wrap(session) -> None:
        try:
            required = {i.name for i in session.get_inputs()}
        except Exception:  # noqa: BLE001 — best-effort patch
            return
        if "token_type_ids" not in required:
            return
        original_run = session.run

        def run(output_names, input_feed, run_options=None):
            if "token_type_ids" not in input_feed and "input_ids" in input_feed:
                input_feed = dict(input_feed)
                input_feed["token_type_ids"] = np.zeros_like(input_feed["input_ids"])
            return original_run(output_names, input_feed, run_options)

        session.run = run  # shadow the bound method on this instance only

    for holder_name in ("accent_model", "omograph_model", "yo_homograph_model", "stress_usage_predictor"):
        session = getattr(getattr(accentizer, holder_name, None), "session", None)
        if session is not None:
            wrap(session)


@lru_cache(maxsize=1)
def load_accentizer():
    """RUAccent stress/omograph model (cached; downloads on first use)."""
    from ruaccent import RUAccent

    acc = RUAccent()
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    _patch_accentizer_onnx(acc)
    return acc


def apply_accent(text: str, accentizer) -> str:
    """Add Russian stress marks. Skipped if disabled or '+' already present."""
    text = (text or "").strip()
    if not text or accentizer is None or "+" in text:
        return text
    return accentizer.process_all(text)


class ESpeech:
    """Loaded F5-TTS model + vocoder bound to a device. Reusable across calls."""

    def __init__(self, model_name: str, device: str, vocab_path: str | None = None):
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        from huggingface_hub import hf_hub_download

        if model_name not in MODELS:
            raise ValueError(f"unknown model {model_name!r}; choose from {list(MODELS)}")

        repo_id, filename = MODELS[model_name]
        self.name = model_name
        self.device = device

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        vocab = vocab_path or hf_hub_download(repo_id=VOCAB_REPO, filename=VOCAB_FILENAME)

        self.model = load_model(DiT, MODEL_CFG, ckpt_path, vocab_file=vocab, device=device)
        self.vocoder = load_vocoder(device=device)


def synthesize(
    engine: ESpeech,
    gen_text: str,
    ref_audio: str,
    ref_text: str = "",
    accentizer=None,
    *,
    speed: float = 1.0,
    nfe_step: int = 48,
    cross_fade: float = 0.15,
    cfg_strength: float = 2.0,
    sway_coef: float = -1.0,
    seed: int = -1,
):
    """Return (waveform: np.float32[N], sample_rate: int, used_seed: int)."""
    import torch
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    torch.manual_seed(int(seed))

    gen = apply_accent(gen_text, accentizer)
    rtext = apply_accent(ref_text, accentizer)

    # Note: preprocess_ref_audio_text() takes no `device` kwarg in current
    # f5-tts; it uses the library's auto-detected device internally.
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, rtext)

    wave, sample_rate, _ = infer_process(
        ref_audio_proc,
        ref_text_proc,
        gen,
        engine.model,
        engine.vocoder,
        mel_spec_type="vocos",
        target_rms=0.1,
        cross_fade_duration=cross_fade,
        nfe_step=int(nfe_step),
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_coef,
        speed=speed,
        device=engine.device,
    )
    return wave, sample_rate, int(seed)


def save_wave(wave, sample_rate: int, out_path: str | Path, remove_silence: bool = False) -> Path:
    import soundfile as sf

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wave, sample_rate)
    if remove_silence:
        from f5_tts.infer.utils_infer import remove_silence_for_generated_wav

        remove_silence_for_generated_wav(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def launch_gui(args) -> None:
    import gradio as gr

    device = pick_device(args.device)
    engines: dict[str, ESpeech] = {}

    def get_engine(model_name: str) -> ESpeech:
        if model_name not in engines:
            gr.Info(f"Loading {model_name} on {device} (first time downloads ~2.7 GB)…")
            engines[model_name] = ESpeech(model_name, device, vocab_path=args.vocab)
        return engines[model_name]

    def run(model_name, ref_audio, ref_text, gen_text, accent, speed, nfe, cross_fade, cfg, sway, seed, remove_sil):
        if not ref_audio:
            raise gr.Error("Upload or record reference audio first (3-12 s, clean speech).")
        if not gen_text or not gen_text.strip():
            raise gr.Error("Enter some text to synthesize.")
        engine = get_engine(model_name)
        accentizer = load_accentizer() if accent else None
        wave, sr, used_seed = synthesize(
            engine, gen_text, ref_audio, ref_text or "", accentizer,
            speed=speed, nfe_step=int(nfe), cross_fade=cross_fade,
            cfg_strength=cfg, sway_coef=sway, seed=int(seed),
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        save_wave(wave, sr, tmp.name, remove_silence=remove_sil)
        return tmp.name, int(used_seed)

    with gr.Blocks(title="ESpeech-TTS") as demo:
        gr.Markdown(
            "# 🗣️ ESpeech-TTS — Russian voice cloning (F5-TTS)\n"
            "Upload a short reference voice, type Russian text, get speech in that voice. "
            "Stress is added automatically (RUAccent); use `+` to set it manually, e.g. `прив+ет`."
        )
        with gr.Row():
            with gr.Column():
                model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
                ref_audio = gr.Audio(label="Reference voice (3-12 s)", type="filepath")
                ref_text = gr.Textbox(label="Reference transcript (optional — auto-ASR if blank)", lines=2)
                gen_text = gr.Textbox(label="Text to speak", lines=4, value="Прив+ет! Это синтез р+ечи на основе F5-TTS.")
                accent = gr.Checkbox(value=True, label="Auto stress (RUAccent)")
                with gr.Accordion("Advanced", open=False):
                    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.05, label="Speed")
                    nfe = gr.Slider(4, 64, value=48, step=1, label="NFE steps (quality ↔ speed)")
                    cross_fade = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Cross-fade (s)")
                    cfg = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="CFG strength")
                    sway = gr.Slider(-1.0, 1.0, value=-1.0, step=0.05, label="Sway sampling coef")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    remove_sil = gr.Checkbox(value=False, label="Remove silences (needs ffmpeg)")
                go = gr.Button("Synthesize", variant="primary")
            with gr.Column():
                out_audio = gr.Audio(label="Output", type="filepath")
                out_seed = gr.Number(label="Seed used", precision=0, interactive=False)

        go.click(
            run,
            inputs=[model_dd, ref_audio, ref_text, gen_text, accent, speed, nfe, cross_fade, cfg, sway, seed, remove_sil],
            outputs=[out_audio, out_seed],
        )

    demo.launch(share=args.share, server_port=args.port)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ESpeech-TTS Russian voice cloning (F5-TTS) — CLI or --gui.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run espeech_tts.py 'Прив+ет, мир!' -r voice.wav -t 'Это мой голос.' -o out.wav\n"
            "  uv run espeech_tts.py --gui --share"
        ),
    )
    p.add_argument("text", nargs="?", help="Russian text to synthesize (omit when using --gui).")
    p.add_argument("-r", "--ref-audio", help="Reference voice clip to clone (3-12 s WAV/MP3).")
    p.add_argument("-t", "--ref-text", default="", help="Transcript of the reference clip. Blank = auto-ASR.")
    p.add_argument("-o", "--out", default="espeech_out.wav", help="Output WAV path (default: %(default)s).")
    p.add_argument("-m", "--model", choices=list(MODELS), default=DEFAULT_MODEL, help="Checkpoint (default: %(default)s).")
    p.add_argument("--vocab", help="Local vocab.txt override (default: downloaded from the Hub).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device (default: auto).")
    p.add_argument("--no-accent", action="store_true", help="Disable RUAccent auto-stressing.")
    p.add_argument("--remove-silence", action="store_true", help="Trim silences from output (needs ffmpeg).")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed 0.3-2.0 (default: 1.0).")
    p.add_argument("--nfe-step", type=int, default=48, help="Diffusion steps 4-64 (default: 48).")
    p.add_argument("--cross-fade", type=float, default=0.15, help="Cross-fade duration in seconds (default: 0.15).")
    p.add_argument("--cfg-strength", type=float, default=2.0, help="Classifier-free guidance strength (default: 2.0).")
    p.add_argument("--sway-coef", type=float, default=-1.0, help="Sway sampling coefficient (default: -1.0).")
    p.add_argument("--seed", type=int, default=-1, help="RNG seed; -1 = random (default: -1).")
    p.add_argument("--gui", action="store_true", help="Launch the Gradio web UI instead of CLI synthesis.")
    p.add_argument("--share", action="store_true", help="(GUI) Create a public Gradio share link.")
    p.add_argument("--port", type=int, default=None, help="(GUI) Server port.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.gui:
        launch_gui(args)
        return

    if not args.text:
        fail("missing TEXT to synthesize (or pass --gui). See -h for usage.")
    if not args.ref_audio:
        fail("--ref-audio is required: the model clones the voice in that clip.")
    if not Path(args.ref_audio).is_file():
        fail(f"reference audio not found: {args.ref_audio}")

    device = pick_device(args.device)
    print(f"loading {args.model} on {device} (first run downloads ~2.7 GB)…", file=sys.stderr)

    try:
        engine = ESpeech(args.model, device, vocab_path=args.vocab)
    except Exception as exc:  # noqa: BLE001 — surface a readable message
        fail(f"failed to load model: {exc}")

    accentizer = None if args.no_accent else load_accentizer()

    try:
        wave, sr, used_seed = synthesize(
            engine, args.text, args.ref_audio, args.ref_text, accentizer,
            speed=args.speed, nfe_step=args.nfe_step, cross_fade=args.cross_fade,
            cfg_strength=args.cfg_strength, sway_coef=args.sway_coef, seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        fail(f"synthesis failed: {exc}")

    out = save_wave(wave, sr, args.out, remove_silence=args.remove_silence)
    print(f"✓ {out}  ({len(wave) / sr:.1f}s @ {sr} Hz, seed={used_seed})")


if __name__ == "__main__":
    main()
