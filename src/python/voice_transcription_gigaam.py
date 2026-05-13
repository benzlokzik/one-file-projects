#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "gigaam[longform]",
# ]
#
# [tool.uv.sources]
# gigaam = { git = "https://github.com/salute-developers/GigaAM" }
# ///
"""
Transcribe an audio file with GigaAM from the terminal.

Notes:
- For short audio (<= 25 seconds), this uses model.transcribe(...)
- For longer audio, it automatically switches to model.transcribe_longform(...)
- Long-form transcription requires extra dependencies and a Hugging Face token:
    pip install "gigaam[longform]"
  and access to:
    pyannote/segmentation-3.0
"""

import argparse
import os
import sys
from pathlib import Path

import gigaam
import torch
from gigaam.preprocess import SAMPLE_RATE
from gigaam.vad_utils import get_pipeline

SHORT_AUDIO_LIMIT_SECONDS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with GigaAM from the terminal."
    )

    parser.add_argument(
        "audio",
        help="Path to the audio file (mp3, wav, m4a, etc.)",
    )

    parser.add_argument(
        "--model",
        default="v3_e2e_rnnt",
        help=(
            "GigaAM ASR model to use. "
            "Examples: ctc, rnnt, e2e_ctc, e2e_rnnt, v2_ctc, v3_rnnt, v3_e2e_rnnt"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Run on CPU or GPU (default: auto)",
    )

    parser.add_argument(
        "--fp16-encoder",
        dest="fp16_encoder",
        action="store_true",
        default=True,
        help="Use FP16 encoder weights on non-CPU devices (default: enabled)",
    )
    parser.add_argument(
        "--no-fp16-encoder",
        dest="fp16_encoder",
        action="store_false",
        help="Disable FP16 encoder weights",
    )

    parser.add_argument(
        "--use-flash",
        action="store_true",
        help="Enable flash attention if supported and installed",
    )

    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Output word-level timestamps",
    )

    parser.add_argument(
        "--no-timestamps",
        dest="print_timestamps",
        action="store_false",
        help="Suppress timestamps in the output",
    )
    parser.set_defaults(print_timestamps=True)

    parser.add_argument(
        "--longform",
        action="store_true",
        help="Force long-form transcription even for short audio",
    )

    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face token for long-form VAD "
            "(or set HF_TOKEN in the environment)"
        ),
    )

    parser.add_argument(
        "--download-root",
        default=None,
        help="Custom directory for downloaded/cached GigaAM checkpoints",
    )

    # Long-form VAD chunking parameters
    parser.add_argument(
        "--max-duration",
        type=float,
        default=22.0,
        help="Preferred max chunk duration for long-form VAD (seconds)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=15.0,
        help="Preferred min chunk duration for long-form VAD (seconds)",
    )
    parser.add_argument(
        "--strict-limit-duration",
        type=float,
        default=30.0,
        help="Hard max chunk duration for long-form VAD (seconds)",
    )
    parser.add_argument(
        "--new-chunk-threshold",
        type=float,
        default=0.2,
        help="Silence/gap threshold for starting a new chunk (seconds)",
    )

    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def format_range(start: float, end: float) -> str:
    return f"[{start:.2f}s → {end:.2f}s]"


def get_audio_duration_seconds(audio_path: str) -> float:
    audio = gigaam.load_audio(audio_path)
    return audio.shape[-1] / SAMPLE_RATE


def print_short_result(
    result,
    *,
    duration_seconds: float,
    print_timestamps: bool,
    word_timestamps: bool,
) -> None:
    if word_timestamps and result.words:
        for word in result.words:
            if print_timestamps:
                print(f"{format_range(word.start, word.end)} {word.text}")
            else:
                print(word.text)
        return

    # Short-form GigaAM returns the full utterance text; if word timestamps are
    # not requested, there are no per-segment timestamps, so we print the full range.
    if print_timestamps:
        print(f"{format_range(0.0, duration_seconds)} {result.text}")
    else:
        print(result.text)


def print_longform_result(
    result,
    *,
    print_timestamps: bool,
    word_timestamps: bool,
) -> None:
    if word_timestamps and result.words:
        for word in result.words:
            if print_timestamps:
                print(f"{format_range(word.start, word.end)} {word.text}")
            else:
                print(word.text)
        return

    for segment in result.segments:
        if print_timestamps:
            print(f"{format_range(segment.start, segment.end)} {segment.text}")
        else:
            print(segment.text)


def segment_audio_in_memory(
    audio_path: str,
    *,
    sr: int,
    max_duration: float,
    min_duration: float,
    strict_limit_duration: float,
    new_chunk_threshold: float,
    device: torch.device,
):
    audio = gigaam.load_audio(audio_path)
    pipeline = get_pipeline(device)
    sad_segments = pipeline(
        {
            "waveform": audio.unsqueeze(0),
            "sample_rate": sr,
        }
    )

    segments = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries = []

    def _update_segments(
        current_start: float, current_end: float, current_duration: float
    ) -> None:
        if current_duration > strict_limit_duration:
            max_segments = int(current_duration / strict_limit_duration) + 1
            segment_duration = current_duration / max_segments
            current_end = current_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(current_start * sr) : int(current_end * sr)])
                boundaries.append((current_start, current_end))
                current_start = current_end
                current_end += segment_duration
        segments.append(audio[int(current_start * sr) : int(current_end * sr)])
        boundaries.append((current_start, current_end))

    for segment in sad_segments.get_timeline().support():
        start = max(0.0, segment.start)
        end = min(audio.shape[0] / sr, segment.end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries


def transcribe_longform_in_memory(
    model,
    audio_path: str,
    *,
    word_timestamps: bool,
    **kwargs,
):
    segments, boundaries = segment_audio_in_memory(
        audio_path,
        sr=SAMPLE_RATE,
        device=model._device,
        **kwargs,
    )

    result_segments = []
    for segment, (seg_start, seg_end) in zip(segments, boundaries):
        wav = segment.to(model._device).unsqueeze(0).to(model._dtype)
        length = torch.full([1], wav.shape[-1], device=model._device)
        encoded, encoded_len = model.forward(wav, length)
        decode_result = model._decode(
            encoded, encoded_len, int(length[0].item()), word_timestamps
        )
        if isinstance(decode_result, tuple):
            text, words = decode_result
        else:
            text, words = decode_result, []

        if word_timestamps and words:
            adjusted_words = [
                gigaam.types.Word(
                    text=word.text,
                    start=round(word.start + seg_start, 3),
                    end=round(word.end + seg_start, 3),
                )
                for word in words
            ]
            result_segments.append(
                gigaam.types.Segment(
                    text=text,
                    start=seg_start,
                    end=seg_end,
                    words=adjusted_words,
                )
            )
        else:
            result_segments.append(
                gigaam.types.Segment(text=text, start=seg_start, end=seg_end)
            )

    return gigaam.types.LongformTranscriptionResult(segments=result_segments)


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        fail(f"audio file does not exist: {audio_path}")

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    try:
        duration_seconds = get_audio_duration_seconds(str(audio_path))
    except RuntimeError as exc:
        fail(
            f"failed to load audio via ffmpeg: {exc}. "
            "Make sure ffmpeg is installed and available in PATH."
        )

    use_longform = args.longform or duration_seconds > SHORT_AUDIO_LIMIT_SECONDS

    try:
        model = gigaam.load_model(
            args.model,
            fp16_encoder=args.fp16_encoder,
            use_flash=args.use_flash,
            device=args.device,
            download_root=args.download_root,
        )
    except Exception as exc:
        fail(f"failed to load model '{args.model}': {exc}")

    if not hasattr(model, "transcribe"):
        fail(
            f"model '{args.model}' is not an ASR model. "
            "Use a model with _ctc or _rnnt in the name."
        )

    try:
        if use_longform:
            result = transcribe_longform_in_memory(
                model,
                str(audio_path),
                word_timestamps=args.word_timestamps,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
                strict_limit_duration=args.strict_limit_duration,
                new_chunk_threshold=args.new_chunk_threshold,
            )
            print_longform_result(
                result,
                print_timestamps=args.print_timestamps,
                word_timestamps=args.word_timestamps,
            )
        else:
            result = model.transcribe(
                str(audio_path),
                word_timestamps=args.word_timestamps,
            )
            print_short_result(
                result,
                duration_seconds=duration_seconds,
                print_timestamps=args.print_timestamps,
                word_timestamps=args.word_timestamps,
            )

    except ModuleNotFoundError as exc:
        if use_longform:
            fail(
                "long-form mode needs extra dependencies. Install them with:\n"
                '  pip install "gigaam[longform]"'
            )
        fail(f"missing dependency: {exc}")

    except RuntimeError as exc:
        message = str(exc)
        if use_longform and "HF_TOKEN" in message:
            fail(
                "long-form mode needs a Hugging Face token with access to "
                "pyannote/segmentation-3.0. Either:\n"
                "  1) export HF_TOKEN=...\n"
                "  2) or pass --hf-token ..."
            )
        fail(message)

    except ValueError as exc:
        fail(str(exc))


if __name__ == "__main__":
    main()
