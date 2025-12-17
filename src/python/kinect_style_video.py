# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "torch",
#     "tqdm",
#     "timm",
# ]
# ///
import cv2
import torch
import numpy as np
from tqdm import tqdm

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


def load_depth_model():
    """
    Load MiDaS (DPT_Large). Note: MiDaS predicts inverse depth (disparity-like):
    closer objects typically have larger values.
    """
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(DEVICE)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform


@torch.no_grad()
def infer_depth(model, transform, frame_bgr, invert_depth: bool = False):
    """
    frame_bgr: (H, W, 3) uint8 BGR
    return: depth (H, W) float32 normalized to [0,1]
            By default: closer -> brighter (because MiDaS is inverse depth).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(DEVICE)

    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.detach().cpu().numpy().astype(np.float32)

    # Robust normalization (percentiles) so outliers don't kill contrast
    lo = np.percentile(depth, 2.0)
    hi = np.percentile(depth, 98.0)
    if hi - lo > 1e-6:
        depth = (depth - lo) / (hi - lo)
        depth = np.clip(depth, 0.0, 1.0)
    else:
        depth[:] = 0.5

    # IMPORTANT: do NOT invert by default.
    # MiDaS already behaves like inverse depth (closer tends to be larger/brighter).
    if invert_depth:
        depth = 1.0 - depth

    return depth


def make_kinect_frame(
    depth,
    point_step=4,
    jitter=1.5,
    point_size=1,
    invert_intensity=False,
    black_level=0.15,
    gamma=1.0,
    add_noise_strength=0.02,
):
    """
    depth: (H, W) float32 [0,1] (default closer -> brighter)
    Produces a dotted grayscale frame.

    black_level: pushes far/background darker (0..0.5 typical)
    gamma: >1 darkens mids, <1 brightens mids
    """
    h, w = depth.shape
    out = np.zeros((h, w), dtype=np.uint8)

    ys = np.arange(0, h, point_step)
    xs = np.arange(0, w, point_step)

    denom = max(1e-6, 1.0 - black_level)

    for y in ys:
        for x in xs:
            val = float(depth[y, x])

            if invert_intensity:
                val = 1.0 - val

            # Keep background darker, lift foreground contrast
            val = (val - black_level) / denom
            val = np.clip(val, 0.0, 1.0)

            if gamma != 1.0:
                val = val**gamma

            # Add small brightness noise
            val += np.random.randn() * add_noise_strength
            val = np.clip(val, 0.0, 1.0)
            intensity = int(val * 255)

            jx = int(round(x + np.random.uniform(-jitter, jitter)))
            jy = int(round(y + np.random.uniform(-jitter, jitter)))

            if 0 <= jx < w and 0 <= jy < h:
                if point_size <= 1:
                    if intensity > out[jy, jx]:
                        out[jy, jx] = intensity
                else:
                    half = point_size // 2
                    y0, y1 = max(0, jy - half), min(h, jy + half + 1)
                    x0, x1 = max(0, jx - half), min(w, jx + half + 1)
                    patch = out[y0:y1, x0:x1]
                    np.maximum(patch, intensity, out=patch)

    return out


def add_kinect_border_and_noise(
    frame_gray, border=20, vignette_strength=0.25, global_noise=2, border_mul=0.25
):
    """
    Add vignette + darker border + small global noise.
    """
    h, w = frame_gray.shape
    out = frame_gray.astype(np.float32)

    # Vignette
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / max(1e-6, r.max())
    vignette = 1.0 - vignette_strength * (r_norm**2)
    out *= vignette

    # Border (dark)
    if border > 0:
        out[:border, :] *= border_mul
        out[-border:, :] *= border_mul
        out[:, :border] *= border_mul
        out[:, -border:] *= border_mul

    # Global noise
    if global_noise > 0:
        out += np.random.randn(h, w).astype(np.float32) * float(global_noise)

    return np.clip(out, 0, 255).astype(np.uint8)


def process_video_to_kinect(
    input_path,
    output_path="output_kinect.mp4",
    max_frames=None,
    point_step=4,
    point_size=1,
    fps_override=None,
    invert_depth=False,
    invert_intensity=False,
    black_level=0.15,
    gamma=1.0,
    add_noise_strength=0.02,
    vignette_strength=0.25,
    global_noise=2,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_override is not None:
        fps = fps_override

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write as BGR for maximum compatibility with players/codecs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    model, transform = load_depth_model()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    pbar = tqdm(total=total_frames, desc="Processing")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break

        depth = infer_depth(model, transform, frame, invert_depth=invert_depth)

        depth_frame = make_kinect_frame(
            depth,
            point_step=point_step,
            point_size=point_size,
            jitter=1.5,
            invert_intensity=invert_intensity,
            black_level=black_level,
            gamma=gamma,
            add_noise_strength=add_noise_strength,
        )

        kinect_frame = add_kinect_border_and_noise(
            depth_frame,
            border=20,
            vignette_strength=vignette_strength,
            global_noise=global_noise,
            border_mul=0.25,
        )

        out.write(cv2.cvtColor(kinect_frame, cv2.COLOR_GRAY2BGR))

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert normal video to Xbox Kinect-style depth video"
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument(
        "--output", "-o", default="output_kinect.mp4", help="Output video path"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit number of frames"
    )
    parser.add_argument("--step", type=int, default=4, help="Point step (grid step)")
    parser.add_argument(
        "--point-size", type=int, default=1, help="Point size in pixels (1-3)"
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Override FPS of output video"
    )

    # New controls (defaults chosen so subject is brighter, background darker)
    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Invert normalized depth (flip near/far)",
    )
    parser.add_argument(
        "--invert-intensity",
        action="store_true",
        help="Invert point brightness mapping",
    )
    parser.add_argument(
        "--black-level",
        type=float,
        default=0.15,
        help="Depth cutoff mapped to black (0..1)",
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Gamma for intensity mapping"
    )
    parser.add_argument(
        "--point-noise",
        type=float,
        default=0.02,
        help="Per-point brightness noise strength",
    )
    parser.add_argument(
        "--vignette",
        type=float,
        default=0.25,
        help="Vignette strength",
    )
    parser.add_argument(
        "--global-noise",
        type=float,
        default=2.0,
        help="Global frame noise (stddev)",
    )

    args = parser.parse_args()

    process_video_to_kinect(
        args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        point_step=args.step,
        point_size=args.point_size,
        fps_override=args.fps,
        invert_depth=args.invert_depth,
        invert_intensity=args.invert_intensity,
        black_level=args.black_level,
        gamma=args.gamma,
        add_noise_strength=args.point_noise,
        vignette_strength=args.vignette,
        global_noise=int(round(args.global_noise)),
    )
