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
import os
import cv2
import torch
# import torchvision.transforms as T
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
    Загружаем MiDaS (DPT_Large) из torchvision.
    Это одна из лучших открытых моделей глубины.
    """
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(DEVICE)
    model.eval()

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    return model, transform


@torch.no_grad()
def infer_depth(model, transform, frame_bgr):
    """
    frame_bgr: numpy (H, W, 3), BGR (OpenCV)
    return: depth (H, W) float32, нормированная в [0, 1]
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

    depth = prediction.cpu().numpy().astype(np.float32)

    # Нормализация в [0, 1]
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-8:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth[:] = 0.5

    # Инвертируем, чтобы ближе к камере было ярче (как у Kinect)
    depth = 1.0 - depth
    return depth


def make_kinect_frame(
    depth, point_step=4, jitter=1.5, point_size=1, invert=False, add_noise_strength=0.05
):
    """
    depth: (H, W) float32 [0,1]
    point_step: шаг сетки по пикселям (чем больше, тем реже точки)
    jitter: разброс координат точки (в пикселях)
    point_size: размер точек (1..3)
    invert: инвертировать яркость (по вкусу)
    add_noise_strength: случайный шум яркости

    return: gray_frame (H, W) uint8
    """
    h, w = depth.shape
    out = np.zeros((h, w), dtype=np.uint8)

    ys = np.arange(0, h, point_step)
    xs = np.arange(0, w, point_step)

    for y in ys:
        for x in xs:
            d = depth[y, x]
            if invert:
                val = 1.0 - d
            else:
                val = d

            # Добавляем шум по яркости
            val += np.random.randn() * add_noise_strength
            val = np.clip(val, 0.0, 1.0)
            intensity = int(val * 255)

            # Джиттер координат (мелкая «дрожь» точек)
            jx = int(round(x + np.random.uniform(-jitter, jitter)))
            jy = int(round(y + np.random.uniform(-jitter, jitter)))

            if 0 <= jx < w and 0 <= jy < h:
                if point_size <= 1:
                    out[jy, jx] = max(out[jy, jx], intensity)
                else:
                    half = point_size // 2
                    y0, y1 = max(0, jy - half), min(h, jy + half + 1)
                    x0, x1 = max(0, jx - half), min(w, jx + half + 1)
                    patch = out[y0:y1, x0:x1]
                    np.maximum(patch, intensity, out=patch)

    return out


def add_kinect_border_and_noise(
    frame_gray, border=20, vignette_strength=0.2, global_noise=3
):
    """
    Добавляем рамку + лёгкую виньетку + глобальный шум.
    frame_gray: (H, W) uint8
    """
    h, w = frame_gray.shape
    out = frame_gray.astype(np.float32)

    # Виньетка
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    vignette = 1.0 - vignette_strength * (r_norm**2)
    out *= vignette

    # Рамка (тёмная)
    if border > 0:
        out[:border, :] *= 0.2
        out[-border:, :] *= 0.2
        out[:, :border] *= 0.2
        out[:, -border:] *= 0.2

    # Глобальный шум
    if global_noise > 0:
        noise = np.random.randn(h, w).astype(np.float32) * global_noise
        out += noise

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def process_video_to_kinect(
    input_path,
    output_path="output_kinect.mp4",
    max_frames=None,
    point_step=4,
    point_size=1,
    fps_override=None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_override is not None:
        fps = fps_override

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)

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

        depth = infer_depth(model, transform, frame)
        depth_frame = make_kinect_frame(
            depth,
            point_step=point_step,
            point_size=point_size,
            jitter=1.5,
            invert=False,
            add_noise_strength=0.05,
        )
        kinect_frame = add_kinect_border_and_noise(
            depth_frame, border=20, vignette_strength=0.25, global_noise=3
        )

        out.write(kinect_frame)
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
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Point step (grid step, the higher the sparser the cloud)",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=1,
        help="Point size in pixels (1-3 gives reasonable look)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS of output video",
    )
    args = parser.parse_args()

    process_video_to_kinect(
        args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        point_step=args.step,
        point_size=args.point_size,
        fps_override=args.fps,
    )
