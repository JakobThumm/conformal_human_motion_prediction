"""Shared `--prepare` back-end for shots s03 (2D pose + covariance) and s04 (triangulation).

Runs the repo's *deployed* perception front-end on real H36M stereo frames:

    ultralytics (custom fork, YOLO26 `Pose26` head with per-keypoint sigma)
      -> pose_estimation.inference_helper_batched.process_frame_2d_yolo   (17 kpts -> 13 joints,
                                                                           mirror map, sigma_x/sigma_y)
      -> utils.batched_transform_torch.create_joint_covariance_batched     (2x 2x2 -> 4x4)
      -> utils.batched_transform_torch.triangulate_points_with_covariance_batched
                                                                           (linear triangulation +
                                                                            first-order covariance
                                                                            propagation)

No math is re-implemented here -- this module only selects frames, calls those functions and
caches the result.  Requires the repo venv (`.venv/bin/python`) and a GPU.

Only imported by `video/shots/s03_pose2d.py --prepare` and `video/shots/s04_triangulate.py
--prepare`; the render paths never touch it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- the clip we cache
# H36M validation subject S11, action "Greeting", the two front cameras the paper triangulates
# with.  Frames 1580..1691 (50 fps) -- the subject waves with both arms raised, fully visible in
# both views, single person, no occlusion.  (frames 1524..1607)
SUBJECT = "S11"
ACTION = "Greeting"
CAMERAS = ("55011271", "60457274")
START_FRAME = 1524
N_FRAMES = 84
SRC_FPS = 50.0

CAMERA_PARAMS = "models/pose_estimation/camera-parameters.json"
YOLO_WEIGHTS = "yolo26n-pose.pt"          # repo root; the Pose26 fork checkpoint


def _read_clip(video_path: Path, start: int, n: int) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        # the two H36M cameras differ by 2 px in height -- crop both to 1000x1000
        out.append(cv2.cvtColor(bgr[:1000, :1000], cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) != n:
        raise RuntimeError(f"only got {len(out)}/{n} frames from {video_path}")
    return out


def run() -> dict:
    """Run the front-end on the cached clip.  Returns plain numpy arrays."""
    import torch
    sys.path.insert(0, str(REPO / "src"))
    from ultralytics import YOLO
    from conformal_human_motion_prediction.pose_estimation.h36m_settings import (
        MIRROR_13_JOINT_MODEL_MAP,
    )
    from conformal_human_motion_prediction.pose_estimation.inference_helper_batched import (
        process_frame_2d_yolo,
    )
    from conformal_human_motion_prediction.pose_estimation.triangulation_helper import (
        load_camera_parameters,
    )
    from conformal_human_motion_prediction.utils.batched_transform_torch import (
        create_joint_covariance_batched,
        triangulate_points_with_covariance_batched,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(REPO / YOLO_WEIGHTS))
    model.to(device)

    vids = [REPO / "datasets/H36M/extracted" / SUBJECT / "Videos" / f"{ACTION}.{c}.mp4"
            for c in CAMERAS]
    clips = [_read_clip(v, START_FRAME, N_FRAMES) for v in vids]

    kp = np.zeros((2, N_FRAMES, 13, 2), np.float64)
    cov = np.zeros((2, N_FRAMES, 13, 2, 2), np.float64)
    conf = np.zeros((2, N_FRAMES, 13), np.float64)
    for c in range(2):
        for i0 in range(0, N_FRAMES, 8):
            batch = np.stack(clips[c][i0:i0 + 8]).astype(np.float32)
            o = process_frame_2d_yolo(
                frames=batch, yolo_pose_model=model, mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=False, verbose=False, device=device,
            )
            n = batch.shape[0]
            kp[c, i0:i0 + n] = o["keypoints"].detach().cpu().numpy()
            cov[c, i0:i0 + n] = o["covariance_matrix"].detach().cpu().numpy()
            conf[c, i0:i0 + n] = o["confidence"].detach().cpu().numpy()
            if not bool(o["mask"].all()):
                raise RuntimeError(f"no detection in camera {c} batch {i0}")

    # ---- triangulation + first-order covariance propagation (repo functions, deployed path)
    intr, extr, proj = load_camera_parameters(
        str(REPO / CAMERA_PARAMS), SUBJECT, list(CAMERAS))
    P1 = torch.from_numpy(proj[CAMERAS[0]]).to(device)
    P2 = torch.from_numpy(proj[CAMERAS[1]]).to(device)

    sig = np.sqrt(np.stack([cov[:, :, :, 0, 0], cov[:, :, :, 1, 1]], -1))  # (2,N,13,2)
    t = lambda a: torch.from_numpy(a).to(device)                          # noqa: E731
    C4 = create_joint_covariance_batched(
        mapped_uncertainty_cam1=t(sig[0]), mapped_covariance_cam1=t(cov[0, :, :, 0, 1]),
        mapped_uncertainty_cam2=t(sig[1]), mapped_covariance_cam2=t(cov[1, :, :, 0, 1]),
        cross_covariance=torch.zeros((N_FRAMES, 13, 2, 2), dtype=torch.float64, device=device),
    )
    p3, C3 = triangulate_points_with_covariance_batched(t(kp[0]), t(kp[1]), P1, P2, C4)
    p3 = p3.detach().cpu().numpy()
    C3 = C3.detach().cpu().numpy()

    cam_centers = np.stack([
        -np.asarray(extr[c])[:, :3].T @ np.asarray(extr[c])[:, 3] for c in CAMERAS])

    return dict(frames=clips, kp=kp, cov=cov, conf=conf, p3=p3, C3=C3,
                cam_centers=cam_centers,
                K=np.stack([np.asarray(intr[c]) for c in CAMERAS]),
                P=np.stack([np.asarray(proj[c]) for c in CAMERAS]))


# ------------------------------------------------------------------ crop bookkeeping

def crop_rects(kp: np.ndarray, cov: np.ndarray, aspect: float = 0.8,
               pad: float = 14.0, img: int = 1000) -> np.ndarray:
    """Fixed per-camera crop rect (x0, y0, w, h) covering the whole clip's skeleton + 2 sigma."""
    rects = []
    for c in range(kp.shape[0]):
        s = 2.0 * np.sqrt(np.stack([cov[c, :, :, 0, 0], cov[c, :, :, 1, 1]], -1))
        lo = (kp[c] - s).reshape(-1, 2).min(0) - pad
        hi = (kp[c] + s).reshape(-1, 2).max(0) + pad
        cx, cy = (lo + hi) / 2
        w, h = hi - lo
        if w / h > aspect:
            h = w / aspect
        else:
            w = h * aspect
        x0, y0 = cx - w / 2, cy - h / 2
        x0 = float(np.clip(x0, 0, img - w)) if w <= img else 0.0
        y0 = float(np.clip(y0, 0, img - h)) if h <= img else 0.0
        rects.append([x0, y0, min(w, img), min(h, img)])
    return np.asarray(rects, np.float64)


def encode_jpegs(frames: list[np.ndarray], rect: np.ndarray, scale: float = 1.0,
                 quality: int = 86) -> tuple[list[bytes], tuple[int, int]]:
    """Crop `rect` out of every RGB frame, optionally rescale, JPEG-encode."""
    import cv2
    x0, y0, w, h = [int(round(v)) for v in rect]
    ow, oh = int(round(w * scale)), int(round(h * scale))
    out = []
    for f in frames:
        c = f[y0:y0 + h, x0:x0 + w]
        if (ow, oh) != (w, h):
            c = cv2.resize(c, (ow, oh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", c[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        out.append(buf.tobytes())
    return out, (ow, oh)
