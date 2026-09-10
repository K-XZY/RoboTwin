import cv2
import numpy as np
import os
import subprocess
import pickle
import pdb

# Rocky 9 (and RHEL generally) ships ffmpeg built without libx264 -- the encoder list
# has libopenh264, h264_nvenc and the hardware ones, but not libx264. Hardcoding
# libx264 makes ffmpeg exit immediately and the write to its stdin fails with
# BrokenPipeError, which reads like a pipe bug rather than a missing codec.
_H264_ENCODERS = ("libx264", "libopenh264", "h264_nvenc")
_ENCODER = None


def _pick_h264_encoder():
    """First H.264 encoder this ffmpeg build actually has."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        listing = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        listing = ""
    for name in _H264_ENCODERS:
        if f" {name} " in listing:
            _ENCODER = name
            break
    else:
        _ENCODER = _H264_ENCODERS[0]
    return _ENCODER


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            _pick_h264_encoder(),
            f"{out_path}",
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )
