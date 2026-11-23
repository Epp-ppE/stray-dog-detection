"""Frame sampling: decode a clip at a fixed rate and downscale, via an ffmpeg pipe.

HEVC 2560x1440 is expensive to decode; ffmpeg's fps filter drops frames before
scaling so only the kept frames are converted. Frame i has timestamp i / fps.
"""
import subprocess
import numpy as np


def iter_frames(path, fps=3, width=1280, height=720):
    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-an",
           "-vf", f"fps={fps},scale={width}:{height}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = width * height * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i, i / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()
