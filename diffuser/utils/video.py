import os
import cv2
import numpy as np

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, fps=60, video_format='mp4'):
    """Save a stack of frames (N, H, W, C) to an mp4 file."""
    _make_dir(filename)
    h, w = video_frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for frame in video_frames:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def save_videos(filename, *video_frames, axis=1, **kwargs):
    """Concatenate multiple videos side-by-side and save."""
    video_frames = np.concatenate(video_frames, axis=axis)
    save_video(filename, video_frames, **kwargs)
