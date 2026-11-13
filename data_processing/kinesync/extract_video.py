import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from kinesync.constants import MIN_SCENE_LEN_FRAMES, MIN_SHOT_S, POOL, STATS_FPS

def detect_cut_times(video_path: str, duration: float) -> list:
    """Detect exact scene cuts on native-fps video and filter out shots < 0.2s."""
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=MIN_SCENE_LEN_FRAMES))
    sm.detect_scenes(video)
    
    raw_times = [float(s.get_seconds()) for s, _ in sm.get_scene_list() if s.get_seconds() > 0.0]
    clean, last = [], 0.0
    for t in sorted(raw_times):
        if t - last >= MIN_SHOT_S and duration - t >= MIN_SHOT_S:
            clean.append(t)
            last = t
    return clean

def _find_crop_box(frames_gray: list) -> tuple:
    """Detect and remove static black letterbox bars (intensity <= 12)."""
    gray_mean = np.mean(frames_gray, axis=0)
    r_idx = np.where(gray_mean.mean(axis=1) > 12.0)[0]
    c_idx = np.where(gray_mean.mean(axis=0) > 12.0)[0]
    return int(r_idx[0]), int(r_idx[-1]) + 1, int(c_idx[0]), int(c_idx[-1]) + 1

def _pool(arr10: np.ndarray, T2: int):
    """Pool 5 consecutive 10 fps frames into one 2 fps bin, ignoring NaNs at scene cuts."""
    out, valid = np.zeros(T2, dtype=np.float32), np.zeros(T2, dtype=np.uint8)
    prev = 0.0
    for j in range(T2):
        seg = arr10[j * POOL : (j + 1) * POOL]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            prev = float(seg.mean())
            valid[j] = 1
        out[j] = prev
    return out, valid

def compute_video_features(frames_bgr: list, cut_times: list, T10: int):
    """Compute HSV brightness/saturation, LAB Bhattacharyya colour drift, and Farneback optical flow."""
    T2 = T10 // POOL
    cut_frames = {int(round(t * STATS_FPS)) for t in cut_times}
    y0, y1, x0, x1 = _find_crop_box([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr[::20]])

    flow10, drift10, bright10, sat10 = [], [], [], []
    prev_gray, prev_hist = None, None

    for i, frame in enumerate(frames_bgr[:T10]):
        crop = frame[y0:y1, x0:x1]
        
        # 1. Brightness & Saturation from HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat10.append(float(np.mean(hsv[:, :, 1]) / 255.0))
        bright10.append(float(np.mean(hsv[:, :, 2]) / 255.0))

        # 2. Colour Drift via 3D LAB Histogram Bhattacharyya distance
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hist = cv2.normalize(cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256]*3), None)
        drift = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)) if prev_hist is not None else np.nan
        prev_hist = hist

        # 3. Dense Optical Flow magnitude via Farneback
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = float(np.mean(np.hypot(fl[..., 0], fl[..., 1])))
        else:
            flow = np.nan
        prev_gray = gray

        # Mask out artificial spikes caused by scene cuts
        if i in cut_frames:
            flow, drift = np.nan, np.nan
        flow10.append(flow)
        drift10.append(drift)

    return _pool(np.array(flow10), T2), _pool(np.array(drift10), T2), _pool(np.array(bright10), T2), _pool(np.array(sat10), T2)