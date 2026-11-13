import cv2
import numpy as np
import torch
from PIL import Image
from kinesync.calibration import label_from_sims

def _estimate_camera_motion(frames_gray: list) -> float:
    """Isolate global camera translation + zoom using Lucas-Kanade tracking and RANSAC Affine fit."""
    motions = []
    step = max(1, (len(frames_gray) - 1) // 15)
    for i in range(0, len(frames_gray) - 1, step):
        pts0 = cv2.goodFeaturesToTrack(frames_gray[i], maxCorners=80, qualityLevel=0.01, minDistance=7)
        if pts0 is None or len(pts0) < 6:
            continue
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(frames_gray[i], frames_gray[i + 1], pts0, None)
        good0, good1 = pts0[st == 1], pts1[st == 1]
        if len(good0) >= 6:
            # RANSAC separates global camera movement (inliers) from actor movement (outliers)
            M, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC)
            if M is not None:
                dx, dy = M[0, 2], M[1, 2]
                zoom = abs(np.hypot(M[0, 0], M[1, 0]) - 1.0) * 100.0
                motions.append(float(np.hypot(dx, dy) + zoom))
    return float(np.median(motions)) if motions else 0.0

def label_single_shot(frames: list, start_s: float, end_s: float, model, preprocess, scale_emb, tag_emb, calib, groups):
    """Extract camera motion and CLIP zero-shot scale/tag labels for one shot."""
    dur_s = end_s - start_s
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    cam_motion = _estimate_camera_motion(gray)

    # Sample 1 keyframe (middle) if <=3s, else 3 keyframes (25%, 50%, 75%)
    n_f = len(frames)
    kf = [n_f // 2] if dur_s <= 3.0 else [n_f // 4, n_f // 2, (3 * n_f) // 4]
    imgs = [preprocess(Image.fromarray(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB))) for k in kf]

    with torch.no_grad():
        emb = model.encode_image(torch.stack(imgs))
        emb /= emb.norm(dim=-1, keepdim=True)
        mean_emb = emb.mean(dim=0, keepdim=True)
        mean_emb /= mean_emb.norm(dim=-1, keepdim=True)

        scale_sim = (mean_emb @ scale_emb.T).squeeze(0).cpu().numpy()
        tag_sim = (mean_emb @ tag_emb.T).squeeze(0).cpu().numpy()

    scale, conf, moving, tags, scores = label_from_sims(calib, groups, scale_sim, tag_sim, cam_motion, dur_s)
    return {"start": round(start_s, 4), "end": round(end_s, 4), "scale": scale,
            "scale_conf": conf, "moving": moving, "camera_motion": round(cam_motion, 4),
            "tags": tags, "tag_scores": scores}