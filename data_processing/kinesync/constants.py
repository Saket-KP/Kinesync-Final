import math

SR = 22050                 # Audio sample rate (22.05 kHz)
AUDIO_FPS = 10             # 10 Hz grid for audio features
HOP = SR // AUDIO_FPS      # 2205 samples per hop = exact 0.1s (10 fps) steps
VIDEO_FPS = 2              # 2 Hz grid saved in .npz
STATS_FPS = 10             # Internal video decode rate before 5:1 pooling
POOL = STATS_FPS // VIDEO_FPS   # Pool 5 frames (0.5s) into 1 video feature frame
CLAP_SR = 48000            # CLAP requires 48 kHz audio

MIN_SCENE_LEN_FRAMES = 6   # Minimum frames per scene for PySceneDetect
MIN_SHOT_S = 0.2           # Minimum allowed shot duration in seconds

def n_audio(duration_s: float) -> int:
    return int(math.floor(float(duration_s) * AUDIO_FPS))  # T10 = floor(10 * D)

def n_video(duration_s: float) -> int:
    return n_audio(duration_s) // POOL                     # T2 = T10 // 5

def n_clap(duration_s: float) -> int:
    return max(1, n_audio(duration_s) // AUDIO_FPS)        # S = T10 // 10 (1 Hz)