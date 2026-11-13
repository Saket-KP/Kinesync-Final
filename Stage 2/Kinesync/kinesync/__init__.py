"""KineSync: song -> shot-by-shot storyboard. This package holds Stage 2 and everything downstream."""

# On some Windows installs, importing pandas before torch makes torch's c10.dll fail to load
# (WinError 1114). Importing torch first, once, avoids it. Harmless elsewhere.
try:
    import torch  # noqa: F401
except ImportError:
    pass
