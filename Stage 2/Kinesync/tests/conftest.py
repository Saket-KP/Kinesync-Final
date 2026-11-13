import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kinesync  # noqa: E402,F401  (imports torch before pandas; see kinesync/__init__.py)
from make_mock import make_mock  # noqa: E402

REAL_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (overfit)")


@pytest.fixture(scope="session")
def mock_root(tmp_path_factory):
    """A small fake data release; KINESYNC_ROOT points at it for the whole session."""
    root = tmp_path_factory.mktemp("mock_release")
    make_mock(root)
    old = os.environ.get("KINESYNC_ROOT")
    os.environ["KINESYNC_ROOT"] = str(root)
    yield root
    if old is None:
        os.environ.pop("KINESYNC_ROOT", None)
    else:
        os.environ["KINESYNC_ROOT"] = old


@pytest.fixture(scope="session")
def tiny_cfg():
    return {"run_name": "test", "seed": 0, "kin_source": "real", "kin_noise": 0.0, "jitter_frames": 0,
            "max_len": 128, "model": {"d": 64, "layers": 2, "heads": 2, "ff": 128, "dropout": 0.0},
            "optim": {"lr": 3e-3, "weight_decay": 0.0, "warmup_steps": 5, "batch_size": 4, "grad_clip": 1.0,
                      "max_epochs": 2, "patience": 5}, "amp": "off"}
