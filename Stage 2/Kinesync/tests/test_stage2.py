"""Stage 2 on mock data: contract shapes, padding masks, one training step, sampling
determinism, overfit (slow), end to end with placeholder panels, renderer smoke test (GPU)."""
import csv
import json
import os

import numpy as np
import pytest
import torch

from conftest import REAL_ROOT


def test_dataset_contract(mock_root):
    from kinesync.common import load_vocab, videos_in
    from kinesync.stage2.dataset import ShotDataset, collate
    from kinesync.stage2.train import load_records
    vocab = load_vocab()
    recs = load_records(videos_in("train"), vocab)
    ds = ShotDataset(recs, max_len=16, train=True, kin_noise=0.3, jitter_frames=2)
    it = ds[0]
    n = len(it["scale"])
    assert 1 <= n <= 16
    dims = {"clap": 512, "lyric": 384, "kin": 4, "tags": 20, "prev_tags": 20}
    for k, d in dims.items():
        assert it[k].shape == (n, d), k
    for k in ("has_lyric", "log_dur", "pos", "section", "genre", "scale", "moving", "tag_mask", "has_prev"):
        assert it[k].shape == (n,), k
    assert it["section"].max() < 8 and it["section"].min() >= 0
    assert set(np.unique(it["has_lyric"])) <= {0.0, 1.0}
    # chunks of max_len, and the first chunk of a song has no previous shot
    assert it["has_prev"][0] == 0 and (it["has_prev"][1:] == 1).all()
    second = [i for i, (r, a, b) in enumerate(ds.index) if a > 0][0]
    assert ds[second]["has_prev"][0] == 1                 # later chunks carry the true previous shot
    # padding: two chunks of different length
    short = [i for i, (r, a, b) in enumerate(ds.index) if b - a < 16][0]
    b = collate([ds[0], ds[short]])
    L = b["mask"].shape[1]
    assert b["mask"][0].sum() == len(ds[0]["scale"])
    assert b["mask"][1].sum() == len(ds[short]["scale"]) < L
    assert not b["mask"][1, -1]
    assert (b["clap"][1, ~b["mask"][1]] == 0).all()


def test_one_step_and_param_count(mock_root, tiny_cfg):
    from kinesync.common import load_vocab, videos_in
    from kinesync.stage2.dataset import ShotDataset, collate
    from kinesync.stage2.model import Stage2Model, stage2_loss
    from kinesync.stage2.train import load_records
    vocab = load_vocab()
    full = Stage2Model(vocab, 5)                          # the real size
    print("Stage 2 parameters:", full.n_params())
    assert 2_000_000 < full.n_params() < 8_000_000
    m = Stage2Model(vocab, 5, d=64, layers=2, heads=2, ff=128)
    b = collate([ShotDataset(load_records(videos_in("train")[:2], vocab), 32)[i] for i in range(2)])
    out = m(b)
    assert out["scale"].shape[-1] == 5 and out["tag_setting"].shape[-1] == 6 and out["tag_subject"].shape[-1] == 4
    terms = stage2_loss(out, b, vocab)
    terms["total"].backward()
    assert set(terms) >= {"scale", "moving", "tag_setting", "tag_style", "total"}
    assert all(torch.isfinite(v) for v in terms.values())


def test_causal_mask_no_future_leak(mock_root):
    from kinesync.common import load_vocab, videos_in
    from kinesync.stage2.dataset import ShotDataset, collate
    from kinesync.stage2.model import Stage2Model
    from kinesync.stage2.train import load_records
    vocab = load_vocab()
    m = Stage2Model(vocab, 5, d=64, layers=2, heads=2, ff=128, dropout=0.0).eval()
    b = collate([ShotDataset(load_records(videos_in("train")[:1], vocab), 20)[0]])
    with torch.no_grad():
        o1 = m(b)["scale"][0, :5]
        b2 = {k: v.clone() for k, v in b.items()}
        b2["clap"][0, 10:] += 5.0                         # change only the future
        o2 = m(b2)["scale"][0, :5]
    assert torch.allclose(o1, o2, atol=1e-5)


def _tiny_trained(mock_root, tiny_cfg):
    from kinesync.stage2.train import train
    res = train(dict(tiny_cfg, run_name="tiny"), steps=10, quiet=True, root=mock_root)
    return os.path.join(res["out_dir"], "best.pt")


def test_sampling_deterministic(mock_root, tiny_cfg):
    from kinesync.common import videos_in
    from kinesync.stage1.mock import write_mock
    from kinesync.stage2.sample import storyboard
    from kinesync.stage2.train import load_checkpoint
    model, ck = load_checkpoint(_tiny_trained(mock_root, tiny_cfg))
    vid = videos_in("test")[0]
    write_mock(vid, 0)
    a = storyboard(model, ck, vid, 0, seed=7)
    b = storyboard(model, ck, vid, 0, seed=7)
    c = storyboard(model, ck, vid, 0, seed=8)
    assert a == b
    assert a["shots"] and a != c
    s = a["shots"][0]
    assert set(s) >= {"start", "end", "section", "scale", "moving", "tags", "lyric", "kin"}
    assert set(s["tags"]) == {"setting", "subject", "action", "lighting", "style"}
    assert all(len(sh["tags"]["setting"]) <= 1 and len(sh["tags"]["subject"]) <= 2 for sh in a["shots"])


@pytest.mark.slow
def test_overfit_5(tiny_cfg):
    """--overfit 5 on the REAL data (milestone 2) must get the weighted train loss under 0.6."""
    if not (REAL_ROOT / "features").exists():
        pytest.skip("real data release not present")
    from kinesync.common import load_yaml
    from kinesync.stage2.train import train
    cfg = load_yaml(REAL_ROOT / "configs" / "stage2" / "run_A.yaml")
    res = train(cfg, overfit=5, steps=400, quiet=True, root=REAL_ROOT, device="cpu")
    assert res["final_train"] < 0.6, res["final_train"]


def test_real_timing_uses_real_shots_and_new_seeds(mock_root, tiny_cfg):
    """timing='real' copies the song's real cuts; samples then differ only by their Stage 2 seed."""
    from kinesync.common import load_shots, videos_in
    from kinesync.stage2.sample import storyboard
    from kinesync.stage2.train import load_checkpoint
    model, ck = load_checkpoint(_tiny_trained(mock_root, tiny_cfg))
    vid = videos_in("test")[0]
    real = load_shots(vid)
    a = storyboard(model, ck, vid, 0, seed=7, timing="real")
    b = storyboard(model, ck, vid, 1, seed=7, timing="real")
    assert [s["start"] for s in a["shots"]] == [round(s["start"], 4) for s in real]
    assert a["sample_id"] == "real0" and b["sample_id"] == "real1" and not a["stage1_mock"]
    assert a["stage2_seed"] != b["stage2_seed"]
    assert [s["start"] for s in a["shots"]] == [s["start"] for s in b["shots"]]


def test_end_to_end_placeholder(mock_root, tiny_cfg):
    from kinesync.common import videos_in
    from kinesync.pipeline import run_song
    ck = _tiny_trained(mock_root, tiny_cfg)
    vid = videos_in("test")[0]
    res = run_song(vid, 2, mock_root / "outputs", ck, placeholder=True, regen_stage1=True, root=mock_root)
    assert len(res) == 2
    for r in res:
        sb = json.load(open(r["storyboard"], encoding="utf-8"))
        rows = list(csv.DictReader(open(r["csv"], encoding="utf-8")))
        assert len(rows) == len(sb["shots"]) > 0
        assert rows[0]["start_tc"] == "00:00:00:00"
        html = open(r["html"], encoding="utf-8").read()
        assert html.count("data:image/jpeg;base64") == len(sb["shots"])
        assert "MOCK Stage 1" in html
        assert open(r["edl"]).read().startswith("TITLE:")


def _storyboard(n=3):
    tags = {"setting": [], "subject": ["solo singer"], "action": [], "lighting": [], "style": []}
    shot = {"start": 0.0, "end": 2.0, "section": "segment 0", "scale": "close-up", "moving": False, "tags": tags,
            "lyric": None, "kin": {"flow": 1.0, "color_drift": 0.1, "brightness": 0.3, "saturation": 0.4}, "short": False}
    return {"video_id": "v0000", "stage1_sample": 0, "stage2_seed": 1, "shots": [dict(shot) for _ in range(n)]}


def test_render_resumes_and_notices_changes(mock_root, tmp_path):
    """A rerun reuses finished panels; changing what decides a panel's look redraws it."""
    from kinesync.common import load_yaml
    from kinesync.render.panels import render_storyboard
    cfg = load_yaml(REAL_ROOT / "configs" / "render.yaml")
    sb = _storyboard()
    first = render_storyboard(sb, tmp_path, cfg, placeholder=True, root=mock_root)
    assert (first["panels_drawn"], first["panels_reused"]) == (3, 0)
    again = render_storyboard(sb, tmp_path, cfg, placeholder=True, root=mock_root)
    assert (again["panels_drawn"], again["panels_reused"]) == (0, 3)
    cfg["panels"]["steps"] = 4
    changed = render_storyboard(sb, tmp_path, cfg, placeholder=True, root=mock_root)
    assert changed["panels_drawn"] == 3
    (tmp_path / "panels" / "panel_002.png").unlink()      # a missing file is redrawn too
    partial = render_storyboard(sb, tmp_path, cfg, placeholder=True, root=mock_root)
    assert (partial["panels_drawn"], partial["panels_reused"]) == (1, 2)


def test_max_shots_draws_only_the_first_panels(mock_root, tmp_path):
    from kinesync.common import load_yaml
    from kinesync.render.panels import render_storyboard
    log = render_storyboard(_storyboard(5), tmp_path, load_yaml(REAL_ROOT / "configs" / "render.yaml"),
                            placeholder=True, root=mock_root, max_shots=2)
    assert log["n_panels"] == 2 and log["n_shots"] == 5
    assert len(list((tmp_path / "panels").glob("panel_*.png"))) == 2
    assert len((tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()) == 5


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a GPU")
def test_no_gpu_refuses_instead_of_silent_placeholders(mock_root, tmp_path):
    from kinesync.common import load_yaml
    from kinesync.render.panels import render_storyboard
    with pytest.raises(RuntimeError, match="placeholder"):
        render_storyboard(_storyboard(), tmp_path, load_yaml(REAL_ROOT / "configs" / "render.yaml"), root=mock_root)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_renderer_smoke():
    """Handoff section 9: 1 step on 2 shots. Also checks seeds reproduce and panels are grey."""
    import numpy as np
    from kinesync.common import load_yaml
    from kinesync.render.panels import SDXLTurbo
    cfg = dict(load_yaml(REAL_ROOT / "configs" / "render.yaml")["panels"], steps=1)
    r = SDXLTurbo(cfg)
    prompts = ["Close-up, static shot: solo singer, even muted palette.",
               "Long shot, moving shot: in empty desert road, even muted palette."]
    a = r(prompts, [1, 2])
    b = r(prompts[:1], [1])
    r.close()
    assert len(a) == 2 and a[0].size == (512, 512)
    arr = np.asarray(a[0]).astype(int)
    assert arr.std() > 5, "blank panel: the VAE probably produced NaNs"
    assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all(), "panel is not grey"
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])), "same seed must give the same panel"


def test_cross_model_shapes_causality_and_memory(mock_root):
    """Cross-attention model: same outputs as the base model, future shots cannot leak backwards,
    and the song memory really is used."""
    from kinesync.common import load_vocab, videos_in
    from kinesync.stage2.dataset import ShotDataset, collate
    from kinesync.stage2.model import Stage2CrossModel
    from kinesync.stage2.train import load_records
    vocab = load_vocab()
    torch.manual_seed(0)
    m = Stage2CrossModel(vocab, 5, d=64, layers=2, heads=2, ff=128, dropout=0.0).eval()
    b = collate([ShotDataset(load_records(videos_in("train")[:2], vocab), 20)[i] for i in range(2)])
    assert b["mem_clap"].shape[2] == 512 and b["mem_lyric"].shape[2] == 384
    with torch.no_grad():
        o = m(b)
        assert o["scale"].shape[:2] == b["mask"].shape and o["tag_setting"].shape[-1] == 6
        b2 = {k: v.clone() for k, v in b.items()}
        b2["clap"][:, 10:] += 5.0                       # change only future shots
        assert torch.allclose(m(b2)["scale"][:, :5], o["scale"][:, :5], atol=1e-5)
        b3 = {k: v.clone() for k, v in b.items()}
        b3["mem_clap"] = b3["mem_clap"] + 1.0           # change only the song memory
        assert not torch.allclose(m(b3)["scale"], o["scale"], atol=1e-4)
        b4 = {k: v.clone() for k, v in b.items()}
        b4["mem_clap"][b4["mem_clap_valid"] == 0] = 99.0  # padding in the memory must be ignored
        assert torch.allclose(m(b4)["scale"], o["scale"], atol=1e-5)


def test_cross_model_trains_and_samples_deterministically(mock_root, tiny_cfg):
    import copy
    from kinesync.common import videos_in
    from kinesync.stage1.mock import write_mock
    from kinesync.stage2.sample import storyboard
    from kinesync.stage2.train import load_checkpoint, train
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["type"] = "cross"
    res = train(dict(cfg, run_name="tiny_cross"), steps=10, quiet=True, root=mock_root)
    model, ck = load_checkpoint(os.path.join(res["out_dir"], "best.pt"))
    assert type(model).__name__ == "Stage2CrossModel"
    vid = videos_in("test")[0]
    write_mock(vid, 0)
    assert storyboard(model, ck, vid, 0, seed=3) == storyboard(model, ck, vid, 0, seed=3)
