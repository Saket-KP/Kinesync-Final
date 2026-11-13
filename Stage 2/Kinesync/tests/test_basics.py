"""Fast unit tests: vocab mapping, timecodes, metrics, template prompts, validator."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from conftest import REAL_ROOT
from kinesync.common import Vocab, load_yaml
from kinesync.eval.metrics import AveragePrecision, macro_f1
from kinesync.output.timecode import to_tc
from kinesync.render.prompt import allowed_words, template_prompt, validate, _tokens


@pytest.fixture
def vocab():
    cfg = load_yaml(REAL_ROOT / "configs" / "vocab_groups.yaml")
    return Vocab(REAL_ROOT / "vocab" / "tags_v1.json", cfg["groups"])


def test_tag_index_roundtrip(vocab):
    assert vocab.n_tags == 20
    for i in range(vocab.n_tags):
        g, j = vocab.to_local(i)
        assert vocab.to_global(g, j) == i
        assert vocab.id_of(vocab.phrase(i)) == i
    assert sum(g.n for g in vocab.groups) == vocab.n_tags
    assert [g.name for g in vocab.groups] == ["setting", "subject", "action", "lighting", "style"]


@pytest.mark.parametrize("sec,tc", [(0, "00:00:00:00"), (61.04, "00:01:01:01"), (0.039, "00:00:00:00"),
                                    (0.04, "00:00:00:01"), (3600.0, "01:00:00:00"), (59.999, "00:00:59:24")])
def test_timecode_25fps(sec, tc):
    assert to_tc(sec) == tc


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 400), np.round(rng.random(400), 1)
    w = rng.integers(0, 3, 400).astype(float)
    assert AveragePrecision(y, s)(w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    yy, pp = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    assert macro_f1(yy, pp, np.ones(300), 5) == pytest.approx(f1_score(yy, pp, average="macro"))


def _random_audio(rng):
    from kinesync.render.audio_context import GROUPS
    a = {}
    for g, (mode, k, items) in GROUPS.items():
        vis = [v for _, v in items if v]
        a[g] = list(rng.choice(vis, size=int(rng.integers(0, k + 1)), replace=False))
    a["vocals"] = ["singer"] if rng.random() < 0.5 else []
    return a


def _random_shot(vocab, rng, audio=False):
    tags = {g.name: [] for g in vocab.groups}
    for g in vocab.groups:
        k = int(rng.integers(0, g.max_picks + 1))
        tags[g.name] = list(rng.choice(g.phrases, size=k, replace=False))
    shot = {"scale": str(rng.choice(["long shot", "close-up", "extreme close-up", "medium shot", "full shot"])),
            "moving": bool(rng.integers(0, 2)), "tags": tags, "lyric": "some words here",
            "kin": {"brightness": 0.3, "saturation": 0.4}}
    if audio:
        shot["audio"] = _random_audio(rng)
    return shot


@pytest.mark.parametrize("mode", ["off", "add", "only"])
def test_template_uses_only_given_attributes(vocab, mode):
    rng = np.random.default_rng(1)
    for _ in range(300):
        shot = _random_shot(vocab, rng, audio=True)
        pal = [str(rng.choice(["low-key", "even", "high-key"])), str(rng.choice(["muted", "natural", "vivid"]))]
        text = template_prompt(shot, pal, mode)
        extra = set(_tokens(text)) - allowed_words(shot, pal)
        assert not extra, (text, extra)
        # and every given tag phrase is present, except in "only" mode, which drops tags
        for ps in shot["tags"].values():
            for p in ps:
                assert (p in text) == (mode != "only") or p in " ".join(shot["audio"].get("vocals", []))
        assert "some" not in text            # the lyric never goes into the template


def test_prompt_wording_avoids_known_failures(vocab):
    """Each of these drew the wrong thing when panels were compared side by side."""
    from kinesync.common import load_yaml
    rng = np.random.default_rng(2)
    for _ in range(100):
        text = template_prompt(_random_shot(vocab, rng), ["even", "muted"]).lower()
        assert "static camera" not in text and "moving camera" not in text   # draws a camera
    style = load_yaml(REAL_ROOT / "configs" / "render.yaml")["panels"]
    words = (style["style_prefix"] + style["style_suffix"]).lower()
    for bad in ("storyboard", "no text", "pencil"):                          # grid, lettering, a pencil
        assert bad not in words


def test_shot_without_content_gets_a_neutral_panel():
    from kinesync.render.prompt import NO_CONTENT
    tags = {"setting": [], "subject": [], "action": [], "lighting": [], "style": []}
    shot = {"scale": "close-up", "moving": False, "tags": tags}
    assert template_prompt(shot, ["even", "muted"]) == f"Close-up, static shot: {NO_CONTENT}, even muted palette."


def test_prompts_fit_clip_token_limit_by_word_count(vocab):
    """Longest possible template prompt + style stays well under CLIP's 77 tokens."""
    from kinesync.common import load_yaml
    style = load_yaml(REAL_ROOT / "configs" / "render.yaml")["panels"]
    tags = {g.name: sorted(g.phrases, key=len, reverse=True)[:g.max_picks] for g in vocab.groups}
    text = style["style_prefix"] + template_prompt({"scale": "extreme close-up", "moving": True, "tags": tags},
                                                   ["high-key", "natural"]) + style["style_suffix"]
    assert len(text.split()) < 55, len(text.split())


def test_clap_context_modes():
    """off ignores audio; add fills only shots with no content tags; only ignores tags."""
    from kinesync.render.prompt import NO_CONTENT, drawn_phrases
    audio = {"mood": ["calm"], "energy": ["stillness"], "instruments": ["violinist"], "vocals": ["singer"]}
    empty = {"setting": [], "subject": [], "action": [], "lighting": ["neon light"], "style": []}
    tagged = dict(empty, subject=["solo singer"])
    s_empty = {"scale": "close-up", "moving": False, "tags": empty, "audio": audio}
    s_tagged = {"scale": "close-up", "moving": False, "tags": tagged, "audio": audio}
    pal = ["even", "muted"]
    assert NO_CONTENT in template_prompt(s_empty, pal, "off")
    assert "calm" not in template_prompt(s_tagged, pal, "off")
    assert drawn_phrases(s_empty, "add") == ["singer", "violinist"]
    assert drawn_phrases(s_tagged, "add") == ["solo singer"]                 # tags win when present
    assert "calm mood" in template_prompt(s_tagged, pal, "add")
    only = template_prompt(s_tagged, pal, "only")
    assert "solo singer" not in only and "neon light" not in only and "violinist" in only


def test_redraw_check_only_scores_drawable_tags():
    from kinesync.render.check import shot_texts
    tags = {"setting": ["recording studio"], "subject": [], "action": [], "lighting": ["neon light"], "style": ["vintage film grain"]}
    assert shot_texts({"tags": tags}) == ["a sketch of recording studio"]
    assert shot_texts({"tags": {"setting": [], "subject": [], "action": [], "lighting": ["neon light"], "style": []}}) == []


def test_pick_offload():
    from kinesync.render.panels import pick_offload
    assert pick_offload({}, 16) == "none"
    assert pick_offload({}, 8) == "model"
    assert pick_offload({}, 4) == "sequential"
    assert pick_offload({"offload": "none"}, 4) == "none"


def test_shot_seed_is_stable():
    from kinesync.render.panels import shot_seed
    assert shot_seed("v0012", 0, 3) == shot_seed("v0012", 0, 3)
    assert shot_seed("v0012", 0, 3) != shot_seed("v0012", 0, 4) != shot_seed("v0012", 1, 4)
    assert shot_seed("v0012", 0, 3) != shot_seed("v0012", 0, 3, attempt=1)


def test_validator(vocab):
    shot = {"scale": "close-up", "moving": False, "tags": {"subject": ["solo singer"]},
            "lyric": "one two three four five six seven eight", "kin": {}}
    pal = ["even", "muted"]
    assert validate("A close-up line-art sketch of a solo singer, static camera, muted palette.", shot, pal, []) == []
    assert validate(" ".join(["word"] * 61), shot, pal, [])
    assert validate("A singer mouthing one two three four five six seven.", shot, pal, [])
    assert validate("A sketch of dua lipa on stage.", shot, pal, ["dua lipa"])
    assert validate("A close-up of a singer beside Taylor Swift.", shot, pal, [])
    assert validate("A storyboard of a solo singer.", shot, pal, [])


def test_strict_tags_cut_by_stored_score():
    from kinesync.stage2.dataset import strict_tags
    shot = {"tags": [1, 13, 11], "tag_scores": [1.09, 2.2, 2.7]}
    assert strict_tags(shot, None) == [1, 13, 11]          # release labels unchanged
    assert strict_tags(shot, 2.0) == [13, 11]
    assert strict_tags(shot, 3.0) == []


def test_lyric_kept_rule_matches_release():
    from kinesync.features.extract import is_kept
    assert is_kept(-0.43, 1.43, "en")
    assert not is_kept(-1.07, 1.2, "en")        # the dropped first line of "Risk It All"
    assert not is_kept(-0.3, 2.6, "en")         # repetitive text (high compression ratio)
    assert not is_kept(-0.3, 1.2, "ko")         # masked language


def test_extractor_rhythm_features_meet_contract():
    """Synthetic 20 s signal: drum-like bursts every 0.5 s over a tone. Checks grids, sections and beat phase."""
    import os
    from kinesync.features import extract as ex
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REAL_ROOT / "cache" / "numba"))
    sr, dur = ex.SR, 20.0
    t = np.arange(int(sr * dur)) / sr
    y = 0.1 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(0)
    for b in np.arange(0.25, dur, 0.5):                    # sharp noise bursts, like drum hits, at 120 BPM
        i = int(b * sr)
        y[i:i + 440] += rng.normal(0, 0.8, 440) * np.exp(-np.arange(440) / 80)
    y = y.astype(np.float32)
    T10 = int(dur * 10)
    feat = ex.rhythm_features(y, T10)
    feat["clap"] = np.zeros((T10 // 10, 512), np.float16)
    assert ex.validate(feat, T10) == []
    assert sorted(np.unique(feat["section"])) == list(range(8))
    assert len(feat["beat_times"]) >= 20
    assert abs(np.median(np.diff(feat["beat_times"])) - 0.5) < 0.05     # finds the 120 BPM pulse
    assert np.allclose(np.hypot(*feat["beat_phase"].T), 1.0, atol=1e-5)
    assert (feat["downbeat_phase"] == 0).all()


def test_clap_players_rotate_and_are_reproducible(mock_root):
    """One player per shot, drawn among supported players with a penalty on over-used ones; same song, same
    players every time; switching it off gives the old top-z behaviour."""
    import yaml
    from kinesync.common import load_shots, videos_in
    from kinesync.render.audio_context import AudioDescriber
    cfg = yaml.safe_load(open(REAL_ROOT / "configs" / "render.yaml", encoding="utf-8"))["clap"]
    vid = videos_in("train")[0]
    shots = load_shots(vid)
    st, en = [s["start"] for s in shots], [s["end"] for s in shots]
    a = AudioDescriber(cfg).describe(vid, st, en)
    b = AudioDescriber(cfg).describe(vid, st, en)
    assert [x["instruments"] for x in a] == [x["instruments"] for x in b]
    assert all(len(x["instruments"]) <= cfg["players_per_shot"] for x in a)
    old = AudioDescriber({**cfg, "player_sampling": False}).describe(vid, st, en)
    assert all(len(x["instruments"]) <= 2 for x in old)
