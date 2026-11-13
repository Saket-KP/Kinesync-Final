"""Portal routes, without the worker (no GPU, nothing rendered)."""
import io

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from kinesync.portal import app as portal
    monkeypatch.setattr(portal, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(portal, "OUT", tmp_path / "out")
    monkeypatch.setattr(portal, "JOBS", tmp_path / "out" / "jobs")
    portal.jobs.clear()
    while not portal.work.empty():
        portal.work.get_nowait()
    portal.app.config["TESTING"] = True
    return portal, portal.app.test_client()


def test_index_lists_genres(client):
    _, c = client
    r = c.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Make storyboard" in body and "Hip hop" in body and "Not sure" in body


def test_upload_rejects_wrong_type_and_tiny_files(client):
    _, c = client
    r = c.post("/upload", data={"song": (io.BytesIO(b"x" * 20000), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = c.post("/upload", data={"song": (io.BytesIO(b"x" * 10), "song.mp3")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_upload_queues_a_job_with_safe_options(client):
    portal, c = client
    r = c.post("/upload", data={"song": (io.BytesIO(b"\x00" * 50000), "My Song.mp3"), "genre": "not-a-genre",
                                "n_samples": "9", "steps": "4", "clap": "bogus"},
               content_type="multipart/form-data")
    assert r.status_code == 302
    jid = r.headers["Location"].rsplit("/", 1)[1]
    job = portal.jobs[jid]
    assert job["genre"] == "unknown" and job["n_samples"] == 3 and job["steps"] == 4 and job["clap"] == "add"
    assert job["file_name"] == "My Song.mp3" and portal.work.get_nowait() == jid
    assert job["content_model"] == "self"                            # missing or unknown -> default
    api = c.get(f"/api/job/{jid}").get_json()
    assert api["state"] == "queued" and "path" not in api            # no file system paths leak out


def test_out_refuses_paths_outside_results(client):
    _, c = client
    assert c.get("/out/../../splits.csv").status_code == 404
    assert c.get("/job/doesnotexist").status_code == 404


def test_content_model_choices_point_at_real_ensembles(client):
    portal, c = client
    from kinesync.common import REPO_ROOT
    for key, (label, path) in portal.CONTENT_MODELS.items():
        assert (REPO_ROOT / path).exists(), path
    r = c.post("/upload", data={"song": (io.BytesIO(b"\x00" * 50000), "a.mp3"), "content_model": "cross"},
               content_type="multipart/form-data")
    assert portal.jobs[r.headers["Location"].rsplit("/", 1)[1]]["content_model"] == "cross"


def test_retry_requeues_a_failed_job_only(client):
    portal, c = client
    r = c.post("/upload", data={"song": (io.BytesIO(b"\\x00" * 50000), "b.mp3")}, content_type="multipart/form-data")
    jid = r.headers["Location"].rsplit("/", 1)[1]
    portal.work.get_nowait()
    assert c.post(f"/job/{jid}/retry").status_code == 302 and portal.work.empty()     # queued: nothing to retry
    portal.jobs[jid].update(state="failed", error="boom", results=[{"x": 1}])
    c.post(f"/job/{jid}/retry")
    job = portal.jobs[jid]
    assert job["state"] == "queued" and job["error"] is None and job["results"] == [] and portal.work.get_nowait() == jid
