"""KineSync portal: upload a song, get a storyboard.

    .venv/Scripts/python.exe -m kinesync.portal.app            # then open http://127.0.0.1:7860

Runs on this machine only (bound to 127.0.0.1). An upload becomes a job; one worker runs the jobs one at a
time (there is one GPU) by calling `python -m kinesync.pipeline --audio ...` in a subprocess, so a crash
or a GPU out-of-memory error ends that job, not the portal. The worker reads the pipeline's progress lines
([extract], [stage1], [stage2], [render], panel counts, [result]) and the job page polls them.

Files: uploads in uploads/, results in outputs/portal/{song id}_s{k}/, job records in outputs/portal/jobs/.
All git-ignored: uploads are your songs, and the sheets contain their lyrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_from_directory, url_for

from kinesync.common import REPO_ROOT, genre_list

UPLOADS = REPO_ROOT / "uploads"
OUT = REPO_ROOT / "outputs" / "portal"
JOBS = OUT / "jobs"
ALLOWED = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
MAX_MB = 60
GENRE_LABELS = {"pop": "Pop", "rock": "Rock", "hiphop": "Hip hop", "electronic": "Electronic",
                "rnb_latin": "R&B / Latin"}
STAGES = [("extract", "Analysing the audio"), ("stage1", "Planning shot timing"),
          ("stage2", "Choosing shot content"), ("render", "Drawing panels"), ("done", "Done")]
# Stage 2 content models offered in the form (results/README.md, "Cross-attention"): all three score about
# the same on val; self-attention is the default because it is as good and the fastest.
CONTENT_MODELS = {
    "self": ("Self-attention (default)", "configs/stage2/final_ensemble.yaml"),
    "cross": ("Cross-attention to the song", "configs/stage2/cross_ensemble.yaml"),
    "both": ("Both together (8 models, slower)", "configs/stage2/mixed_ensemble.yaml"),
}
IMAGE_MODELS = {   # first = default; set from the bake-off in results/README.md
    "sana-sprint": "Sana-Sprint 1.6B (recommended, about 4 s a panel)",
    "sdxl-turbo": "SDXL-Turbo (about 3 s a panel)",
}
PANEL_LINE = re.compile(r"^\s+(\S+): (\d+)/(\d+) panels, ([\d.]+) s/panel")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 2**20
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


# --------------------------------------------------------------------------- jobs

def save(job: dict) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    tmp = JOBS / f"{job['id']}.tmp"
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    tmp.replace(JOBS / f"{job['id']}.json")


def load_jobs() -> None:
    """Earlier jobs survive a restart. One that was running when the portal stopped is marked failed."""
    for p in sorted(JOBS.glob("*.json")):
        job = json.loads(p.read_text(encoding="utf-8"))
        if job["state"] in ("queued", "running"):
            job.update(state="failed", error="The portal was stopped while this job was running. Upload again.")
            save(job)
        jobs[job["id"]] = job


def progress(job: dict) -> float:
    """Rough overall fraction: audio analysis 12%, timing and content 8%, panels the rest."""
    if job["state"] == "done":
        return 1.0
    stage = job["stage"]
    if stage == "extract":
        return 0.04 + 0.08 * job.get("extract_frac", 0.0)
    if stage in ("stage1", "stage2"):
        return 0.14 if stage == "stage1" else 0.18
    if stage == "render":
        total = job["n_samples"] * max(job.get("panels_per_sample", 1), 1)
        done = sum(job.get("panels_done", {}).values())
        return 0.2 + 0.8 * min(done / max(total, 1), 1.0)
    return 0.02


def eta_s(job: dict) -> float | None:
    if job["stage"] != "render" or not job.get("s_per_panel"):
        return None
    total = job["n_samples"] * job.get("panels_per_sample", 0)
    left = total - sum(job.get("panels_done", {}).values())
    return max(left, 0) * job["s_per_panel"]


def run(job: dict) -> None:
    # each job gets its own folder, so runs of one song with different options never overwrite each other
    cmd = [sys.executable, "-u", "-m", "kinesync.pipeline", "--audio", job["path"], "--n_samples",
           str(job["n_samples"]), "--out", str(OUT / job["id"]), "--steps", str(job["steps"]),
           "--clap_context", job["clap"],
           "--ckpt", str(REPO_ROOT / CONTENT_MODELS[job.get("content_model", "self")][1]),
           "--image_model", job.get("image_model", next(iter(IMAGE_MODELS)))]
    if job["genre"] != "unknown":
        cmd += ["--genre", job["genre"]]
    cmd += ["--title", job["file_name"]]                  # uploads are stored under a hash
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    last_save = 0.0
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line.strip() or "it/s]" in line or "iB/s]" in line:       # skip download / progress bars
            continue
        with lock:
            job["log"] = (job["log"] + [line])[-60:]
            if line.startswith("[extract]"):
                job["stage"], job["message"] = "extract", line[10:]
                m = re.search(r"CLAP (\d+)/(\d+) seconds", line)
                if m:
                    job["extract_frac"] = 0.2 + 0.5 * int(m.group(1)) / int(m.group(2))
                elif "lyrics" in line:
                    job["extract_frac"] = 0.75
                elif "reusing" in line:
                    job["extract_frac"] = 1.0
                    job["song_info"] = "analysis reused from an earlier upload of this file"
                elif line.startswith("[extract] done"):
                    job["extract_frac"] = 1.0
                    job["song_info"] = line[10:]
            elif line.startswith("[stage1]"):
                job["stage"], job["message"] = "stage1", line[9:]
            elif line.startswith("[stage2]"):
                job["stage"], job["message"] = "stage2", line[9:]
            elif line.startswith("[render]"):
                job["stage"], job["message"] = "render", line[9:]
                m = re.search(r": (\d+) panels", line)
                if m:
                    job["panels_per_sample"] = max(job.get("panels_per_sample", 0), int(m.group(1)))
            elif PANEL_LINE.match(line):
                m = PANEL_LINE.match(line)
                job.setdefault("panels_done", {})[m.group(1)] = int(m.group(2))
                job["s_per_panel"] = float(m.group(4))
                job["message"] = f"{m.group(2)} of {m.group(3)} panels in {m.group(1)}"
            elif line.startswith("[result] "):
                r = json.loads(line[9:])
                rel = lambda p: Path(p).resolve().relative_to(OUT.resolve()).as_posix()
                job["results"].append({"sample_id": r["sample_id"], "n_shots": r["n_shots"],
                                       "html": rel(r["html"]), "csv": rel(r["csv"]), "edl": rel(r["edl"]),
                                       "contact": rel(r["contact_sheet"]), "renderer": r["renderer"]})
                job["song_id"] = Path(r["out_dir"]).name.rsplit("_", 1)[0]
            if time.time() - last_save > 2:
                save(job)
                last_save = time.time()
    code = proc.wait()
    with lock:
        job["finished"] = time.time()
        if code == 0 and job["results"]:
            job.update(state="done", stage="done", message="Storyboard ready")
        else:
            tail = [l for l in job["log"] if l.strip()][-6:]
            job.update(state="failed", error="The pipeline stopped with an error:\n" + "\n".join(tail))
        save(job)


def worker() -> None:
    while True:
        jid = work.get()
        job = jobs[jid]
        with lock:
            job.update(state="running", started=time.time(), stage="extract", message="Starting")
            save(job)
        try:
            run(job)
        except Exception as e:                                   # never let one job stop the worker
            with lock:
                job.update(state="failed", error=f"{type(e).__name__}: {e}", finished=time.time())
                save(job)


# --------------------------------------------------------------------------- routes

def public(job: dict) -> dict:
    """What the job page needs (no file system paths)."""
    keys = ("id", "file_name", "genre", "n_samples", "steps", "clap", "content_model", "image_model", "state", "stage", "message", "results",
            "error", "created", "started", "finished", "song_info", "song_id")
    out = {k: job.get(k) for k in keys}
    out["progress"] = round(progress(job), 3)
    out["eta_s"] = eta_s(job)
    out["log"] = job["log"][-15:]
    out["position"] = (list(work.queue).index(job["id"]) + 1) if job["state"] == "queued" and job["id"] in list(work.queue) else None
    out["stage_label"] = dict(STAGES).get(job["stage"], "Waiting")
    return out


@app.get("/")
def index():
    recent = sorted(jobs.values(), key=lambda j: j["created"], reverse=True)[:12]
    genres = [(g, GENRE_LABELS.get(g, g)) for g in genre_list()]
    return render_template_string(INDEX, jobs=recent, genres=genres, max_mb=MAX_MB, allowed=sorted(ALLOWED),
                                  content_models=CONTENT_MODELS, image_models=IMAGE_MODELS)


@app.post("/upload")
def upload():
    f = request.files.get("song")
    if f is None or not f.filename:
        return render_template_string(INDEX_ERROR, message="Choose an audio file first."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return render_template_string(INDEX_ERROR, message=f"{ext or 'That file'} is not supported. "
                                      f"Use one of: {', '.join(sorted(ALLOWED))}."), 400
    data = f.read()
    if len(data) < 10_000:
        return render_template_string(INDEX_ERROR, message="That file is too small to be a song."), 400
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{hashlib.sha256(data).hexdigest()[:16]}{ext}"
    path.write_bytes(data)
    genre = request.form.get("genre", "unknown")
    if genre not in genre_list() + ["unknown"]:
        genre = "unknown"
    clap = request.form.get("clap", "add") if request.form.get("clap") in ("off", "add", "only") else "add"
    content = request.form.get("content_model", "self")
    content = content if content in CONTENT_MODELS else "self"
    image_model = request.form.get("image_model", "")
    image_model = image_model if image_model in IMAGE_MODELS else next(iter(IMAGE_MODELS))
    job = {"id": uuid.uuid4().hex[:10], "file_name": Path(f.filename).name[:120], "path": str(path),
           "genre": genre, "n_samples": min(max(int(request.form.get("n_samples", 1)), 1), 3),
           "steps": 4 if request.form.get("steps") == "4" else 2, "clap": clap, "content_model": content, "image_model": image_model,
           "state": "queued", "stage": "queued", "message": "Waiting for the GPU", "log": [], "results": [],
           "created": time.time(), "started": None, "finished": None, "error": None}
    with lock:
        jobs[job["id"]] = job
        save(job)
    work.put(job["id"])
    return redirect(url_for("job_page", jid=job["id"]))


@app.get("/job/<jid>")
def job_page(jid):
    if jid not in jobs:
        abort(404)
    return render_template_string(JOB, job=public(jobs[jid]))


@app.post("/job/<jid>/retry")
def retry(jid):
    """Run a failed job again in the same folder: panels it already drew are reused (panels/manifest.json)."""
    if jid not in jobs:
        abort(404)
    with lock:
        job = jobs[jid]
        if job["state"] != "failed":
            return redirect(url_for("job_page", jid=jid))
        job.update(state="queued", stage="queued", message="Waiting for the GPU", error=None, results=[], log=[],
                   started=None, finished=None, panels_done={})
        save(job)
    work.put(jid)
    return redirect(url_for("job_page", jid=jid))


@app.get("/api/job/<jid>")
def job_api(jid):
    if jid not in jobs:
        abort(404)
    with lock:
        return jsonify(public(jobs[jid]))


@app.get("/out/<path:p>")
def out_file(p):
    return send_from_directory(OUT, p, as_attachment=p.endswith((".csv", ".edl")))


@app.errorhandler(413)
def too_big(_):
    return render_template_string(INDEX_ERROR, message=f"That file is over {MAX_MB} MB."), 413


# --------------------------------------------------------------------------- pages

BASE_CSS = """
:root { --bg:#f6f5f2; --card:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e0db; --accent:#2f5d8a;
        --accent-ink:#ffffff; --bar:#e9e6df; --ok:#2e7d4f; --bad:#a33a3a; --warn-bg:#fbf4e4; --warn:#7a5a12; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#151517; --card:#1f1f22; --ink:#ececee; --muted:#a0a0a8; --line:#34343a; --accent:#7fb0e0;
        --accent-ink:#0f1720; --bar:#2c2c31; --ok:#6cc28f; --bad:#e08a8a; --warn-bg:#2b2516; --warn:#e2c27a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; } h2 { font-size: 17px; margin: 28px 0 10px; }
.sub { color: var(--muted); margin: 0 0 24px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px; }
label { display:block; font-weight:600; margin: 14px 0 6px; } .hint { color:var(--muted); font-size:13px; font-weight:400; }
select, input[type=file] { width:100%; padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--ink); font:inherit; }
.row { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0 16px; }
button, .btn { display:inline-block; margin-top:20px; padding:10px 18px; border:0; border-radius:8px; background:var(--accent);
  color:var(--accent-ink); font:600 15px/1.2 inherit; text-decoration:none; cursor:pointer; }
.btn.ghost { background:transparent; color:var(--accent); border:1px solid var(--line); margin-top:0; }
.note { background:var(--warn-bg); color:var(--warn); border-radius:8px; padding:10px 12px; font-size:13px; margin-top:16px; }
table { width:100%; border-collapse: collapse; font-size: 14px; } td, th { text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:500; } a { color: var(--accent); }
.state-done { color:var(--ok); } .state-failed { color:var(--bad); } .muted { color:var(--muted); }
.bar { height:10px; background:var(--bar); border-radius:6px; overflow:hidden; margin:12px 0 6px; }
.bar > div { height:100%; background:var(--accent); width:0; transition: width .6s ease; }
.steps { display:flex; flex-wrap:wrap; gap:6px 14px; font-size:13px; color:var(--muted); margin-top:10px; }
.steps .on { color:var(--ink); font-weight:600; } .steps .past { color:var(--ok); }
pre { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:10px; font-size:12px; white-space:pre-wrap; max-height:220px; overflow:auto; }
.result img { width:100%; height:auto; border:1px solid var(--line); border-radius:8px; display:block; margin:10px 0; }
.links { display:flex; flex-wrap:wrap; gap:10px; }
"""

INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineSync Portal</title><style>""" + BASE_CSS + """</style></head><body><main>
<h1>KineSync</h1><p class="sub">Upload a song and get a shot-by-shot storyboard: timing, framing, camera, content and a sketch panel for every shot.</p>
<form class="card" action="/upload" method="post" enctype="multipart/form-data">
  <label for="song">Song <span class="hint">({{ allowed|join(', ') }}; up to {{ max_mb }} MB)</span></label>
  <input id="song" name="song" type="file" accept="{{ allowed|join(',') }},audio/*" required>
  <div class="row">
    <div><label for="genre">Genre</label><select id="genre" name="genre">
      {% for g, name in genres %}<option value="{{ g }}">{{ name }}</option>{% endfor %}
      <option value="unknown">Not sure</option></select></div>
    <div><label for="n_samples">Storyboards <span class="hint">(different takes)</span></label><select id="n_samples" name="n_samples">
      <option value="1">1</option><option value="2">2</option><option value="3">3</option></select></div>
    <div><label for="steps">Panel quality</label><select id="steps" name="steps">
      <option value="2">Standard (about 4 s a panel)</option><option value="4">Cleaner lines (about 7 s a panel)</option></select></div>
    <div><label for="clap">Use the sound for content</label><select id="clap" name="clap">
      <option value="add">Tags + sound (recommended)</option><option value="off">Tags only</option><option value="only">Sound only</option></select></div>
    <div><label for="image_model">Panel model</label><select id="image_model" name="image_model">
      {% for key, label in image_models.items() %}<option value="{{ key }}">{{ label }}</option>{% endfor %}</select></div>
    <div><label for="content_model">Content model</label><select id="content_model" name="content_model">
      {% for key, (label, _) in content_models.items() %}<option value="{{ key }}">{{ label }}</option>{% endfor %}</select></div>
  </div>
  <button type="submit">Make storyboard</button>
  <div class="note">On this laptop a 3-minute song takes about 1 minute to analyse and 5 to 8 minutes to draw per storyboard.
  Shot timing for an uploaded song comes from a simple beat-based stand-in, because the trained timing model (Stage 1) is not in the project yet.
  Songs and lyrics stay on this computer.</div>
</form>
{% if jobs %}<h2>Recent</h2><div class="card"><table><tr><th>Song</th><th>Genre</th><th>Status</th><th></th></tr>
{% for j in jobs %}<tr><td>{{ j.file_name }}</td><td>{{ j.genre }}</td><td class="state-{{ j.state }}">{{ j.state }}</td>
<td><a href="/job/{{ j.id }}">Open</a></td></tr>{% endfor %}</table></div>{% endif %}
</main></body></html>"""

INDEX_ERROR = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineSync Portal</title><style>""" + BASE_CSS + """</style></head><body><main>
<h1>KineSync</h1><div class="card"><p class="state-failed">{{ message }}</p><a class="btn" href="/">Back</a></div></main></body></html>"""

JOB = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineSync Job</title><style>""" + BASE_CSS + """</style></head><body><main>
<p><a href="/">&larr; New song</a></p>
<h1 id="title">{{ job.file_name }}</h1>
<p class="sub" id="meta">{{ job.genre }} &middot; {{ job.n_samples }} storyboard{{ 's' if job.n_samples > 1 else '' }} &middot; {{ job.steps }} steps &middot; sound: {{ job.clap }} &middot; content model: {{ job.content_model or 'self' }} &middot; panels: {{ job.image_model or 'sdxl-turbo' }}</p>
<div class="card" id="status">
  <div><strong id="stage">{{ job.stage_label }}</strong> <span class="muted" id="msg"></span></div>
  <div class="bar"><div id="fill"></div></div>
  <div class="muted" id="eta"></div>
  <div class="steps" id="steps"></div>
  <p class="muted" id="info"></p>
  <div id="error" class="state-failed" style="white-space:pre-wrap"></div>
  <form id="retry" method="post" action="/job/{{ job.id }}/retry" style="display:none">
    <button type="submit">Retry</button> <span class="muted">Panels already drawn are reused.</span></form>
  <details><summary class="muted">Log</summary><pre id="log"></pre></details>
</div>
<div id="results"></div>
<script>
const STAGES = [["extract","Analysing the audio"],["stage1","Planning shot timing"],["stage2","Choosing shot content"],["render","Drawing panels"],["done","Done"]];
const jid = {{ job.id|tojson }};
function fmt(s){ s=Math.round(s); return s<60 ? s+" s" : Math.floor(s/60)+" min "+(s%60)+" s"; }
function esc(t){ const d=document.createElement('div'); d.textContent=t??''; return d.innerHTML; }
let shown = 0;
async function tick(){
  let j; try { j = await (await fetch('/api/job/'+jid)).json(); } catch(e){ setTimeout(tick, 4000); return; }
  document.getElementById('stage').textContent = j.state==='queued' ? 'Waiting for the GPU' : (j.state==='failed' ? 'Failed' : j.stage_label);
  document.getElementById('msg').textContent = j.state==='queued' ? (j.position ? '(position '+j.position+' in the queue)' : '') : (j.state==='running' ? (j.message||'') : '');
  document.getElementById('fill').style.width = (100*j.progress)+'%';
  const el = j.started ? ((j.finished||Date.now()/1000) - j.started) : 0;
  document.getElementById('eta').textContent = j.state==='done' ? 'Finished in '+fmt(el) : (j.started ? 'Running for '+fmt(el)+(j.eta_s!=null ? ', about '+fmt(j.eta_s)+' to go' : '') : '');
  const cur = STAGES.findIndex(s=>s[0]===j.stage);
  document.getElementById('steps').innerHTML = STAGES.slice(0,4).map((s,i)=>'<span class="'+(i<cur||j.state==='done'?'past':(i===cur?'on':''))+'">'+(i+1)+'. '+s[1]+'</span>').join('');
  document.getElementById('info').textContent = j.song_info ? 'Audio: '+j.song_info : '';
  document.getElementById('error').textContent = j.error||'';
  document.getElementById('log').textContent = (j.log||[]).join('\\n');
  if ((j.results||[]).length > shown){
    const box = document.getElementById('results');
    for (const r of j.results.slice(shown)){
      box.insertAdjacentHTML('beforeend', '<h2>Storyboard '+esc(r.sample_id)+' &middot; '+r.n_shots+' shots</h2><div class="card result">'
        +'<div class="links"><a class="btn" style="margin-top:0" href="/out/'+r.html+'" target="_blank" rel="noopener">Open storyboard sheet</a>'
        +'<a class="btn ghost" href="/out/'+r.csv+'">Cut list (CSV)</a><a class="btn ghost" href="/out/'+r.edl+'">EDL (experimental)</a></div>'
        +'<a href="/out/'+r.html+'" target="_blank" rel="noopener"><img alt="All panels of storyboard '+esc(r.sample_id)+'" src="/out/'+r.contact+'"></a>'
        +'<p class="muted">C marks shots whose content came from the sound (CLAP) because no tags were predicted for them.</p></div>');
    }
    shown = j.results.length;
  }
  document.getElementById('retry').style.display = j.state==='failed' ? 'block' : 'none';
  if (j.state==='queued' || j.state==='running') setTimeout(tick, 2000);
}
tick();
</script></main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()
    print(f"KineSync portal on http://127.0.0.1:{a.port}", flush=True)
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
