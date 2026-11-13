#!/usr/bin/env python3
# download_videos.py — Builds the 800-candidate list AND downloads split by laptop (even/odd).
import random
import re
import time
from pathlib import Path
import pandas as pd
import yt_dlp

# 5 Genres x 50 Artists = 250 Artists -> up to 3 videos each (~750-800 candidates)
GENRE_ARTISTS = {
    "pop": [
        "Dua Lipa", "The Weeknd", "Taylor Swift", "Ariana Grande", "Bruno Mars",
        "Billie Eilish", "Olivia Rodrigo", "Ed Sheeran", "Harry Styles", "Katy Perry",
        "Lady Gaga", "Justin Bieber", "Rihanna", "Shawn Mendes", "Camila Cabello",
        "Selena Gomez", "Charlie Puth", "Halsey", "Sam Smith", "Doja Cat",
        "Sabrina Carpenter", "Chappell Roan", "Troye Sivan", "Lorde", "Carly Rae Jepsen",
        "Sia", "Adele", "Miley Cyrus", "Demi Lovato", "Zara Larsson",
        "Ava Max", "Bebe Rexha", "Rita Ora", "Ellie Goulding", "Anne-Marie",
        "Lewis Capaldi", "Niall Horan", "Zayn", "Rosalia", "Tate McRae",
        "Madison Beer", "Conan Gray", "Lauv", "Bazzi", "Meghan Trainor",
        "onerepublic", "Maroon 5", "Jonas Brothers", "Backstreet Boys", "Little Mix"
    ],
    "rock": [
        "Linkin Park", "Green Day", "Foo Fighters", "Red Hot Chili Peppers", "Arctic Monkeys",
        "The Killers", "Muse", "Paramore", "Fall Out Boy", "My Chemical Romance",
        "Blink-182", "Twenty One Pilots", "Imagine Dragons", "The Strokes", "Franz Ferdinand",
        "Kings of Leon", "Queens of the Stone Age", "Nirvana", "Pearl Jam", "Soundgarden",
        "The Smashing Pumpkins", "Weezer", "The Offspring", "Sum 41", "Simple Plan",
        "Three Days Grace", "Breaking Benjamin", "Evanescence", "Bring Me The Horizon", "Nothing But Thieves",
        "Royal Blood", "The Black Keys", "Jack White", "Kasabian", "Oasis",
        "Blur", "Radiohead", "Coldplay", "Keane", "Snow Patrol",
        "Incubus", "Hoobastank", "Papa Roach", "Disturbed", "Avenged Sevenfold",
        "Halestorm", "The Pretty Reckless", "Greta Van Fleet", "Maneskin", "Ghost"
    ],
    "hiphop": [
        "Kendrick Lamar", "Drake", "J. Cole", "Travis Scott", "Post Malone",
        "Eminem", "Kanye West", "Jay-Z", "50 Cent", "Snoop Dogg",
        "Dr. Dre", "Ice Cube", "Nas", "OutKast", "A$AP Rocky",
        "Tyler, The Creator", "Mac Miller", "Childish Gambino", "Logic", "Big Sean",
        "Future", "21 Savage", "Metro Boomin", "Lil Baby", "Gunna",
        "Jack Harlow", "Central Cee", "Dave", "Stormzy", "Skepta",
        "Nicki Minaj", "Cardi B", "Megan Thee Stallion", "Ice Spice", "GloRilla",
        "Missy Elliott", "Busta Rhymes", "Ludacris", "T.I.", "Lil Wayne",
        "Rick Ross", "Wiz Khalifa", "Kid Cudi", "Joey Bada$$", "Denzel Curry",
        "JID", "EarthGang", "Vince Staples", "Schoolboy Q", "Pusha T"
    ],
    "electronic": [
        "Calvin Harris", "David Guetta", "Avicii", "Martin Garrix", "Zedd",
        "Kygo", "Alan Walker", "Marshmello", "The Chainsmokers", "Major Lazer",
        "Skrillex", "Diplo", "Disclosure", "Clean Bandit", "Rudimental",
        "Galantis", "Swedish House Mafia", "Tiesto", "Armin van Buuren", "Hardwell",
        "Afrojack", "Steve Aoki", "Dimitri Vegas & Like Mike", "Don Diablo", "Oliver Heldens",
        "Robin Schulz", "Felix Jaehn", "Lost Frequencies", "Medusa", "Topic",
        "Fred again..", "Peggy Gou", "Bicep", "ODESZA", "Flume",
        "Porter Robinson", "Madeon", "Illenium", "Gryffin", "San Holo",
        "Kaytranada", "Justice", "Daft Punk", "The Chemical Brothers", "Fatboy Slim",
        "Pendulum", "Sub Focus", "Wilkinson", "Chase & Status", "Netsky"
    ],
    "rnb_latin": [
        "SZA", "Frank Ocean", "Daniel Caesar", "H.E.R.", "Khalid",
        "Bryson Tiller", "Summer Walker", "Jhene Aiko", "Kehlani", "Ella Mai",
        "Brent Faiyaz", "Giveon", "Lucky Daye", "Victoria Monet", "Chloe x Halle",
        "Usher", "Chris Brown", "Ne-Yo", "Alicia Keys", "Beyonce",
        "John Legend", "Miguel", "Tinashe", "Jorja Smith", "Kali Uchis",
        "Bad Bunny", "J Balvin", "Karol G", "Maluma", "Ozuna",
        "Daddy Yankee", "Rauw Alejandro", "Feid", "Anuel AA", "Nicky Jam",
        "Shakira", "Enrique Iglesias", "Luis Fonsi", "Camila", "Romeo Santos",
        "Peso Pluma", "Fuerza Regida", "Grupo Frontera", "Christian Nodal", "Sebastián Yatra",
        "Manuel Turizo", "Camilo", "Danny Ocean", "Myke Towers", "Farruko"
    ]
}

# Reject non-official or altered video formats
BAD_TITLE_WORDS = re.compile(
    r"\b(lyric|lyrics|audio|visuali[sz]er|live|concert|tour|teaser|trailer|behind the scenes|"
    r"making of|interview|reaction|karaoke|instrumental|sped up|slowed|reverb|8d|cover|parody|"
    r"remix|fan made|tribute|lesson|tutorial|shorts)\b",
    re.IGNORECASE,
)

def build_candidates_csv(out_csv: Path, per_genre_target: int = 160, max_per_artist: int = 3):
    """Stage 1: Flat-search YouTube metadata (no video download) to build candidates.csv."""
    rows, seen_yt_ids = [], set()
    ydl_opts = {"quiet": True, "extract_flat": True, "skip_download": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for genre, artists in GENRE_ARTISTS.items():
            genre_count = 0
            for artist in artists:
                if genre_count >= per_genre_target:
                    break
                info = ydl.extract_info(f"ytsearch10:{artist} official music video", download=False)
                artist_kept = 0
                for entry in (info.get("entries") or []):
                    if artist_kept >= max_per_artist or genre_count >= per_genre_target:
                        break
                    yt_id, title, dur = entry.get("id"), entry.get("title", ""), entry.get("duration")
                    if (not yt_id or len(yt_id) != 11 or yt_id in seen_yt_ids
                            or BAD_TITLE_WORDS.search(title) or dur is None or not (120 <= float(dur) <= 420)):
                        continue
                    seen_yt_ids.add(yt_id)
                    artist_kept += 1
                    genre_count += 1
                    rows.append({
                        "video_id": f"v{len(rows):04d}", "youtube_id": yt_id,
                        "artist": artist, "genre": genre, "year": 2020,
                        "video_type": "mixed", "duration_s": round(float(dur), 2),
                        "title": title.replace(",", " "),
                    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def download_candidates(candidates_csv: Path, manifest_csv: Path, raw_dir: Path,
                        worker_id: int = 0, num_workers: int = 2):
    """Stage 2: Partition candidates by even/odd index across laptops and download 480p MP4s."""
    df = pd.read_csv(candidates_csv)
    # Partition by even/odd global index so Laptop 0 and Laptop 1 never overlap
    rows = [r.to_dict() for idx, (_, r) in enumerate(df.iterrows()) if idx % num_workers == worker_id]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for r in rows:
        vid, yt_id = str(r["video_id"]), str(r["youtube_id"])
        ydl_opts = {
            "format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best[height<=720]",
            "merge_output_format": "mp4",
            "outtmpl": str(raw_dir / f"{vid}.%(ext)s"),
            "quiet": True, "retries": 3,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={yt_id}", download=True)
                dur = float(info.get("duration") or r["duration_s"])
                manifest.append({**r, "duration_s": round(dur, 2), "status": "ok", "error": ""})
        except Exception as e:
            manifest.append({**r, "status": "failed", "error": str(e)[:120]})
        time.sleep(random.uniform(2.0, 4.5))  # Rate-limit sleep to avoid YouTube IP blocks

    pd.DataFrame(manifest).to_csv(manifest_csv, index=False)