"""clipkaro: turn a long video (YouTube link or local file) into vertical 1080x1920
Shorts/Reels with face tracking and word-by-word burned-in captions.

    python clipkaro.py <youtube_url | path/to/video.mp4> --count 5 --captions english

Stages (each is cached in work/<video_id>/ so re-runs are cheap):
  1. download    yt-dlp, best quality <= 1080p        -> source.mp4
  2. transcribe  faster-whisper, word timestamps       -> transcript.json
  3. select      clips.json (hand-picked if present, else auto-scored)
  4. render      face-tracked 9:16 crop + ASS captions -> output/clip_NN.mp4
                 (captions re-transcribed per clip in romanized Hinglish by default)

To hand-pick clips: run with --stop-after transcribe, write
work/<video_id>/clips.json (list of {start, end, title, reason}), re-run.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

import overlays

ROOT = Path(__file__).resolve().parent
OUT_W, OUT_H = 1080, 1920
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")


def log(msg: str) -> None:
    print(f"[reels] {msg}", flush=True)


def run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


# --------------------------------------------------------------------------- 1
def video_id(url: str) -> str:
    if Path(url).is_file():
        return re.sub(r"\W+", "_", Path(url).stem)[:60]
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-40:]


def video_title(url: str, work: Path) -> str:
    if Path(url).is_file():
        return Path(url).stem.replace("_", " ")
    f = work / "title.txt"
    if not f.exists():
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--skip-download", "--print", "%(title)s | %(channel)s", "--encoding", "utf-8",
                            url, *(["--js-runtimes", "node"] if shutil.which("node") else [])],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        f.write_text(r.stdout.strip() if r.returncode == 0 else "", encoding="utf-8")
    return f.read_text(encoding="utf-8")


def download(url: str, work: Path) -> Path:
    src = work / "source.mp4"
    if src.exists():
        log(f"download: cached {src}")
        return src
    if Path(url).is_file():  # local file: just copy it into the work dir
        log(f"download: using local file {url}")
        shutil.copy(url, src)
        return src
    log("download: yt-dlp (<=1080p)")
    cmd = [sys.executable, "-m", "yt_dlp", url,
           "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]",
           "--merge-output-format", "mp4", "-o", str(src), "--no-playlist"]
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    run(cmd)
    return src


# --------------------------------------------------------------------------- 2
_MODELS: dict = {}


def whisper(model_name: str):
    if model_name not in _MODELS:
        from faster_whisper import WhisperModel
        try:
            _MODELS[model_name] = WhisperModel(model_name, device="cuda", compute_type="float16")
            log(f"whisper: {model_name} on CUDA")
        except Exception as e:  # no GPU / missing cuDNN
            log(f"whisper: CUDA unavailable ({e}); using CPU int8")
            _MODELS[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _MODELS[model_name]


def load_audio(src: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """16 kHz mono float32 via ffmpeg (faster-whisper's own PyAV decoder breaks
    across PyAV releases)."""
    cut = [] if start is None else ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", *cut, "-i", str(src), "-vn", "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


# Prompting Whisper's English mode with romanized Hindi makes it write Hindi speech
# down in Latin letters ("kya bolte hain") instead of translating it.
HINGLISH_PROMPT = ("Haan bhai, kya haal hai? Main theek hoon yaar. Aaj ka show bahut mast hone "
                   "wala hai, matlab sach mein. Tu kya kar raha hai abhi? Bro, yeh toh bahut funny tha.")


def caption_words(src: Path, clip: dict, work: Path, model_name: str, mode: str,
                  full_words: list[dict], hint: str = "") -> list[dict]:
    """Word timings for a clip's captions, relative to the clip start."""
    a, b = clip["start"], clip["end"]
    if mode == "english":
        return [{**w, "start": w["start"] - a, "end": w["end"] - a}
                for w in full_words if w["start"] >= a - 0.05 and w["end"] <= b + 0.05]
    cache = work / f"captions_{mode}_{a:.2f}_{b:.2f}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pad = 1.0  # context either side so the first and last words aren't clipped
    a0 = max(0.0, a - pad)
    audio = load_audio(src, a0, b + pad)

    def run_pass(lang: str, prompt: str | None) -> list[dict]:
        segs, _ = whisper(model_name).transcribe(audio, language=lang, initial_prompt=prompt,
                                                 word_timestamps=True, vad_filter=True, beam_size=5)
        out = []
        for s in segs:
            for w in s.words or []:
                t0, t1 = w.start + a0 - a, w.end + a0 - a
                if w.word.strip() and t0 >= -0.05 and t1 <= b - a + 0.05:
                    out.append({"start": round(max(t0, 0), 3), "end": round(t1, 3), "word": w.word.strip()})
        return out

    if mode == "devanagari":
        words = run_pass("hi", hint or None)
    else:
        # Title goes *inside* the Hinglish prompt as a spelling hint for names; leading
        # with English text nudges Whisper back into translating.
        prompts = [f"{HINGLISH_PROMPT} Aaj ka video hai: {hint}." if hint else HINGLISH_PROMPT,
                   HINGLISH_PROMPT,
                   "Arey yaar, sun na. Kal main ghar gaya tha aur mummy ne bola ki khana kha le. "
                   "Maine bola abhi nahi, baad mein khaunga. Bhai sach mein, bahut maza aaya."]
        best, best_score = [], -1.0
        for k, p in enumerate(prompts):
            cand = run_pass("en", p)
            score = hinglish_score(cand)
            if score > best_score:
                best, best_score = cand, score
            if score >= 0.08:   # Hinglish (code-mixed speech still scores ~0.1+); a translation scores ~0.02
                break
            log(f"  captions look translated (hindi-word ratio {score:.2f}); retrying, prompt {k + 2}")
        words = best
    cache.write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    return words


HINDI_MARKERS = set("hai hain nahi nahin main mein ki ka ke toh to bhai kya ho tha thi raha rahe "
                    "yeh ye woh wo se ko ne bhi aur kar karo diya liya hoon hu bol bola yaar "
                    "matlab kuch sab apna aap tum tu mera mujhe usne maine phir fir".split())


def hinglish_score(words: list[dict]) -> float:
    toks = [re.sub(r"[^a-z]", "", w["word"].lower()) for w in words]
    toks = [t for t in toks if t]
    return sum(t in HINDI_MARKERS for t in toks) / max(1, len(toks))


def transcribe(src: Path, work: Path, model_name: str, language: str | None) -> dict:
    out = work / "transcript.json"
    if out.exists():
        log(f"transcribe: cached {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    model = whisper(model_name)
    audio = load_audio(src)
    segments, info = model.transcribe(audio, language=language, word_timestamps=True,
                                      vad_filter=True, beam_size=5)
    data = {"language": info.language, "duration": info.duration, "segments": []}
    for seg in segments:
        data["segments"].append({
            "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip(),
            "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
                      for w in (seg.words or []) if w.word.strip()],
        })
        print(f"\r  {seg.end/info.duration*100:5.1f}%", end="", flush=True)
    print()
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    # A human-readable copy for picking clips by eye.
    (work / "transcript.txt").write_text(
        "\n".join(f"[{s['start']:8.2f}-{s['end']:8.2f}] {s['text']}" for s in data["segments"]),
        encoding="utf-8")
    log(f"transcribe: {len(data['segments'])} segments, language={info.language}")
    return data


def all_words(tr: dict) -> list[dict]:
    return [w for s in tr["segments"] for w in s["words"]]


# --------------------------------------------------------------------------- 3
HOOK_WORDS = {"why", "how", "what", "never", "secret", "truth", "nobody", "everyone", "worst",
              "best", "crazy", "insane", "actually", "imagine", "listen", "you", "stop", "biggest",
              "kya", "kyun", "kaise", "sach", "kabhi", "bhai"}


def auto_select(tr: dict, count: int, min_len: float, max_len: float) -> list[dict]:
    """Fallback picker: windows that start and end on sentence boundaries, scored by
    hook words in the first 3 s, speech density and a complete-sentence ending."""
    segs = tr["segments"]
    ends_sentence = [bool(re.search(r"[.!?।]$", s["text"])) for s in segs]
    cands = []
    for i, s in enumerate(segs):
        if i and not ends_sentence[i - 1]:
            continue  # must start a sentence
        for j in range(i, len(segs)):
            dur = segs[j]["end"] - s["start"]
            if dur > max_len:
                break
            if dur < min_len or not ends_sentence[j]:
                continue
            words = [w for k in range(i, j + 1) for w in segs[k]["words"]]
            first = " ".join(w["word"].lower() for w in words if w["start"] < s["start"] + 3)
            hook = sum(t.strip(".,!?") in HOOK_WORDS for t in first.split()) + ("?" in first) * 2
            density = len(words) / dur
            gaps = sum(max(0, b["start"] - a["end"]) > 1.5 for a, b in zip(words, words[1:]))
            score = hook * 2 + min(density, 4) - gaps * 1.5 - abs(dur - 50) / 40
            cands.append((score, s["start"], segs[j]["end"], s["text"]))
    cands.sort(reverse=True)
    chosen = []
    for score, a, b, text in cands:
        if all(b <= c["start"] or a >= c["end"] for c in chosen):
            chosen.append({"start": a, "end": b, "title": text[:60],
                           "reason": f"auto score {score:.2f}"})
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda c: c["start"])


# --------------------------------------------------------------------------- 4
class FaceFinder:
    """Faces via OpenCV YuNet (good on small faces in wide shots); per-face mouth
    openness via MediaPipe Face Landmarker, used to find who is talking.
    Falls back to a Haar cascade if the models can't be loaded."""

    def __init__(self, models: Path):
        import cv2
        self.cv2 = cv2
        self.yunet = self.lm = self.mp = None
        try:
            p = models / "face_detection_yunet_2023mar.onnx"
            if not p.exists():
                urllib.request.urlretrieve(YUNET_URL, p)
            self.yunet = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.6)
        except Exception as e:
            log(f"YuNet unavailable ({e}); using Haar cascade")
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades +
                                              "haarcascade_frontalface_default.xml")
        try:
            p = models / "face_landmarker.task"
            if not p.exists():
                urllib.request.urlretrieve(LANDMARKER_URL, p)
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            self.mp = mp
            self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(p)), num_faces=1))
        except Exception as e:
            log(f"MediaPipe landmarker unavailable ({e}); tracking largest face only")

    def faces(self, bgr) -> list[tuple[float, float, float, float]]:
        """(center_x, center_y, width, mouth_openness) per face, in source pixels.
        mouth_openness is NaN when it can't be measured."""
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        scale = min(1.0, 1280 / w)
        small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
        if self.yunet is not None:
            self.yunet.setInputSize((small.shape[1], small.shape[0]))
            _, det = self.yunet.detect(small)
            boxes = [] if det is None else [tuple(r[:4]) for r in det]
        else:
            boxes = self.haar.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5,
                                               minSize=(30, 30))
        out = []
        for x, y, bw, bh in boxes:
            x, y, bw, bh = x / scale, y / scale, bw / scale, bh / scale
            if bw < w * 0.025:   # ignore tiny background faces / posters
                continue
            out.append((x + bw / 2, y + bh / 2, bw, self._mouth(bgr, x, y, bw, bh)))
        return out

    def _mouth(self, bgr, x, y, bw, bh) -> float:
        if self.lm is None:
            return float("nan")
        cv2 = self.cv2
        H, W = bgr.shape[:2]
        pad = 0.5 * max(bw, bh)
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(W, x + bw + pad)), int(min(H, y + bh + pad))
        crop = cv2.cvtColor(cv2.resize(bgr[y0:y1, x0:x1], (256, 256)), cv2.COLOR_BGR2RGB)
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(crop)))
        if not res.face_landmarks:
            return float("nan")
        p = res.face_landmarks[0]
        face_h = abs(p[152].y - p[10].y) or 1e-6
        return abs(p[14].y - p[13].y) / face_h   # inner-lip gap / face height


def track_crop_x(seg: Path, finder: FaceFinder, crop_w: int, every: int,
                 fps: float) -> np.ndarray | None:
    """Per-frame left edge of the crop window, following the active speaker.
    Returns None when faces are found in too few frames (caller centre-crops)."""
    cv2 = finder.cv2
    cap = cv2.VideoCapture(str(seg))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    samples: list[tuple[int, list]] = []
    i = 0
    while cap.grab():
        if i % every == 0:
            _, frame = cap.retrieve()
            samples.append((i, finder.faces(frame)))
        i += 1
    cap.release()
    n = i
    if sum(bool(f) for _, f in samples) < max(3, 0.25 * len(samples)):
        return None

    # Speaking activity of each face = spread of its mouth openness over ~1.5 s,
    # matching the same face in neighbouring samples by position.
    win = max(2, int(0.75 * fps / every))
    targets = np.full(len(samples), np.nan)
    for s, (_, faces) in enumerate(samples):
        if not faces:
            continue
        best, best_score = None, -1.0
        for f in faces:
            vals = []
            for t in range(max(0, s - win), min(len(samples), s + win + 1)):
                m = [g for g in samples[t][1] if abs(g[0] - f[0]) < f[2] * 0.6]
                if m and not np.isnan(m[0][3]):
                    vals.append(m[0][3])
            activity = np.std(vals) if len(vals) >= 3 else 0.0
            score = activity + 1e-4 * f[2]  # size only breaks ties
            if score > best_score:
                best, best_score = f, score
        targets[s] = best[0]

    # Hold each speaker for at least ~1.2 s so the crop doesn't flicker between people.
    hold = max(1, int(1.2 * fps / every))
    cur, run_start = None, 0
    held = targets.copy()
    for s in range(len(targets)):
        if np.isnan(targets[s]):
            held[s] = cur if cur is not None else np.nan
            continue
        if cur is None or abs(targets[s] - cur) < crop_w * 0.25:
            cur = targets[s] if cur is None else cur * 0.7 + targets[s] * 0.3
        else:
            nxt = targets[s:s + hold]
            if np.sum(np.abs(nxt[~np.isnan(nxt)] - targets[s]) < crop_w * 0.25) >= 0.7 * hold:
                cur = targets[s]
        held[s] = cur

    idx = np.array([k for k, _ in samples])
    ok = ~np.isnan(held)
    cx = np.interp(np.arange(n), idx[ok], held[ok])
    # Steps (speaker change / camera cut) become hard cuts; inside a shot the
    # camera eases toward the subject with a dead zone so it stays mostly still.
    cam = np.empty_like(cx)
    cam[0] = cx[0]
    dead = crop_w * 0.08
    for k in range(1, n):
        d = cx[k] - cam[k - 1]
        if abs(d) > crop_w * 0.3:
            cam[k] = cx[k]
        elif abs(d) > dead:
            cam[k] = cam[k - 1] + (d - np.sign(d) * dead) * 0.06
        else:
            cam[k] = cam[k - 1]
    return np.clip(cam - crop_w / 2, 0, W - crop_w).astype(int)


# --------------------------------------------------------------------------- 5
def ass_time(t: float) -> str:
    t = max(t, 0)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def chunk_words(words: list[dict], max_words=4, max_chars=18, max_gap=0.6) -> list[list[dict]]:
    lines, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words
                    or w["start"] - cur[-1]["end"] > max_gap
                    or (len(cur) >= 2 and len(" ".join(x["word"] for x in cur + [w])) > max_chars)):
            lines.append(cur)
            cur = []
        cur.append(w)
        if len(cur) >= 2 and re.search(r"[.!?,।]$", w["word"]):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    # Fold a lone trailing word into the previous line when it fits.
    out = []
    for ln in lines:
        joinable = out and len(ln) == 1 and ln[0]["start"] - out[-1][-1]["end"] < max_gap
        if joinable and len(out[-1]) < max_words:
            out[-1] = out[-1] + ln
        elif joinable and len(out[-1]) >= 3:   # 4 + 1 -> 3 + 2
            out[-1], ln = out[-1][:-1], out[-1][-1:] + ln
            out.append(ln)
        else:
            out.append(ln)
    return out


def needs_indic_font(words: list[dict]) -> bool:
    return any(re.search(r"[ऀ-ॿ]", w["word"]) for w in words)


def write_ass(words: list[dict], path: Path, font: str, clip: dict | None = None,
              dur: float = 0.0) -> None:
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,2,60,60,560,1
{overlays.STYLES}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    hi = r"{\c&H00E6FF&\fscx112\fscy112}"   # yellow, slightly larger
    lo = r"{\c&HFFFFFF&\fscx100\fscy100}"
    ev = []
    lines = chunk_words(words)
    for li, line in enumerate(lines):
        toks = [re.sub(r"[{}\\]", "", w["word"]).upper() for w in line]
        # A line never outlives the start of the next one (no stacked captions).
        limit = lines[li + 1][0]["start"] if li + 1 < len(lines) else float("inf")
        for k, w in enumerate(line):
            start = w["start"]
            end = line[k + 1]["start"] if k + 1 < len(line) else min(w["end"] + 0.25, limit)
            if end <= start:
                continue
            text = " ".join((hi + t + lo) if j == k else t for j, t in enumerate(toks))
            pop = r"{\fad(40,0)}" if k == 0 else ""
            if clip and start < overlays.caption_start(clip):
                pop += r"{\an2\pos(540,1180)}"   # lift above the headline card
            ev.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{pop}{text}")
    if clip:
        ev += overlays.events(clip, dur)
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def probe_fps(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                        "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True).stdout.strip()
    num, den = r.split("/")
    return round(float(num) / float(den), 3)


def append_outro(out: Path, enc: list[str]) -> None:
    """Concatenate an optional end card (brand/outro.py, not included) onto a rendered clip."""
    if not (ROOT / "brand" / "outro.py").exists():
        return
    sys.path.insert(0, str(ROOT / "brand"))
    import outro
    fps = probe_fps(out)
    card = outro.make(fps)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(card), "-filter_complex",
         f"[0:v]setsar=1[v0];[1:v]setsar=1,fps={fps}[v1];"
         "[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
         "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
         "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
         "-map", "[v]", "-map", "[a]", *enc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)


def upload_copy(out: Path, max_mib: float = 9.4) -> Path:
    """Two-pass x264 copy under 10 MB, handy for platforms with small upload limits."""
    dst = out.parent / "upload" / out.name
    dst.parent.mkdir(exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    vb = int(max_mib * 1024 * 1024 * 8 / dur - 128_000)
    log_base = str(dst.parent / f"x264log-{out.stem}")  # per clip: parallel runs share this folder
    common = ["-c:v", "libx264", "-preset", "veryslow", "-x264-params", "aq-mode=3",
              "-b:v", str(vb), "-passlogfile", log_base]
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), *common, "-pass", "1", "-an",
         "-f", "mp4", "NUL" if sys.platform == "win32" else "/dev/null"])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), *common, "-pass", "2",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)])
    for f in dst.parent.glob(f"x264log-{out.stem}*"):
        f.unlink()
    return dst


def encoder_args() -> list[str]:
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    if probe.returncode == 0:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def render_clip(src: Path, clip: dict, words: list[dict], out: Path, tmp: Path,
                finder: FaceFinder, enc: list[str]) -> str:
    """words: caption word timings relative to the clip start."""
    import cv2
    a, b = clip["start"], clip["end"]
    seg = tmp / "seg.mp4"
    # Frame-accurate cut to an intermediate so tracking and rendering see identical frames.
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-c:a", "aac", "-b:a", "192k",
         str(seg)])
    cap = cv2.VideoCapture(str(seg))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    crop_w = min(W, int(round(H * OUT_W / OUT_H / 2)) * 2)

    xs = track_crop_x(seg, finder, crop_w, every=max(1, int(fps // 6)), fps=fps)
    mode = "face-tracked" if xs is not None else "center crop (no face)"

    font = "Nirmala UI" if needs_indic_font(words) else "Arial Black"
    write_ass(words, tmp / "subs.ass", font, clip, b - a)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps}", "-i", "-",
         "-i", "seg.mp4", "-map", "0:v", "-map", "1:a",
         "-vf", "ass=subs.ass", *enc, "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out.resolve())],
        stdin=subprocess.PIPE, cwd=tmp)  # cwd keeps the ASS path free of "C:" escaping
    cap = cv2.VideoCapture(str(seg))
    i = 0
    center = (W - crop_w) // 2
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = xs[min(i, len(xs) - 1)] if xs is not None else center
        crop = frame[:, x:x + crop_w]
        ff.stdin.write(cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4).tobytes())
        i += 1
    cap.release()
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError(f"ffmpeg failed on {out.name}")
    return mode


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--clips", type=Path, help="clips.json to use instead of work/<id>/clips.json")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model")
    ap.add_argument("--language", default=None, help="force a language code, e.g. hi or en")
    ap.add_argument("--captions", choices=["hinglish", "english", "devanagari"], default="hinglish",
                    help="hinglish: Hindi in Latin letters, as spoken (default); english: the "
                         "full-video transcript, which Whisper may translate; devanagari: Hindi script")
    ap.add_argument("--min-len", type=float, default=30)
    ap.add_argument("--max-len", type=float, default=90)
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    ap.add_argument("--stop-after", choices=["download", "transcribe", "select"])
    ap.add_argument("--no-outro", action="store_true", help="skip the end card (only used if brand/outro.py exists)")
    ap.add_argument("--small-copy", action="store_true",
                    help="also write a <10 MB copy of each clip to <out>/upload/")
    args = ap.parse_args()

    work = ROOT / "work" / video_id(args.url)
    work.mkdir(parents=True, exist_ok=True)

    src = download(args.url, work)
    if args.stop_after == "download":
        return
    tr = transcribe(src, work, args.model, args.language)
    if args.stop_after == "transcribe":
        log(f"transcript at {work / 'transcript.txt'}")
        return

    clips_path = args.clips or work / "clips.json"
    if clips_path.exists():
        clips = json.loads(clips_path.read_text(encoding="utf-8"))
        log(f"select: using {len(clips)} clips from {clips_path}")
    else:
        clips = auto_select(tr, args.count, args.min_len, args.max_len)
        clips_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"select: auto-picked {len(clips)} clips -> {clips_path}")
    clips = clips[:args.count]
    if args.stop_after == "select":
        return

    args.out.mkdir(parents=True, exist_ok=True)
    words = all_words(tr)
    hint = video_title(args.url, work)
    finder = FaceFinder(ROOT / "work")
    enc = encoder_args()
    tmp = work / "tmp"
    tmp.mkdir(exist_ok=True)
    for n, clip in enumerate(clips, 1):
        out = args.out / (f"{clip['slug']}.mp4" if clip.get("slug") else f"clip_{n:02d}.mp4")
        log(f"render {n}/{len(clips)}: {clip['start']:.1f}-{clip['end']:.1f}s  \"{clip['title']}\"")
        mode_c = clip.get("captions", args.captions)
        cw = [] if mode_c == "none" else caption_words(src, clip, work, args.model, mode_c, words, hint)
        mode = render_clip(src, clip, cw, out, tmp, finder, enc)
        if not args.no_outro:
            append_outro(out, enc)
        log(f"  -> {out}  ({mode})")
        if args.small_copy:
            log(f"  -> {upload_copy(out)}  (upload copy)")
    log("done")


if __name__ == "__main__":
    main()
