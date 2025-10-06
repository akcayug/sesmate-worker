import os
import json
import time
import shlex
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod

import numpy as _np
import torch as _torch
print("[env] numpy", _np.__version__, "| torch", _torch.__version__)
print("[env] numpy", _np.__version__, "| torch", _torch.__version__, "| cuda?", _torch.cuda.is_available())
print("[env] HF_TOKEN set?", bool(os.getenv("HF_TOKEN")), "| PIPELINE_ID:", os.getenv("PYANNOTE_PIPELINE_ID"))


# ======================
# ENV / Defaults
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "whisper-1")
HF_TOKEN = os.getenv("HF_TOKEN", "")
PYANNOTE_PIPELINE_ID = os.getenv("PYANNOTE_PIPELINE_ID", "pyannote/speaker-diarization-3.1")
CALLBACK_BEARER = os.getenv("CALLBACK_BEARER", "")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))  # 30 dk

# ======================
# Helpers: progress / callback
# ======================
def progress_update(job: Dict[str, Any], pct: int, stage: str, detail: str = "", extra: Optional[Dict[str, Any]] = None):
    payload = {"pct": int(max(0, min(100, pct))), "stage": stage, "detail": detail, "extra": extra or {}}
    try:
        runpod.serverless.progress_update(job, payload)
    except Exception:
        pass

def post_callback(callback_url: str, payload: Dict[str, Any]):
    headers = {"Content-Type": "application/json"}
    if CALLBACK_BEARER:
        headers["Authorization"] = f"Bearer {CALLBACK_BEARER}"
    r = requests.post(callback_url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()

# ======================
# Download + FFmpeg
# ======================
def download_to_temp(src_url: str) -> str:
    """
    URL içeriğini (redirect/HTML dâhil) diske kaydeder.
    """
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "input.bin")
    with requests.get(
        src_url, stream=True, allow_redirects=True, timeout=180,
        headers={"User-Agent": "Mozilla/5.0"}
    ) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
    return out_path

def to_wav16k_mono(src_url: str) -> str:
    """
    1) indir → 2) ffmpeg ile 16 kHz mono WAV'a çevir
    """
    in_path = download_to_temp(src_url)
    tmpdir = tempfile.mkdtemp()
    out = os.path.join(tmpdir, "audio.wav")
    cmd = f'ffmpeg -y -i "{in_path}" -vn -ac 1 -ar 16000 -f wav "{out}"'
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    return out

# ======================
# OpenAI Whisper
# ======================
def openai_transcribe(wav_path: str, language: Optional[str] = None, model: str = OPENAI_MODEL) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment.")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language and language.lower() not in ("auto", "auto-detect"):
        data["language"] = language

    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI transcribe error {resp.status_code}: {resp.text}")
    return resp.json()

# ======================
# Pyannote diarization
# ======================
def run_pyannote(wav_path: str, min_spk: Optional[int] = None, max_spk: Optional[int] = None,
                 hparams: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Dönen format:
      [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing in environment for pyannote diarization.")

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(f"pyannote import failed: {e}")

    try:
        pipe = Pipeline.from_pretrained(PYANNOTE_PIPELINE_ID, use_auth_token=HF_TOKEN)
        if pipe is None:
            raise RuntimeError("Pipeline.from_pretrained returned None (check HF token & access).")
    except Exception as e:
        raise RuntimeError(f"pyannote load failed: {e}. "
                           f"Hint: Ensure your HF token has access to {PYANNOTE_PIPELINE_ID} "
                           f"(accept license on the model page).")

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    try:
        pipe.to(device)
    except Exception as e:
        raise RuntimeError(f"pyannote .to({device}) failed: {e}")

    # Hparam (opsiyonel)
    if hparams:
        for k1, sub in hparams.items():
            try:
                obj = getattr(pipe, k1)
                for k2, v in (sub or {}).items():
                    setattr(obj, k2, v)
            except Exception:
                pass

    try:
        diar = pipe(wav_path, min_speakers=min_spk, max_speakers=max_spk)
    except Exception as e:
        raise RuntimeError(f"pyannote inference failed: {e}")

    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    turns.sort(key=lambda x: (x["start"], x["end"]))
    return turns

def normalize_turns(turns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set]:
    """
    'SPEAKER_00/01' → 'S1/S2' map edip:
      [{"start":.., "end":.., "spk":"S1"}, ...], speaker_set döndürür.
    """
    mapping: Dict[str, str] = {}
    counter = 1
    norm: List[Dict[str, Any]] = []
    spk_set: set = set()

    for t in sorted(turns or [], key=lambda x: (x["start"], x["end"])):
        raw = str(t.get("spk") or t.get("speaker") or "")
        if not raw:
            continue
        if raw not in mapping:
            mapping[raw] = f"S{counter}"
            counter += 1
        spk = mapping[raw]
        norm.append({"start": float(t["start"]), "end": float(t["end"]), "spk": spk})
        spk_set.add(spk)

    return norm, spk_set

# ======================
# Postprocess (utils ile birebir sıra)
# ======================
def build_base_subs_from_asr(
    segs: List[Dict[str, Any]],
    *,
    merge_gap_ms: int = 200,
    min_dur_ms: int = 600,
    max_dur_ms: int = 4000,
    pause_split_ms: int = 300,
    target_chars: int = 42,
    max_lines: int = 2,
) -> List[Dict[str, Any]]:
    """
    Whisper segmentlerini makul bloklara birleştirir (Django utils mantığına paralel).
    """
    def ms(x: float) -> int: return int(round(x * 1000))
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for s in segs:
        st, en = float(s["start"]), float(s["end"])
        txt = (s.get("text") or "").strip()
        if not txt:
            continue

        if not cur:
            cur = {"start": st, "end": en, "text": txt}
            continue

        gap = ms(st - cur["end"])
        if gap <= merge_gap_ms and (len(cur["text"]) + 1 + len(txt) <= target_chars * max_lines):
            cur["end"] = en
            cur["text"] = (cur["text"] + " " + txt).strip()
        else:
            dur = ms(cur["end"] - cur["start"])
            if dur < min_dur_ms and out:
                out[-1]["end"] = cur["end"]
                out[-1]["text"] = (out[-1]["text"] + " " + cur["text"]).strip()
            else:
                out.append(cur)
            cur = {"start": st, "end": en, "text": txt}

    if cur:
        out.append(cur)

    final: List[Dict[str, Any]] = []
    for s in out:
        dur = ms(s["end"] - s["start"])
        if dur < min_dur_ms and final:
            final[-1]["end"] = s["end"]
            final[-1]["text"] = (final[-1]["text"] + " " + s["text"]).strip()
        else:
            final.append(s)
    return final

def split_on_punctuation(
    subs: List[Dict[str, Any]],
    *,
    regex: str = r"[.?!]+",
    keep_delim: bool = True,
    min_part_ms: int = 200,
) -> List[Dict[str, Any]]:
    import re
    if not subs:
        return []
    pat = re.compile(regex)
    out: List[Dict[str, Any]] = []
    for s in subs:
        st = float(s["start"]); en = float(s["end"])
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spans = [m.span() for m in pat.finditer(text)]
        if not spans:
            out.append(s); continue

        total_len = max(1, len(text))
        cur_i = 0
        for (a, b) in spans:
            piece = text[cur_i:b].strip() if keep_delim else text[cur_i:a].strip()
            if not piece:
                cur_i = b; continue
            pst = st + (en - st) * (cur_i / total_len)
            pen = st + (en - st) * (b / total_len)
            if (pen - pst) * 1000 >= min_part_ms:
                out.append({"start": pst, "end": pen, "text": piece})
            cur_i = b
        tail = text[cur_i:].strip()
        if tail:
            pst = st + (en - st) * (cur_i / total_len)
            if (en - pst) * 1000 >= min_part_ms:
                out.append({"start": pst, "end": en, "text": tail})
    return out

def enforce_srt_limits(
    subs: List[Dict[str, Any]],
    *,
    max_dur_ms: int = 3000,
    target_chars: int = 36,
    max_lines: int = 2,
) -> List[Dict[str, Any]]:
    """
    Çok uzun/kısa satırları güvenle böl/düzelt (utils ile aynı mantık).
    """
    if not subs:
        return []
    HARD = target_chars * max_lines
    out: List[Dict[str, Any]] = []
    for s in subs:
        st = float(s["start"]); en = float(s["end"])
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        dur = int(round((en - st) * 1000))
        if dur <= max_dur_ms and len(txt) <= HARD:
            out.append(s)
            continue
        # kaba ikiye bölme (metin ortası)
        mid = st + (en - st) / 2.0
        toks = txt.split()
        if len(toks) >= 2:
            half = len(toks) // 2
            tL = " ".join(toks[:half])
            tR = " ".join(toks[half:])
        else:
            tL, tR = txt, ""
        out.append({"start": st, "end": mid, "text": tL})
        if tR:
            out.append({"start": mid, "end": en, "text": tR})
    return out

def split_on_switch_for_dialog(
    subs: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    *,
    dominance: float = 0.70,
    min_part_ms: int = 250,
) -> List[Dict[str, Any]]:
    if not subs or not turns:
        return subs

    def majority(st: float, en: float) -> Tuple[Optional[str], Dict[str, float]]:
        by = {}
        for t in turns:
            ts, te, sp = float(t["start"]), float(t["end"]), t.get("spk")
            if te <= st or ts >= en or not sp:
                continue
            ov = min(en, te) - max(st, ts)
            if ov > 0:
                by[sp] = by.get(sp, 0.0) + ov
        if not by:
            return None, {}
        winner = max(by, key=by.get)
        ratio = by[winner] / (sum(by.values()) or 1.0)
        return (winner if ratio >= dominance else None), by

    out: List[Dict[str, Any]] = []
    for s in subs:
        st = float(s["start"]); en = float(s["end"])
        win, _ = majority(st, en)
        if win:
            out.append(s); continue

        # turn sınırlarında böl
        cut_points = []
        for t in turns:
            ts, te = float(t["start"]), float(t["end"])
            if st < ts < en: cut_points.append(ts)
            if st < te < en: cut_points.append(te)
        cut_points = sorted(set(cut_points))
        if not cut_points:
            out.append(s); continue

        pts = [st] + cut_points + [en]
        tiny = False
        for i in range(len(pts) - 1):
            if (pts[i+1] - pts[i]) * 1000 < min_part_ms:
                tiny = True; break
        if tiny:
            out.append(s); continue

        text = (s.get("text") or "").strip()
        total = en - st if en > st else 1.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i+1]
            fa, fb = (a - st) / total, (b - st) / total
            ia, ib = int(round(fa * len(text))), int(round(fb * len(text)))
            piece = (text[ia:ib] or "").strip()
            if piece:
                out.append({"start": a, "end": b, "text": piece})
    return out

def label_speakers_no_split(
    subs: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    *,
    allowed_speakers: Optional[set] = None,
    collar_ms: int = 140,
    attach_gap_ms: int = 900,
) -> List[Dict[str, Any]]:
    if not subs:
        return []
    # normalize already S1/S2 in 'turns'
    _turns: List[Dict[str, Any]] = []
    for t in turns or []:
        sp = t.get("spk")
        if not sp:
            continue
        if allowed_speakers and sp not in allowed_speakers:
            continue
        _turns.append({"start": float(t["start"]), "end": float(t["end"]), "spk": sp})
    _turns.sort(key=lambda x: (x["start"], x["end"]))

    def majority(st: float, en: float) -> Optional[str]:
        by = {}
        for t in _turns:
            if t["end"] <= st or t["start"] >= en:
                continue
            ov = min(en, t["end"]) - max(st, t["start"])
            if ov > 0:
                by[t["spk"]] = by.get(t["spk"], 0.0) + ov
        return max(by, key=by.get) if by else None

    def nearest(st: float, en: float) -> Optional[str]:
        mid = 0.5 * (st + en)
        best = (1e9, None)
        for t in _turns:
            if t["start"] <= mid <= t["end"]:
                return t["spk"]
            d = min(abs(mid - t["start"]), abs(mid - t["end"]))
            if d < best[0]:
                best = (d, t["spk"])
        if best[0] * 1000 <= attach_gap_ms:
            return best[1]
        return None

    out: List[Dict[str, Any]] = []
    prev_spk: Optional[str] = None
    for s in subs:
        st, en = float(s["start"]), float(s["end"])
        spk = majority(st, en) or nearest(st, en) or prev_spk or "S1"
        prev_spk = spk
        d = dict(s); d["spk"] = spk
        out.append(d)
    return out

def build_su_from_subs_lite(
    labeled_subs: List[Dict[str, Any]],
    *,
    merge_gap_ms: int = 700,
    max_su_ms: int = 10000,
    max_su_chars: int = 240,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    def flush():
        nonlocal cur
        if cur:
            out.append(cur); cur = None
    for s in labeled_subs:
        if not cur:
            cur = {**s, "covers": [(s["start"], s["end"])]}
            continue
        gap_ms = int(round((float(s["start"]) - float(cur["end"])) * 1000))
        same_spk = (s.get("spk") == cur.get("spk"))
        cand_txt = (cur.get("text", "") + " " + s.get("text", "")).strip()
        cand_ms = int(round((float(s["end"]) - float(cur["start"])) * 1000))
        if same_spk and gap_ms <= merge_gap_ms and cand_ms <= max_su_ms and len(cand_txt) <= max_su_chars:
            cur["end"] = s["end"]; cur["text"] = cand_txt
            cov = cur.get("covers") or []; cov.append((s["start"], s["end"])); cur["covers"] = cov
        else:
            flush(); cur = {**s, "covers": [(s["start"], s["end"])]}
    flush()
    return out

# ======================
# Handler
# ======================
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    input:
    {
      "audio_url": "...",
      "job_id": "uuid",
      "callback_url": "https://.../api/gpu-stt/callback/",
      "options": {
        "language": "tr" | "en" | "auto",
        "diarization": true,
        "min_speakers": 1,
        "max_speakers": 3,
        "use_su": false,
        # Postprocess tuning (Django defaultlarına paralel — opsiyonel):
        "merge_gap_ms": 200,
        "min_dur_ms": 600,
        "max_dur_ms": 4000,
        "pause_split_ms": 300,
        "target_chars": 42,
        "max_lines": 2,
        "srt_max_dur_ms": 3000,
        "su_merge_gap_ms": 700
      }
    }
    """
    started = time.time()
    try:
        ip = job.get("input", {}) or {}
        audio_url: str = ip.get("audio_url")
        job_id: str = ip.get("job_id", "")
        callback_url: Optional[str] = ip.get("callback_url")
        opt: Dict[str, Any] = ip.get("options", {}) or {}

        if not audio_url:
            raise ValueError("input.audio_url is required.")

        # 1) download + preprocess
        progress_update(job, 3, "download", "fetching audio")
        wav_path = to_wav16k_mono(audio_url)
        progress_update(job, 8, "preprocess", "ffmpeg: 16k mono wav ready")

        # 2) transcribe
        progress_update(job, 12, "transcribe", "sending to OpenAI")
        language = opt.get("language")
        tr_json = openai_transcribe(wav_path, language=language, model=OPENAI_MODEL)

        # normalize ASR (segments → [{start,end,text}...])
        raw_segments: List[Dict[str, Any]] = []
        for seg in tr_json.get("segments", []) or []:
            raw_segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip()
            })
        if not raw_segments:
            txt = (tr_json.get("text") or "").strip()
            if txt:
                raw_segments = [{"start": 0.0, "end": max(0.6, len(txt)/14.0), "text": txt}]

        progress_update(job, 60, "transcribe", f"segments: {len(raw_segments)}")

        # 3) diarization (opsiyonel; view'deki retry mantığı ile)
        turns_norm: List[Dict[str, Any]] = []
        spk_set: set = set()
        if bool(opt.get("diarization", True)) and HF_TOKEN:
            progress_update(job, 68, "diarize", "pyannote running")
            turns = run_pyannote(wav_path, min_spk=opt.get("min_speakers"), max_spk=opt.get("max_speakers"))
            turns_norm, spk_set = normalize_turns(turns)
            # tek konuşmacı ise retry hparam (view'deki gibi biraz daha agresif)
            if len(spk_set) <= 1:
                retry_hparams = {
                    "segmentation": {"min_duration_on": 0.15, "min_duration_off": 0.15},
                    "clustering":   {"min_cluster_size": 6, "threshold": 0.62},
                }
                turns_retry = run_pyannote(wav_path, min_spk=opt.get("min_speakers"), max_spk=opt.get("max_speakers"), hparams=retry_hparams)
                turns2, spk_set2 = normalize_turns(turns_retry)
                if len(spk_set2) > len(spk_set):
                    turns_norm, spk_set = turns2, spk_set2
            progress_update(job, 78, "diarize", f"speakers: {sorted(spk_set) or ['S1']}")
        elif bool(opt.get("diarization", True)) and not HF_TOKEN:
            progress_update(job, 68, "diarize", "skipped (HF_TOKEN missing)")

        # 4) postprocess — Django utils sırası
        # 4.1 base merge
        base_subs = build_base_subs_from_asr(
            raw_segments,
            merge_gap_ms=int(opt.get("merge_gap_ms", 200)),
            min_dur_ms=int(opt.get("min_dur_ms", 600)),
            max_dur_ms=int(opt.get("max_dur_ms", 4000)),
            pause_split_ms=int(opt.get("pause_split_ms", 300)),
            target_chars=int(opt.get("target_chars", 42)),
            max_lines=int(opt.get("max_lines", 2)),
        )
        base_subs = [s for s in base_subs if (s.get("text") or "").strip()]

        # 4.2 punctuation
        punct_subs = split_on_punctuation(base_subs, regex=r"[.?!]+", keep_delim=True, min_part_ms=200)
        punct_subs = [s for s in punct_subs if (s.get("text") or "").strip()]

        # 4.3 limits
        limited = enforce_srt_limits(
            punct_subs,
            max_dur_ms=int(opt.get("srt_max_dur_ms", 3000)),
            target_chars=int(opt.get("target_chars", 36)),
            max_lines=int(opt.get("max_lines", 2)),
        )

        # 4.4 dialog split on switch (opsiyonel; view'de True)
        SPLIT_ON_SWITCH_FOR_DIALOG = True
        if SPLIT_ON_SWITCH_FOR_DIALOG and turns_norm:
            limited = split_on_switch_for_dialog(limited, turns_norm, dominance=0.70, min_part_ms=250)
            limited = [s for s in limited if (s.get("text") or "").strip()]

        # 4.5 label without split
        if turns_norm:
            labeled = label_speakers_no_split(
                limited, turns_norm,
                allowed_speakers=None,
                collar_ms=140,
                attach_gap_ms=900,
            )
        else:
            labeled = [{**s, "spk": None} for s in limited]

        # 4.6 SU (opsiyonel)
        use_su = bool(opt.get("use_su", False))
        if use_su:
            su_list = build_su_from_subs_lite(
                labeled,
                merge_gap_ms=int(opt.get("su_merge_gap_ms", 700)),
                max_su_ms=10000,
                max_su_chars=240,
            )
        else:
            su_list = [{**s, "covers": [(s["start"], s["end"])]} for s in labeled]

        progress_update(job, 95, "postprocess", f"out: seg={len(labeled)}, su={len(su_list)}")

        # 5) result — SRT üretimi Django’da
        result = {
            "language": tr_json.get("language") or opt.get("language"),
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"], "spk": s.get("spk")} for s in labeled],
            "sentence_units": su_list,
            "num_speakers": len(spk_set) if turns_norm else 0,
            "turns": [{"start": t["start"], "end": t["end"], "spk": t.get("spk")} for t in (turns_norm or [])],
            "srt_plain": None,
            "srt_spk": None,
            "cost_stats": {"gpu_seconds": 0, "avg_vram_gb": 0},
        }

        payload = {"job_id": job_id, "status": "success", "result": result}
        progress_update(job, 100, "done", "completed")

        if callback_url:
            post_callback(callback_url, payload)
        return payload

    except subprocess.CalledProcessError as e:
        err = f"ffmpeg failed: {e.stderr.decode('utf-8', 'ignore') if e.stderr else str(e)}"
        payload = {"job_id": job.get("input", {}).get("job_id"), "status": "error", "error": err}
        if job.get("input", {}).get("callback_url"):
            try: post_callback(job["input"]["callback_url"], payload)
            except Exception: pass
        return payload

    except Exception as e:
        payload = {"job_id": job.get("input", {}).get("job_id"), "status": "error", "error": str(e)}
        if job.get("input", {}).get("callback_url"):
            try: post_callback(job["input"]["callback_url"], payload)
            except Exception: pass
        return payload

# RunPod entry
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
