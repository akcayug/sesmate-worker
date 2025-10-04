import os
import json
import time
import tempfile
import subprocess
import shlex
from typing import Dict, Any, List, Optional

import requests
import runpod

# === ENV ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "whisper-1")  # ilk test için güvenli varsayılan
HF_TOKEN = os.getenv("HF_TOKEN", "")                   # pyannote için şart
PYANNOTE_PIPELINE_ID = os.getenv("PYANNOTE_PIPELINE_ID", "pyannote/speaker-diarization")
CALLBACK_BEARER = os.getenv("CALLBACK_BEARER", "")     # Django callback'inde kontrol edilecek sabit bearer
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))  # 30dk

# === Progress helper (RunPod progress_update hata verirse sessizce atla) ===
def progress_update(job: Dict[str, Any], pct: int, stage: str, detail: str = "", extra: Optional[Dict[str, Any]] = None):
    payload = {"pct": int(max(0, min(100, pct))), "stage": stage, "detail": detail, "extra": extra or {}}
    try:
        runpod.serverless.progress_update(job, payload)  # worker-basic ile uyumlu
    except Exception:
        pass

# === FFmpeg: herhangi bir formatı 16k mono WAV'a çevir ===
def to_wav16k_mono(src_url: str) -> str:
    tmpdir = tempfile.mkdtemp()
    out = os.path.join(tmpdir, "audio.wav")
    # Not: ffmpeg, URL'den direkt okuyabilir; gerekirse pre-download eklenebilir.
    cmd = f'ffmpeg -y -i "{src_url}" -vn -ac 1 -ar 16000 -f wav "{out}"'
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    return out

# === OpenAI Whisper tek-parça transcription (verbose_json ile segment döndürür) ===
def openai_transcribe(wav_path: str, language: Optional[str] = None, model: str = OPENAI_MODEL) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment.")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language:
        data["language"] = language

    with open(wav_path, "rb") as f:
        files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI transcribe error {resp.status_code}: {resp.text}")
    return resp.json()

# === Pyannote diarization ===
def run_pyannote(wav_path: str, min_spk: Optional[int] = None, max_spk: Optional[int] = None) -> List[Dict[str, Any]]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing in environment for pyannote.")
    # import here to reduce cold import in jobs without diarization
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE_ID, use_auth_token=HF_TOKEN).to("cuda")
    diar = pipeline(wav_path, min_speakers=min_spk, max_speakers=max_spk)

    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return turns

# === Segmentlere konuşmacı etiketi atama (basit overlap ile) ===
def attach_speakers(asr_segments: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not turns:
        return [{"start": s["start"], "end": s["end"], "text": s.get("text", ""), "spk": None} for s in asr_segments]

    labeled = []
    for s in asr_segments:
        s_mid = (float(s["start"]) + float(s["end"])) / 2.0
        # mid-point en çok hangi konuşmacı aralığına düşüyorsa onu ata
        best = None
        for t in turns:
            if t["start"] <= s_mid <= t["end"]:
                best = t["speaker"]
                break
        labeled.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", ""), "spk": best})
    return labeled

# === Callback POST ===
def post_callback(callback_url: str, payload: Dict[str, Any]):
    headers = {"Content-Type": "application/json"}
    if CALLBACK_BEARER:
        headers["Authorization"] = f"Bearer {CALLBACK_BEARER}"
    r = requests.post(callback_url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()

# === Ana handler ===
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Beklenen input:
    {
      "input": {
        "audio_url": "https://...signed.wav",
        "job_id": "uuid",
        "callback_url": "https://app.sesmate.com/api/gpu-stt/callback/",
        "options": {
          "mode": "openai",              # v1
          "language": null,
          "diarization": true,
          "min_speakers": 1,
          "max_speakers": 3,
          "model": "whisper-1",          # veya env OPENAI_MODEL
          "beam_size": 1,
          "vad": false,
          "align_output": false
        }
      }
    }
    """
    started = time.time()
    try:
        # ---- Parse input
        ip = job.get("input", {}) or {}
        audio_url: str = ip.get("audio_url")
        job_id: str = ip.get("job_id", "")
        callback_url: Optional[str] = ip.get("callback_url")
        options: Dict[str, Any] = ip.get("options", {}) or {}

        if not audio_url:
            raise ValueError("input.audio_url is required.")

        # ---- download
        progress_update(job, 2, "download", "fetching audio")
        wav_path = to_wav16k_mono(audio_url)

        # ---- preprocess
        progress_update(job, 8, "preprocess", "ffmpeg: 16k mono wav ready")

        # ---- transcribe (V1: OpenAI single-shot; küçük dosya ile test)
        progress_update(job, 12, "transcribe", "sending to OpenAI")
        lang = options.get("language")
        model = options.get("model", OPENAI_MODEL)

        tr_json = openai_transcribe(wav_path, language=lang, model=model)

        # OpenAI verbose_json formatında "segments" döner
        raw_segments = []
        for seg in tr_json.get("segments", []):
            raw_segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip()
            })

        if not raw_segments:
            # fallback: tüm text tek blok
            txt = (tr_json.get("text") or "").strip()
            raw_segments = [{"start": 0.0, "end": 0.0, "text": txt}] if txt else []

        progress_update(job, 75, "transcribe", f"segments: {len(raw_segments)}")

        # ---- diarization (opsiyonel)
        labeled_segments = [{"start": s["start"], "end": s["end"], "text": s["text"], "spk": None} for s in raw_segments]
        num_speakers = 0
        if options.get("diarization", True):
            progress_update(job, 80, "diarize", "pyannote running")
            min_spk = options.get("min_speakers")
            max_spk = options.get("max_speakers")
            turns = run_pyannote(wav_path, min_spk=min_spk, max_spk=max_spk)
            labeled_segments = attach_speakers(raw_segments, turns)
            spk_set = {s["spk"] for s in labeled_segments if s.get("spk")}
            num_speakers = len(spk_set)
            progress_update(job, 95, "diarize", f"speakers: {sorted(spk_set)}")

        # ---- postprocess (ilk sürümde SU/SRT üretimini Django'ya bırakıyoruz)
        result = {
            "language": tr_json.get("language") or options.get("language"),
            "segments": labeled_segments,
            "sentence_units": [],     # v1: Django üretecek
            "num_speakers": num_speakers,
            "srt_plain": None,        # v1: Django üretecek
            "srt_spk": None,          # v1: Django üretecek
            "cost_stats": {
                "gpu_seconds": 0,     # v1: OpenAI'li olduğu için şimdilik 0
                "avg_vram_gb": 0
            }
        }

        payload = {"job_id": job_id, "status": "success", "result": result}
        progress_update(job, 100, "postprocess", "done")

        # ---- callback (varsa)
        if callback_url:
            post_callback(callback_url, payload)

        return payload

    except subprocess.CalledProcessError as e:
        err = f"ffmpeg failed: {e.stderr.decode('utf-8', 'ignore') if e.stderr else str(e)}"
        payload = {"job_id": job.get("input", {}).get("job_id"), "status": "error", "error": err}
        if job.get("input", {}).get("callback_url"):
            try:
                post_callback(job["input"]["callback_url"], payload)
            except Exception:
                pass
        return payload

    except Exception as e:
        payload = {"job_id": job.get("input", {}).get("job_id"), "status": "error", "error": str(e)}
        if job.get("input", {}).get("callback_url"):
            try:
                post_callback(job["input"]["callback_url"], payload)
            except Exception:
                pass
        return payload

# Worker'ı başlat
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
