import runpod
import base64
import tempfile
import os
import soundfile as sf
import numpy as np
from LavaSR.model import LavaEnhance

# ── Load model once at cold-start (cached between requests) ──────────────────
print("Loading LavaSR model...")
model = LavaEnhance("YatharthS/LavaSR", device="cpu")
print("Model ready.")


def handler(job):
    """
    RunPod serverless handler for LavaSR audio enhancement.

    Expected input (job["input"]):
    {
        "audio_b64": "<base64-encoded WAV/MP3 bytes>",
        "input_sr":  16000,   # optional, default 16000 (8000–48000)
        "denoise":   false,   # optional, set true to also remove noise
        "batch":     false    # optional, set true for very long audio
    }

    Returns:
    {
        "enhanced_audio_b64": "<base64-encoded WAV bytes at 16 kHz>",
        "message": "ok"
    }
    """
    job_input = job["input"]

    # ── Validate input ────────────────────────────────────────────────────────
    if "audio_b64" not in job_input:
        return {"error": "Missing required field: audio_b64"}

    audio_b64 = job_input["audio_b64"]
    input_sr  = job_input.get("input_sr", 16000)
    denoise   = job_input.get("denoise", False)
    batch     = job_input.get("batch", False)

    # ── Decode audio from base64 → temp file ─────────────────────────────────
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Failed to decode base64 audio: {str(e)}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.wav")
        output_path = os.path.join(tmpdir, "output.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # ── Run enhancement ───────────────────────────────────────────────────
        try:
            audio_tensor, sr = model.load_audio(input_path, input_sr=input_sr)
            enhanced = model.enhance(
                audio_tensor,
                denoise=denoise,
                batch=batch,
            ).cpu().numpy().squeeze()
        except Exception as e:
            return {"error": f"Enhancement failed: {str(e)}"}

        # ── Encode output as base64 WAV ───────────────────────────────────────
        sf.write(output_path, enhanced, 16000)
        with open(output_path, "rb") as f:
            enhanced_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "enhanced_audio_b64": enhanced_b64,
        "message": "ok",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
