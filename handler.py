import runpod
import base64
import tempfile
import os
import soundfile as sf

# Lazy-loaded global model
model = None

def get_model():
    global model
    if model is None:
        # Imports are here so RunPod's scanner doesn't trigger model validation
        from huggingface_hub import snapshot_download
        from LavaSR.model import LavaEnhance

        print("Downloading model weights...")
        local_path = snapshot_download(
            repo_id="YatharthS/LavaSR",
            repo_type="model",
            local_dir="/app/model_weights",
        )
        print("Loading model...")
        model = LavaEnhance(local_path, device="cpu")
        print("Model ready.")
    return model


def handler(job):
    job_input = job["input"]

    if "audio_b64" not in job_input:
        return {"error": "Missing required field: audio_b64"}

    audio_b64 = job_input["audio_b64"]
    input_sr  = job_input.get("input_sr", 16000)
    denoise   = job_input.get("denoise", False)
    batch     = job_input.get("batch", False)

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Failed to decode base64 audio: {str(e)}"}

    lava = get_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.wav")
        output_path = os.path.join(tmpdir, "output.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        try:
            audio_tensor, sr = lava.load_audio(input_path, input_sr=input_sr)
            enhanced = lava.enhance(
                audio_tensor,
                denoise=denoise,
                batch=batch,
            ).cpu().numpy().squeeze()
        except Exception as e:
            return {"error": f"Enhancement failed: {str(e)}"}

        sf.write(output_path, enhanced, 16000)
        with open(output_path, "rb") as f:
            enhanced_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "enhanced_audio_b64": enhanced_b64,
        "message": "ok",
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
