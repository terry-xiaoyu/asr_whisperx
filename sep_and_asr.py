from importlib.resources import contents
import os
import io
import gc
import numpy as np
import logging
import torch
import torchaudio
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import scipy.io.wavfile
import whisperx
import asyncio

# Configure basic logging with a timestamp format
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Get a logger instance
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

language_code = "zh"  # set to your language code
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

# Load heavy whiper_models once at startup
logging.info("Loading WhisperX ASR whiper_model...")
whiper_model = whisperx.load_model("large-v3", device, language="zh", compute_type=compute_type)

logging.info("Loading speech separation pipeline...")
pipeline = Pipeline.from_pretrained(
  "pyannote/speech-separation-ami-1.0",
  use_auth_token=hf_token)
pipeline.to(torch.device("cuda"))

app = FastAPI(title="WhisperX ASR Upload API")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        results = await separate_and_transcribe(contents)
        # Convert to JSON-serializable structure and return
        return JSONResponse(content=jsonable_encoder(results))

    finally:
        # Optionally free some GPU memory between requests if needed
        gc.collect()

def transcribe(audio):
    # Use uploaded bytes directly as audio (no temp file)
    #audio = np.frombuffer(audio_source, np.int16).flatten().astype(np.float32) / 32768.0

    # 1. Transcribe with original whisper (batched)
    logging.info("--transcribing")
    result = whiper_model.transcribe(audio, batch_size=batch_size)
    return result

async def separate_and_transcribe(contents: bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(contents))
    diarization, sources = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    results = []

    # Launch concurrent transcriptions in threads (whisperx transcribe is blocking / GPU-bound)
    tasks = []
    for s, _speaker in enumerate(diarization.labels()):
        audio_source = sources.data[:, s].astype(np.float32)
        tasks.append(asyncio.to_thread(transcribe, audio_source))

    transcriptions = await asyncio.gather(*tasks)

    # Collect per-speaker segments and tag with speaker id
    for _, res in enumerate(transcriptions):
        results.append(res)
    return results

if __name__ == "__main__":
    ## test separate_and_transcribe from local file
    with open("audio_demo_16k.wav", "rb") as f:
        contents = f.read()
    logger.info("Testing separate_and_transcribe function...")
    results = asyncio.run(separate_and_transcribe(contents))
    logger.info(f"Results: {results}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
