import os
import gc
import numpy as np
import logging
import tempfile
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn

import whisperx
from whisperx.diarize import DiarizationPipeline

# Configure basic logging with a timestamp format
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

language_code = "zh"  # set to your language code
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

# Load heavy models once at startup
logging.info("Loading WhisperX ASR model...")
model = whisperx.load_model("large-v3", device, language="zh", compute_type=compute_type)

logging.info("Loading diarization pipeline...")
diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

# logging.info("--loading align model")
# model_a, metadata = whisperx.load_align_model(
#     language_code=language_code, device=device
# )

app = FastAPI(title="WhisperX ASR Upload API")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Use uploaded bytes directly as audio (no temp file)
        audio = np.frombuffer(contents, np.int16).flatten().astype(np.float32) / 32768.0

        # 1. Transcribe with original whisper (batched)
        logging.info("--transcribing")
        result = model.transcribe(audio, batch_size=batch_size)
        logging.info(f"--before alignment: {result['segments']}")
        logging.info("--aligning")

        # 2. Align text with time
        #result1 = whisperx.align(
        #    result["segments"],
        #    model_a,
        #    metadata,
        #    audio,
        #    device,
        #    return_char_alignments=False,
        #)
        #logging.info(f"--after alignment: {result1['segments']}")

        # 3. Assign speaker labels
        logging.info("--diarizing")
        diarize_segments = diarize_model(audio, max_speakers=3)  # pass audio bytes directly
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logging.info(f"--diarize segs assigned: {result}")
        # Convert to JSON-serializable structure and return
        return JSONResponse(content=jsonable_encoder(result['segments']))

    finally:
        # Optionally free some GPU memory between requests if needed
        gc.collect()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
