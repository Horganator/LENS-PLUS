from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import logging
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from random import choice, random
from time import monotonic
from typing import Any

import av.logging
import torch
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import Idefics3ForConditionalGeneration, Idefics3Processor
from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel


SetLogLevel(-1)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VOSK_MODEL_PATH = BASE_DIR / "models" / "vosk-model"
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
SAY_BIN = shutil.which("say")
ESPEAK_BIN = shutil.which("espeak-ng") or shutil.which("espeak")
SMOLVLM_MODEL_ID = os.getenv("SMOLVLM_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")
SMOLVLM_MAX_NEW_TOKENS = int(os.getenv("SMOLVLM_MAX_NEW_TOKENS", "120"))
SMOLVLM_NUM_BEAMS = int(os.getenv("SMOLVLM_NUM_BEAMS", "1"))
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if MODEL_DEVICE == "cuda" else torch.float32
ENABLE_MOCK_RESULTS = os.getenv("ENABLE_MOCK_RESULTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_vosk_model: VoskModel | None = None
_vosk_model_lock = threading.Lock()
_smolvlm_processor: Idefics3Processor | None = None
_smolvlm_model: Idefics3ForConditionalGeneration | None = None
_smolvlm_lock = threading.Lock()


class AudioProcessingError(RuntimeError):
    pass


class VisionModelError(RuntimeError):
    pass


class OfferRequest(BaseModel):
    sdp: str
    type: str
    session_id: str | None = None


class OfferResponse(BaseModel):
    sdp: str
    type: str
    session_id: str


class IceRequest(BaseModel):
    session_id: str
    candidate: str
    sdpMid: str | None = None
    sdpMLineIndex: int | None = None


class IceResponse(BaseModel):
    ok: bool


DEFAULT_ANALYSIS_TARGET_FPS = 5.0
MIN_ANALYSIS_TARGET_FPS = 1.0
MAX_ANALYSIS_TARGET_FPS = 30.0
FPS_WINDOW_SECONDS = 1.0
SESSION_MANIFEST_FILENAME = "session.json"


def clamp_analysis_fps(value: float) -> float:
    if not math.isfinite(value):
        return DEFAULT_ANALYSIS_TARGET_FPS

    return max(MIN_ANALYSIS_TARGET_FPS, min(MAX_ANALYSIS_TARGET_FPS, value))


def load_analysis_target_fps_from_env() -> float:
    raw_value = os.getenv("ANALYSIS_TARGET_FPS")
    if raw_value is None:
        return DEFAULT_ANALYSIS_TARGET_FPS

    try:
        return clamp_analysis_fps(float(raw_value))
    except ValueError:
        return DEFAULT_ANALYSIS_TARGET_FPS


CONFIGURED_ANALYSIS_TARGET_FPS = load_analysis_target_fps_from_env()


def load_session_artifacts_root_from_env() -> Path:
    raw_value = os.getenv("SESSION_ARTIFACTS_DIR")
    if raw_value:
        return Path(raw_value).expanduser().resolve()

    return Path(__file__).resolve().parent / "session_artifacts"


SESSION_ARTIFACTS_ROOT = load_session_artifacts_root_from_env()


class DebugVisionRequest(BaseModel):
    question: str = "What is in this image?"
    session_id: str | None = None
    use_test_image: bool = False


@dataclass
class Session:
    peer_connection: RTCPeerConnection
    data_channel: Any | None = None
    frame_task: asyncio.Task[None] | None = None
    audio_task: asyncio.Task[None] | None = None
    mock_task: asyncio.Task[None] | None = None
    analysis_target_fps: float = CONFIGURED_ANALYSIS_TARGET_FPS
    total_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    incoming_fps: float = 0.0
    processed_fps: float = 0.0
    last_frame_at: datetime | None = None
    latest_jpeg: bytes | None = None
    latest_jpeg_at: datetime | None = None
    snapshot_errors: int = 0
    last_snapshot_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_id: str | None = None
    artifact_dir: Path | None = None
    artifact_manifest_path: Path | None = None
    dumped_frames: int = 0
    dump_errors: int = 0
    last_dump_error: str | None = None
    pending_audio_chunks: dict[str, list[str]] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


app = FastAPI(title="lens-plus-signaling")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, Session] = {}
DEFAULT_FALLBACK_ANSWER = "Sorry, I didn't get that."
logger = logging.getLogger("lens-plus.api")


def get_vosk_model_path() -> Path:
    configured_path = os.getenv("VOSK_MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_VOSK_MODEL_PATH


def get_vosk_model() -> VoskModel:
    global _vosk_model

    if _vosk_model is not None:
        return _vosk_model

    model_path = get_vosk_model_path()
    if not model_path.exists():
        raise AudioProcessingError(
            "Vosk model files were not found. Set VOSK_MODEL_PATH or place a model at "
            f"{model_path}"
        )

    with _vosk_model_lock:
        if _vosk_model is None:
            _vosk_model = VoskModel(str(model_path))

    return _vosk_model


def run_subprocess(command: list[str], failure_message: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise AudioProcessingError(f"{failure_message}: missing command {command[0]}") from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode("utf-8", errors="ignore").strip()
        if details:
            raise AudioProcessingError(f"{failure_message}: {details}") from error
        raise AudioProcessingError(failure_message) from error


def convert_audio_bytes_to_wav(audio_bytes: bytes) -> Path:
    if not audio_bytes:
        raise AudioProcessingError("Received empty audio data")

    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as source_file:
        source_file.write(audio_bytes)
        source_path = Path(source_file.name)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target_file:
        target_path = Path(target_file.name)

    try:
        run_subprocess(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(target_path),
            ],
            "Failed to convert the uploaded audio into Vosk-compatible PCM WAV",
        )
        return target_path
    finally:
        source_path.unlink(missing_ok=True)


def transcribe_audio_sync(audio_bytes: bytes) -> str:
    wav_path = convert_audio_bytes_to_wav(audio_bytes)

    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise AudioProcessingError("Transcription audio must be mono after conversion")
            if wav_file.getsampwidth() != 2:
                raise AudioProcessingError("Transcription audio must be 16-bit PCM after conversion")

            recognizer = KaldiRecognizer(get_vosk_model(), wav_file.getframerate())
            recognizer.SetWords(True)

            parts: list[str] = []
            while True:
                chunk = wav_file.readframes(4000)
                if not chunk:
                    break
                if recognizer.AcceptWaveform(chunk):
                    result = json.loads(recognizer.Result())
                    text = str(result.get("text", "")).strip()
                    if text:
                        parts.append(text)

            final_result = json.loads(recognizer.FinalResult())
            final_text = str(final_result.get("text", "")).strip()
            if final_text:
                parts.append(final_text)

            transcript = " ".join(parts).strip()
            if not transcript:
                raise AudioProcessingError("Transcription was empty")

            return transcript
    except wave.Error as error:
        raise AudioProcessingError(f"Converted audio could not be decoded as WAV: {error}") from error
    finally:
        wav_path.unlink(missing_ok=True)


def synthesize_speech_sync(text: str) -> bytes:
    if not text.strip():
        raise AudioProcessingError("Cannot generate speech for an empty response")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = Path(wav_file.name)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
        mp3_path = Path(mp3_file.name)

    try:
        if SAY_BIN:
            run_subprocess(
                [SAY_BIN, "-o", str(wav_path), text],
                "Failed to synthesize speech with the macOS voice engine",
            )
        elif ESPEAK_BIN:
            run_subprocess(
                [ESPEAK_BIN, "-w", str(wav_path), text],
                "Failed to synthesize speech with the local voice engine",
            )
        else:
            raise AudioProcessingError(
                "No local TTS engine is available. Install espeak-ng on Linux or use macOS say."
            )

        run_subprocess(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(mp3_path),
            ],
            "Failed to encode the speech response as MP3",
        )

        return mp3_path.read_bytes()
    finally:
        wav_path.unlink(missing_ok=True)
        mp3_path.unlink(missing_ok=True)


async def send_json(session: Session, payload: dict[str, Any]) -> None:
    channel = session.data_channel
    if channel is None or getattr(channel, "readyState", "") != "open":
        return

    channel.send(json.dumps(payload))
    session.updated_at = datetime.now(timezone.utc)
    logger.info(
        "Sent data-channel payload type=%s updated_at=%s",
        payload.get("type", "unknown"),
        session.updated_at.isoformat(),
    )


async def send_status(session: Session, text: str) -> None:
    logger.info("Status update: %s", text)
    await send_json(session, {"type": "status", "message": text})


async def send_error(session: Session, text: str) -> None:
    logger.warning("Error sent to client: %s", text)
    await send_json(session, {"type": "error", "message": text})


async def text_to_speech_bytes(text: str) -> bytes:
    return await asyncio.to_thread(synthesize_speech_sync, text)


async def transcribe_audio(audio_bytes: bytes) -> str:
    return await asyncio.to_thread(transcribe_audio_sync, audio_bytes)


def query_image_model_sync(image_bytes: bytes, question: str) -> str:
    if not question.strip():
        raise VisionModelError("Question was empty")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
    except Exception as error:
        raise VisionModelError(f"Could not decode input image: {error}") from error

    processor, model = get_smolvlm_components()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Answer briefly in one short sentence. "
                        "Do not repeat the prompt or include role labels. "
                        f"Question: {question}"
                    ),
                },
            ],
        }
    ]

    logger.info(
        "Running SmolVLM query model_id=%s image_bytes=%d question_len=%d",
        SMOLVLM_MODEL_ID,
        len(image_bytes),
        len(question),
    )

    try:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )
        inputs = inputs.to(MODEL_DEVICE)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=SMOLVLM_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=SMOLVLM_NUM_BEAMS,
                use_cache=True,
            )

        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
    except Exception as error:
        logger.exception("SmolVLM query failed")
        raise VisionModelError(f"SmolVLM query failed: {error}") from error

    answer = clean_generated_answer(answer, prompt)

    if not answer:
        raise VisionModelError("SmolVLM returned an empty answer")

    logger.info("SmolVLM answer generated answer_len=%d", len(answer))
    return answer


async def query_image_model(image_bytes: bytes, question: str) -> str:
    return await asyncio.to_thread(query_image_model_sync, image_bytes, question)


async def send_answer(
    session: Session,
    answer: str,
    *,
    transcript: str = "",
) -> None:
    logger.info(
        "Preparing answer payload transcript_len=%d answer_len=%d",
        len(transcript),
        len(answer),
    )
    audio_response_bytes = await text_to_speech_bytes(answer)
    audio_base64 = base64.b64encode(audio_response_bytes).decode("utf-8")
    await send_json(
        session,
        {
            "type": "answer",
            "transcript": transcript,
            "answer": answer,
            "audio_base64": audio_base64,
        },
    )


async def send_fallback_answer(
    session: Session,
    *,
    transcript: str = "",
    status_message: str | None = None,
) -> None:
    logger.warning(
        "Fallback answer triggered transcript_len=%d status=%s",
        len(transcript),
        status_message or "",
    )
    if status_message:
        await send_status(session, status_message)

    try:
        await send_answer(session, DEFAULT_FALLBACK_ANSWER, transcript=transcript)
    except Exception as error:
        await send_error(session, f"{DEFAULT_FALLBACK_ANSWER} (audio generation failed: {error})")


def clean_generated_answer(answer: str, prompt: str) -> str:
    cleaned = answer.strip()

    if prompt and prompt in cleaned:
        cleaned = cleaned.replace(prompt, "", 1).strip()

    if "Assistant:" in cleaned:
        cleaned = cleaned.split("Assistant:", 1)[1].strip()

    if "User:" in cleaned:
        cleaned = cleaned.split("User:", 1)[0].strip()

    return cleaned.strip()


async def run_text_pipeline(session: Session, text: str) -> None:
    if session.latest_jpeg is None:
        await send_error(session, "No image frame available yet")
        return

    logger.info("Running text pipeline text_len=%d", len(text))

    try:
        logger.info("Querying SmolVLM for text pipeline")
        answer = await query_image_model(session.latest_jpeg, text)
    except AudioProcessingError as error:
        await send_error(session, str(error))
        return
    except VisionModelError as error:
        await send_error(session, str(error))
        return
    except Exception as error:
        await send_error(session, f"Failed to process the request: {error}")
        return

    if not answer.strip():
        await send_fallback_answer(session, transcript=text)
        return

    try:
        logger.info("Returning text pipeline answer")
        await send_answer(session, answer, transcript=text)
    except AudioProcessingError as error:
        await send_error(session, str(error))
    except Exception as error:
        await send_error(session, f"Failed to return the answer audio: {error}")


async def run_ai_pipeline(session: Session, audio_bytes: bytes) -> None:
    if session.latest_jpeg is None:
        await send_error(session, "No image frame available yet")
        return

    logger.info(
        "Running audio pipeline audio_bytes=%d has_snapshot=%s",
        len(audio_bytes),
        session.latest_jpeg is not None,
    )
    await send_status(session, "Transcribing audio...")

    try:
        logger.info("Starting Vosk transcription")
        transcript = await transcribe_audio(audio_bytes)
        logger.info("Finished Vosk transcription transcript_len=%d transcript=%r", len(transcript), transcript)
    except AudioProcessingError as error:
        await send_fallback_answer(session, status_message=str(error))
        return
    except Exception as error:
        await send_fallback_answer(session, status_message=f"Audio transcription failed: {error}")
        return

    if not transcript.strip():
        await send_fallback_answer(session, status_message="Transcription was empty")
        return

    await send_status(session, "Querying image model...")

    try:
        logger.info("Querying SmolVLM for audio pipeline")
        answer = await query_image_model(session.latest_jpeg, transcript)
        logger.info("SmolVLM answer received answer_len=%d", len(answer))
    except VisionModelError as error:
        await send_fallback_answer(session, transcript=transcript, status_message=str(error))
        return
    except Exception as error:
        await send_fallback_answer(
            session,
            transcript=transcript,
            status_message=f"SmolVLM query failed: {error}",
        )
        return

    await send_status(session, "Generating speech...")

    if not answer.strip():
        await send_fallback_answer(session, transcript=transcript)
        return

    try:
        logger.info("Generating and returning answer audio")
        await send_answer(session, answer, transcript=transcript)
    except AudioProcessingError as error:
        await send_error(session, str(error))
        return
    except Exception as error:
        await send_error(session, f"Speech generation failed: {error}")
        return


async def handle_client_message(session: Session, message: Any) -> None:
    session.updated_at = datetime.now(timezone.utc)

    try:
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        data = json.loads(message)
        msg_type = data.get("type")
        if msg_type == "question_audio":
            logger.info("Parsed data-channel message type=question_audio")
        elif msg_type == "question_audio_chunk":
            logger.info("Parsed data-channel message type=question_audio_chunk")
        else:
            logger.info("Parsed data-channel message type=%s", msg_type)

        if msg_type == "question_audio":
            audio_base64 = data.get("audio_base64")
            if not isinstance(audio_base64, str) or not audio_base64:
                await send_error(session, "No audio provided")
                return

            try:
                audio_bytes = base64.b64decode(audio_base64, validate=True)
                logger.info("Decoded question_audio bytes=%d", len(audio_bytes))
            except Exception:
                await send_error(session, "Audio payload was not valid base64")
                return

            await run_ai_pipeline(session, audio_bytes)
        elif msg_type == "question_audio_chunk":
            upload_id = str(data.get("upload_id", "")).strip()
            chunk_index = data.get("chunk_index")
            total_chunks = data.get("total_chunks")
            chunk_data = data.get("chunk_data")

            if not upload_id:
                await send_error(session, "Missing upload id for audio chunk")
                return
            if not isinstance(chunk_index, int) or not isinstance(total_chunks, int):
                await send_error(session, "Invalid audio chunk metadata")
                return
            if not isinstance(chunk_data, str) or not chunk_data:
                await send_error(session, "Missing audio chunk data")
                return
            if total_chunks <= 0 or chunk_index < 0 or chunk_index >= total_chunks:
                await send_error(session, "Audio chunk indexes were out of range")
                return

            logger.info(
                "Received audio chunk upload_id=%s chunk=%d/%d size=%d",
                upload_id,
                chunk_index + 1,
                total_chunks,
                len(chunk_data),
            )

            parts = session.pending_audio_chunks.setdefault(upload_id, [""] * total_chunks)
            if len(parts) != total_chunks:
                session.pending_audio_chunks.pop(upload_id, None)
                await send_error(session, "Audio chunk count changed during upload")
                return

            parts[chunk_index] = chunk_data

            if any(part == "" for part in parts):
                return

            session.pending_audio_chunks.pop(upload_id, None)
            audio_base64 = "".join(parts)
            logger.info(
                "Reassembled chunked audio upload_id=%s total_base64_size=%d",
                upload_id,
                len(audio_base64),
            )

            try:
                audio_bytes = base64.b64decode(audio_base64, validate=True)
                logger.info("Decoded chunked question_audio bytes=%d", len(audio_bytes))
            except Exception:
                await send_error(session, "Chunked audio payload was not valid base64")
                return

            await run_ai_pipeline(session, audio_bytes)
        elif msg_type == "question_text":
            text = str(data.get("text", "")).strip()
            if not text:
                await send_error(session, "No text provided")
                return

            logger.info("Received question_text text_len=%d", len(text))
            await run_text_pipeline(session, text)
        else:
            await send_error(session, f"Unknown message type: {msg_type}")
    except Exception as error:
        logger.exception("Failed to process client message")
        await send_error(session, f"Failed to process message: {error}")


@app.on_event("startup")
async def startup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(message)s",
    )
    av.logging.set_level(av.logging.ERROR)
    SESSION_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Starting lens-plus API")
    app.state.gc_task = asyncio.create_task(session_gc())


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("Shutting down lens-plus API")
    gc_task: asyncio.Task[None] | None = getattr(app.state, "gc_task", None)
    if gc_task:
        gc_task.cancel()
    for session_id in list(sessions.keys()):
        await close_session(session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/debug/vision")
async def debug_vision(payload: DebugVisionRequest) -> dict[str, Any]:
    image_bytes: bytes

    if payload.use_test_image:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        source = "generated_test_image"
    else:
        if not payload.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required unless use_test_image=true",
            )

        session = sessions.get(payload.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Unknown session")
        if not session.latest_jpeg:
            raise HTTPException(status_code=404, detail="No snapshot available yet")

        image_bytes = session.latest_jpeg
        source = f"session:{payload.session_id}"

    try:
        answer = await query_image_model(image_bytes, payload.question)
        return {
            "ok": True,
            "source": source,
            "question": payload.question,
            "answer": answer,
        }
    except Exception as error:
        return {
            "ok": False,
            "source": source,
            "question": payload.question,
            "error": str(error),
        }


def get_smolvlm_components() -> tuple[Idefics3Processor, Idefics3ForConditionalGeneration]:
    global _smolvlm_processor
    global _smolvlm_model

    if _smolvlm_processor is not None and _smolvlm_model is not None:
        return _smolvlm_processor, _smolvlm_model

    with _smolvlm_lock:
        if _smolvlm_processor is None or _smolvlm_model is None:
            logger.info(
                "Loading SmolVLM model_id=%s device=%s dtype=%s",
                SMOLVLM_MODEL_ID,
                MODEL_DEVICE,
                MODEL_DTYPE,
            )
            processor = Idefics3Processor.from_pretrained(SMOLVLM_MODEL_ID)
            model = Idefics3ForConditionalGeneration.from_pretrained(
                SMOLVLM_MODEL_ID,
                torch_dtype=MODEL_DTYPE,
                _attn_implementation="eager",
            ).to(MODEL_DEVICE)
            model.eval()
            _smolvlm_processor = processor
            _smolvlm_model = model

    return _smolvlm_processor, _smolvlm_model


@app.get("/debug/sessions")
async def debug_sessions() -> dict[str, list[dict[str, Any]]]:
    session_list: list[dict[str, Any]] = []
    for session_id, session in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "connection_state": session.peer_connection.connectionState,
                "ice_state": session.peer_connection.iceConnectionState,
                "analysis_target_fps": session.analysis_target_fps,
                "total_frames": session.total_frames,
                "processed_frames": session.processed_frames,
                "dropped_frames": session.dropped_frames,
                "incoming_fps": session.incoming_fps,
                "processed_fps": session.processed_fps,
                "last_frame_at": (
                    session.last_frame_at.isoformat() if session.last_frame_at else None
                ),
                "latest_jpeg_at": (
                    session.latest_jpeg_at.isoformat() if session.latest_jpeg_at else None
                ),
                "has_snapshot": session.latest_jpeg is not None,
                "snapshot_errors": session.snapshot_errors,
                "last_snapshot_error": session.last_snapshot_error,
                "started_at": session.started_at.isoformat(),
                "artifact_id": session.artifact_id,
                "artifact_dir": str(session.artifact_dir)
                if session.artifact_dir
                else None,
                "dumped_frames": session.dumped_frames,
                "dump_errors": session.dump_errors,
                "last_dump_error": session.last_dump_error,
                "updated_at": session.updated_at.isoformat(),
            }
        )

    return {"sessions": session_list}


@app.get("/debug/sessions/history")
async def debug_sessions_history() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": load_session_history()}


@app.get("/debug/sessions/{session_id}/latest.jpg")
async def debug_latest_snapshot(session_id: str) -> Response:
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    if not session.latest_jpeg:
        raise HTTPException(status_code=404, detail="No snapshot available yet")

    return Response(
        content=session.latest_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/webrtc/offer", response_model=OfferResponse)
async def offer(payload: OfferRequest) -> OfferResponse:
    for existing_session_id in list(sessions.keys()):
        await close_session(existing_session_id)

    peer_connection = RTCPeerConnection()

    session_id = payload.session_id or str(uuid.uuid4())
    session_started_at = datetime.now(timezone.utc)
    artifact_id, artifact_dir, artifact_manifest_path, artifact_error = (
        create_session_artifact(session_id=session_id, started_at=session_started_at)
    )
    session = Session(
        peer_connection=peer_connection,
        started_at=session_started_at,
        updated_at=session_started_at,
        artifact_id=artifact_id,
        artifact_dir=artifact_dir,
        artifact_manifest_path=artifact_manifest_path,
    )
    if artifact_error is not None:
        session.dump_errors += 1
        session.last_dump_error = artifact_error

    sessions[session_id] = session
    write_session_manifest(session_id, session)
    logger.info("Created session session_id=%s", session_id)

    @peer_connection.on("track")
    async def on_track(track: Any) -> None:
        kind = getattr(track, "kind", "")
        logger.info("Track received kind=%s session_id=%s", kind, session_id)

        if kind == "video":
            session.updated_at = datetime.now(timezone.utc)

            async def consume_frames() -> None:
                last_snapshot_at = datetime.min.replace(tzinfo=timezone.utc)
                fps_window_started_at = monotonic()
                window_incoming_frames = 0
                window_processed_frames = 0
                next_analysis_at = 0.0

                def update_fps_window(now_mono: float) -> None:
                    nonlocal fps_window_started_at
                    nonlocal window_incoming_frames
                    nonlocal window_processed_frames

                    elapsed = now_mono - fps_window_started_at
                    if elapsed < FPS_WINDOW_SECONDS:
                        return

                    session.incoming_fps = round(window_incoming_frames / elapsed, 2)
                    session.processed_fps = round(window_processed_frames / elapsed, 2)
                    fps_window_started_at = now_mono
                    window_incoming_frames = 0
                    window_processed_frames = 0

                while True:
                    try:
                        frame = await track.recv()
                        now = datetime.now(timezone.utc)
                        now_mono = monotonic()
                        session.total_frames += 1
                        session.last_frame_at = now
                        session.updated_at = now
                        window_incoming_frames += 1

                        target_analysis_fps = clamp_analysis_fps(
                            session.analysis_target_fps
                        )
                        analysis_interval_seconds = 1.0 / target_analysis_fps

                        if now_mono < next_analysis_at:
                            session.dropped_frames += 1
                            update_fps_window(now_mono)
                            continue

                        next_analysis_at = now_mono + analysis_interval_seconds
                        session.processed_frames += 1
                        window_processed_frames += 1

                        frame_jpeg, frame_jpeg_error = frame_to_jpeg(frame)
                        if frame_jpeg:
                            persist_processed_frame(
                                session=session,
                                frame_jpeg=frame_jpeg,
                                frame_at=now,
                            )

                            if (now - last_snapshot_at).total_seconds() >= 0.5:
                                session.latest_jpeg = frame_jpeg
                                session.latest_jpeg_at = now
                                last_snapshot_at = now
                                session.last_snapshot_error = None
                        else:
                            session.snapshot_errors += 1
                            session.last_snapshot_error = frame_jpeg_error

                        update_fps_window(now_mono)
                    except Exception:
                        break

            session.frame_task = asyncio.create_task(consume_frames())
        elif kind == "audio":
            async def consume_audio() -> None:
                while True:
                    try:
                        await track.recv()
                        session.updated_at = datetime.now(timezone.utc)
                    except Exception:
                        break

            session.audio_task = asyncio.create_task(consume_audio())

    @peer_connection.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        session.data_channel = channel
        session.updated_at = datetime.now(timezone.utc)
        logger.info(
            "Data channel attached session_id=%s label=%s readyState=%s",
            session_id,
            getattr(channel, "label", "unknown"),
            getattr(channel, "readyState", "unknown"),
        )

        @channel.on("message")
        def on_message(message: Any) -> None:
            asyncio.create_task(handle_client_message(session, message))

        @channel.on("open")
        def on_open() -> None:
            logger.info("Data channel opened session_id=%s", session_id)
            if ENABLE_MOCK_RESULTS and session.mock_task is None:
                session.mock_task = asyncio.create_task(send_mock_results(session))

        # aiortc can hand us a channel that is already open before the open
        # callback is registered. Handle that case explicitly so inbound
        # messages and mock events are not gated on a missed event.
        if getattr(channel, "readyState", "") == "open":
            logger.info("Data channel already open on attach session_id=%s", session_id)
            if ENABLE_MOCK_RESULTS and session.mock_task is None:
                session.mock_task = asyncio.create_task(send_mock_results(session))

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info(
            "Peer connection state changed session_id=%s state=%s",
            session_id,
            peer_connection.connectionState,
        )
        if peer_connection.connectionState in {"failed", "closed"}:
            await close_session(session_id)

    offer_description = RTCSessionDescription(sdp=payload.sdp, type=payload.type)
    await peer_connection.setRemoteDescription(offer_description)
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    await wait_for_ice_gathering(peer_connection)
    session.updated_at = datetime.now(timezone.utc)

    if peer_connection.localDescription is None:
        raise HTTPException(status_code=500, detail="Failed to build answer")

    return OfferResponse(
        sdp=peer_connection.localDescription.sdp,
        type=peer_connection.localDescription.type,
        session_id=session_id,
    )


@app.post("/webrtc/ice", response_model=IceResponse)
async def ice(payload: IceRequest) -> IceResponse:
    session = sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    try:
        candidate_sdp = payload.candidate
        if candidate_sdp.startswith("candidate:"):
            candidate_sdp = candidate_sdp[len("candidate:") :]

        candidate = candidate_from_sdp(candidate_sdp)
        candidate.sdpMid = payload.sdpMid
        candidate.sdpMLineIndex = payload.sdpMLineIndex
        await session.peer_connection.addIceCandidate(candidate)
        session.updated_at = datetime.now(timezone.utc)
    except Exception as error:
        raise HTTPException(
            status_code=400, detail=f"Invalid ICE candidate: {error}"
        ) from error

    return IceResponse(ok=True)


async def send_mock_results(session: Session) -> None:
    object_labels = ["chair", "desk", "door", "bag", "keys", "person"]
    while True:
        channel = session.data_channel
        if channel is None:
            await asyncio.sleep(1)
            continue

        if getattr(channel, "readyState", "") != "open":
            await asyncio.sleep(1)
            continue

        label = choice(object_labels)
        confidence = round(0.55 + random() * 0.4, 2)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "guidance_text": f"Caution: {label} ahead.",
            "scene_summary": f"Detected one {label} in view.",
            "objects": [
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [0.2, 0.2, 0.35, 0.4],
                }
            ],
        }

        try:
            channel.send(json.dumps(payload))
            session.updated_at = datetime.now(timezone.utc)
        except Exception:
            break

        await asyncio.sleep(1)


async def close_session(session_id: str) -> None:
    session = sessions.pop(session_id, None)
    if not session:
        return
    logger.info("Closing session session_id=%s", session_id)

    closed_at = datetime.now(timezone.utc)

    for task in [session.frame_task, session.audio_task, session.mock_task]:
        if task:
            task.cancel()

    await session.peer_connection.close()
    session.updated_at = closed_at
    write_session_manifest(session_id, session, closed_at=closed_at)


async def session_gc(timeout_seconds: int = 60) -> None:
    while True:
        await asyncio.sleep(10)
        now = datetime.now(timezone.utc)
        stale_sessions: list[str] = []

        for session_id, session in sessions.items():
            if (now - session.updated_at).total_seconds() > timeout_seconds:
                stale_sessions.append(session_id)

        for session_id in stale_sessions:
            await close_session(session_id)


async def wait_for_ice_gathering(
    peer_connection: RTCPeerConnection, timeout_seconds: float = 3.0
) -> None:
    if peer_connection.iceGatheringState == "complete":
        return

    async def _wait() -> None:
        while peer_connection.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(_wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return


def sanitize_artifact_component(value: str) -> str:
    safe_chars: list[str] = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "session"


def create_session_artifact(
    session_id: str, started_at: datetime
) -> tuple[str | None, Path | None, Path | None, str | None]:
    safe_session_id = sanitize_artifact_component(session_id)
    started_fragment = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    artifact_id = f"{started_fragment}--{safe_session_id}"
    artifact_dir = SESSION_ARTIFACTS_ROOT / artifact_id
    artifact_manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME

    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        artifact_id = f"{artifact_id}--{uuid.uuid4().hex[:8]}"
        artifact_dir = SESSION_ARTIFACTS_ROOT / artifact_id
        artifact_manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            return None, None, None, f"artifact init failed: {error}"
    except Exception as error:
        return None, None, None, f"artifact init failed: {error}"

    return artifact_id, artifact_dir, artifact_manifest_path, None


def persist_processed_frame(
    session: Session, frame_jpeg: bytes, frame_at: datetime
) -> None:
    if session.artifact_dir is None:
        return

    frame_name = (
        f"frame-{session.processed_frames:06d}-"
        f"{frame_at.strftime('%Y%m%dT%H%M%S%fZ')}.jpg"
    )
    frame_path = session.artifact_dir / frame_name

    try:
        frame_path.write_bytes(frame_jpeg)
        session.dumped_frames += 1
        session.last_dump_error = None
    except Exception as error:
        session.dump_errors += 1
        session.last_dump_error = f"frame dump failed: {error}"


def build_session_manifest(
    session_id: str,
    session: Session,
    closed_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "artifact_id": session.artifact_id,
        "artifact_dir": str(session.artifact_dir) if session.artifact_dir else None,
        "started_at": session.started_at.isoformat(),
        "closed_at": closed_at.isoformat() if closed_at else None,
        "analysis_target_fps": session.analysis_target_fps,
        "total_frames": session.total_frames,
        "processed_frames": session.processed_frames,
        "dropped_frames": session.dropped_frames,
        "incoming_fps": session.incoming_fps,
        "processed_fps": session.processed_fps,
        "dumped_frames": session.dumped_frames,
        "dump_errors": session.dump_errors,
        "last_dump_error": session.last_dump_error,
        "latest_jpeg_at": (
            session.latest_jpeg_at.isoformat() if session.latest_jpeg_at else None
        ),
        "last_frame_at": session.last_frame_at.isoformat()
        if session.last_frame_at
        else None,
        "updated_at": session.updated_at.isoformat(),
    }


def write_session_manifest(
    session_id: str,
    session: Session,
    closed_at: datetime | None = None,
) -> None:
    if session.artifact_manifest_path is None:
        return

    payload = build_session_manifest(
        session_id=session_id,
        session=session,
        closed_at=closed_at,
    )

    try:
        session.artifact_manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as error:
        session.dump_errors += 1
        session.last_dump_error = f"manifest write failed: {error}"


def load_session_history(limit: int = 100) -> list[dict[str, Any]]:
    if not SESSION_ARTIFACTS_ROOT.exists():
        return []

    history: list[dict[str, Any]] = []
    artifact_dirs = sorted(
        [entry for entry in SESSION_ARTIFACTS_ROOT.iterdir() if entry.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )

    for artifact_dir in artifact_dirs[:limit]:
        manifest_path = artifact_dir / SESSION_MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text())
                if isinstance(payload, dict):
                    history.append(payload)
                    continue
            except Exception as error:
                history.append(
                    {
                        "artifact_id": artifact_dir.name,
                        "artifact_dir": str(artifact_dir),
                        "manifest_error": str(error),
                    }
                )
                continue

        frame_files = [
            entry for entry in artifact_dir.iterdir() if entry.name.endswith(".jpg")
        ]
        history.append(
            {
                "artifact_id": artifact_dir.name,
                "artifact_dir": str(artifact_dir),
                "dumped_frames": len(frame_files),
            }
        )

    return history


def frame_to_jpeg(frame: Any) -> tuple[bytes | None, str | None]:
    errors: list[str] = []

    try:
        image = frame.to_image()
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"to_image failed: {error}")

    try:
        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"rgb24 fallback failed: {error}")

    try:
        gray = frame.to_ndarray(format="gray")
        image = Image.fromarray(gray, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"gray fallback failed: {error}")

    return None, " | ".join(errors)
