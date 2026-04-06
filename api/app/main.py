from __future__ import annotations

import asyncio
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from random import choice, random
from typing import Any

import av.logging
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel


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


@dataclass
class Session:
    peer_connection: RTCPeerConnection
    data_channel: Any | None = None
    frame_task: asyncio.Task[None] | None = None
    mock_task: asyncio.Task[None] | None = None
    total_frames: int = 0
    last_frame_at: datetime | None = None
    latest_jpeg: bytes | None = None
    latest_jpeg_at: datetime | None = None
    snapshot_errors: int = 0
    last_snapshot_error: str | None = None
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


def read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


SNAPSHOT_INTERVAL_SECONDS = max(0.03, read_float_env("SNAPSHOT_INTERVAL_SECONDS", 0.1))
SNAPSHOT_JPEG_QUALITY = min(95, max(60, read_int_env("SNAPSHOT_JPEG_QUALITY", 92)))


@app.on_event("startup")
async def startup() -> None:
    av.logging.set_level(av.logging.ERROR)
    app.state.gc_task = asyncio.create_task(session_gc())


@app.on_event("shutdown")
async def shutdown() -> None:
    gc_task: asyncio.Task[None] | None = getattr(app.state, "gc_task", None)
    if gc_task:
        gc_task.cancel()
    for session_id in list(sessions.keys()):
        await close_session(session_id)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/debug/sessions")
async def debug_sessions() -> dict[str, list[dict[str, Any]]]:
    session_list: list[dict[str, Any]] = []
    for session_id, session in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "connection_state": session.peer_connection.connectionState,
                "ice_state": session.peer_connection.iceConnectionState,
                "total_frames": session.total_frames,
                "last_frame_at": (
                    session.last_frame_at.isoformat() if session.last_frame_at else None
                ),
                "latest_jpeg_at": (
                    session.latest_jpeg_at.isoformat()
                    if session.latest_jpeg_at
                    else None
                ),
                "has_snapshot": session.latest_jpeg is not None,
                "snapshot_errors": session.snapshot_errors,
                "last_snapshot_error": session.last_snapshot_error,
                "updated_at": session.updated_at.isoformat(),
            }
        )

    return {"sessions": session_list}


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
    peer_connection = RTCPeerConnection()

    session_id = payload.session_id or str(uuid.uuid4())
    session = Session(peer_connection=peer_connection)
    sessions[session_id] = session

    @peer_connection.on("track")
    async def on_track(track: Any) -> None:
        if getattr(track, "kind", "") != "video":
            return

        session.updated_at = datetime.now(timezone.utc)

        async def consume_frames() -> None:
            last_snapshot_at = datetime.min.replace(tzinfo=timezone.utc)
            while True:
                try:
                    frame = await track.recv()
                    now = datetime.now(timezone.utc)
                    session.total_frames += 1
                    session.last_frame_at = now
                    session.updated_at = now

                    if (now - last_snapshot_at).total_seconds() >= SNAPSHOT_INTERVAL_SECONDS:
                        snapshot, snapshot_error = frame_to_jpeg(
                            frame, jpeg_quality=SNAPSHOT_JPEG_QUALITY
                        )
                        if snapshot:
                            session.latest_jpeg = snapshot
                            session.latest_jpeg_at = now
                            last_snapshot_at = now
                            session.last_snapshot_error = None
                        else:
                            session.snapshot_errors += 1
                            session.last_snapshot_error = snapshot_error
                except Exception:
                    break

        session.frame_task = asyncio.create_task(consume_frames())

    @peer_connection.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        session.data_channel = channel
        session.updated_at = datetime.now(timezone.utc)

        @channel.on("open")
        def on_open() -> None:
            if session.mock_task is None:
                session.mock_task = asyncio.create_task(send_mock_results(session))

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
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

    for task in [session.frame_task, session.mock_task]:
        if task:
            task.cancel()

    await session.peer_connection.close()


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


def frame_to_jpeg(
    frame: Any, jpeg_quality: int = SNAPSHOT_JPEG_QUALITY
) -> tuple[bytes | None, str | None]:
    errors: list[str] = []

    try:
        image = frame.to_image()
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"to_image failed: {error}")

    try:
        rgb = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"rgb24 fallback failed: {error}")

    try:
        gray = frame.to_ndarray(format="gray")
        image = Image.fromarray(gray, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        return buffer.getvalue(), None
    except Exception as error:
        errors.append(f"gray fallback failed: {error}")

    return None, " | ".join(errors)
