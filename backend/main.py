# backend/main.py
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from celery.result import AsyncResult

from backend.celery_app import celery
from backend.video_service import task_video_analysis
from backend.audio_service import task_audio_analysis
from backend.metadata_service import task_metadata_analysis
from backend.scoring import compute_final_score

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/analyze")
async def analyze(video_path: str):

    analysis_id = str(uuid4())

    video_task = task_video_analysis.delay(video_path)
    audio_task = task_audio_analysis.delay(video_path)
    metadata_task = task_metadata_analysis.delay(video_path)

    return {
        "analysis_id": analysis_id,
        "video_task_id": video_task.id,
        "audio_task_id": audio_task.id,
        "metadata_task_id": metadata_task.id
    }

@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    await websocket.accept()

    try:
        # Receive task IDs ONCE
        data = await websocket.receive_json()

        video_id = data["video_task_id"]
        audio_id = data["audio_task_id"]
        metadata_id = data["metadata_task_id"]

        video_result = AsyncResult(video_id, app=celery)
        audio_result = AsyncResult(audio_id, app=celery)
        metadata_result = AsyncResult(metadata_id, app=celery)

        sent_video = False
        sent_audio = False
        sent_meta = False

        while True:

            if video_result.ready() and not sent_video:
                await websocket.send_json({
                    "stage": "video_complete",
                    "result": video_result.result
                })
                sent_video = True

            if audio_result.ready() and not sent_audio:
                await websocket.send_json({
                    "stage": "audio_complete",
                    "result": audio_result.result
                })
                sent_audio = True

            if metadata_result.ready() and not sent_meta:
                await websocket.send_json({
                    "stage": "metadata_complete",
                    "result": metadata_result.result
                })
                sent_meta = True

            if sent_video and sent_audio and sent_meta:
                final = compute_final_score(
                    video_result.result,
                    audio_result.result,
                    metadata_result.result
                )

                await websocket.send_json({
                    "stage": "final",
                    "result": final
                })
                await websocket.close()
                break

            await asyncio.sleep(0.5)

    except Exception:
        await websocket.close()