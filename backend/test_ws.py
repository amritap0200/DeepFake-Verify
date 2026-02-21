import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8000/ws/test123") as ws:

        await ws.send(json.dumps({
            "video_task_id": "c503650d-8f40-4abf-ba22-59fd6fe22d8c",
            "audio_task_id": "f280db81-1dcf-47d5-8835-88cb4be28e90",
            "metadata_task_id": "fe34746c-93ca-42ae-b63b-a12638a73ced"
        }))

        while True:
            try:
                message = await ws.recv()
                print(message)
            except websockets.exceptions.ConnectionClosedOK:
                print("WebSocket closed normally.")
                break

asyncio.run(test())