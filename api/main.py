from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import asyncio
import numpy as np
from api.load_model import get_camera, process_frame

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = get_camera()

    if cap is None:
        await websocket.close()
        return  # ✅ also add return so it doesn't continue

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.sum() == 0:
                continue

            frame = process_frame(frame)

            if frame is None:
                continue

            # ✅ contiguous check is here, inside the loop
            frame = np.ascontiguousarray(frame)

            # ✅ correct syntax
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()