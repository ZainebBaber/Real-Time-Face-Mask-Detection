from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from contextlib import asynccontextmanager
import cv2
import asyncio
import numpy as np
from api.load_model import get_camera, process_frame
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app:FastAPI):
    # Start background task
    asyncio.create_task(summary_task())
    asyncio.create_task(video_broadcast_loop())
    yield  # The app runs


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static",StaticFiles(directory="static"),name="static")


cap = None

connected_clients = []
INTRUDER_THRESHOLD = 2 

# For AI summary counts
latest_masked = 0
latest_unmasked = 0

async def broadcast_bytes(frame_bytes: bytes):
    to_remove = []
    for client in connected_clients:
        try:
            await client.send_bytes(frame_bytes)
        except:
            to_remove.append(client)
    for client in to_remove:
        connected_clients.remove(client)


async def broadcast(message: str):
    to_remove = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except:
            to_remove.append(client)
    for client in to_remove:
        connected_clients.remove(client)


@app.get("/")
async def index(request: Request):
    global cap
    if cap is None:
        cap = get_camera()
        print("Camera opened on first page visit")
    return templates.TemplateResponse("index.html", {"request": request})



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)  # add client
    print(f"Client connected: {len(connected_clients)} clients active")



    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"Client disconnected: {len(connected_clients)} active clients")
            
async def video_broadcast_loop():
        global latest_masked, latest_unmasked

        try:
            previous_violation = False

            frame_count = 0

            last_mode = "NORMAL_MODE"


            while True:
                if cap is None:
                    await asyncio.sleep(1)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None or frame.sum() == 0:
                    await asyncio.sleep(0.01)
                    continue

                frame_count += 1
                 # Skip 2 out of every 3 frames
                if frame_count % 3 != 0:
                    continue


                frame, masked_count, unmasked_count = process_frame(frame)
                    
                latest_masked = masked_count
                latest_unmasked = unmasked_count

                # Resize frame for faster transmission (optional, improves FPS)
                frame = cv2.resize(frame, (320, 240))  # <- Resize here

                
                # Trigger beep alert broadcast
                current_violation = unmasked_count > 0

                # Trigger beep only when violation first appears
                if current_violation and not previous_violation:
                    await broadcast("beep")

                

                previous_violation = current_violation

                # Trigger RED alert if threshold exceeded
                if unmasked_count >= INTRUDER_THRESHOLD and last_mode!="HIGH_RISK_MODE":
                    await broadcast("HIGH_RISK_MODE")
                    last_mode="HIGH_RISK_MODE"

                else:
                    if last_mode != "NORMAL_MODE":
                        await broadcast("NORMAL_MODE")
                        last_mode = "NORMAL_MODE"
                        
                


                #  contiguous check is here, inside the loop
                frame = np.ascontiguousarray(frame) #converting into nupmy array

                #  correct syntax
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success:
                    await broadcast_bytes(buffer.tobytes())
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            print("Video broadcast loop cancelled")
            if cap:
                cap.release()

        
#background summary
async def summary_task():
    global latest_masked, latest_unmasked
    while True:
        await asyncio.sleep(5)
        summary_msg = (
            f"Summary (last 5 sec):\n"
            f"Masked: {latest_masked}\n"
            f"Unmasked: {latest_unmasked}"
        )
        await broadcast(summary_msg)
        # Reset counters for next interval
        latest_masked = 0
        latest_unmasked = 0


