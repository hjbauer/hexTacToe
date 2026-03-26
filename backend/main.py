from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_game import router as game_router
from backend.api.routes_training import router as training_router
from backend.api.ws_manager import WebSocketManager
from backend.state.app_state import AppState

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Hex Self-Play Training")
app.state.app_state = AppState()
app.state.app_state.ws_manager = WebSocketManager()

app.include_router(training_router)
app.include_router(game_router)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager: WebSocketManager = app.state.app_state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "subscribe":
                manager.subscribe(websocket, message.get("channels", []))
            elif message.get("type") == "ping":
                await manager.send(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
