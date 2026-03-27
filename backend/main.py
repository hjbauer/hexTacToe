from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_auth import router as auth_router
from backend.api.routes_game import router as game_router
from backend.api.routes_training import router as training_router
from backend.api.ws_manager import WebSocketManager
from backend.auth import is_websocket_authenticated
from backend.state.app_state import AppState

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Hex Self-Play Training")
app.state.app_state = AppState()
app.state.app_state.ws_manager = WebSocketManager()

app.include_router(auth_router)
app.include_router(training_router)
app.include_router(game_router)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/auth/"):
        from backend.auth import is_request_authenticated

        if not is_request_authenticated(request):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "authentication required"}, status_code=401)
    return await call_next(request)


@app.get("/")
async def index():
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not is_websocket_authenticated(websocket):
        await websocket.close(code=1008)
        return
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
