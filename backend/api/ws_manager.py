from __future__ import annotations

import json
from collections import defaultdict

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self):
        self.connections: dict[WebSocket, set[str]] = {}
        self.channel_map: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        channels = self.connections.pop(websocket, set())
        for channel in channels:
            self.channel_map[channel].discard(websocket)

    def subscribe(self, websocket: WebSocket, channels: list[str]):
        active = self.connections.setdefault(websocket, set())
        for channel in channels:
            active.add(channel)
            self.channel_map[channel].add(websocket)

    async def send(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict, channels: list[str]):
        recipients: set[WebSocket] = set()
        for channel in channels:
            recipients.update(self.channel_map.get(channel, set()))
        dead = []
        for websocket in recipients:
            try:
                await self.send(websocket, message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)
