from __future__ import annotations

from fastapi import HTTPException, Request, WebSocket, status

AUTH_COOKIE_NAME = "hex_auth"
AUTH_COOKIE_VALUE = "ok"
AUTH_PASSWORD = "Dewey808"


def is_request_authenticated(request: Request) -> bool:
    return request.cookies.get(AUTH_COOKIE_NAME) == AUTH_COOKIE_VALUE


def is_websocket_authenticated(websocket: WebSocket) -> bool:
    return websocket.cookies.get(AUTH_COOKIE_NAME) == AUTH_COOKIE_VALUE


def require_auth(request: Request):
    if not is_request_authenticated(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
