from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.auth import AUTH_COOKIE_NAME, AUTH_COOKIE_VALUE, AUTH_PASSWORD, is_request_authenticated

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/status")
async def auth_status(request: Request):
    return {"authenticated": is_request_authenticated(request)}


@router.post("/login")
async def login(request: Request):
    payload = await request.json()
    password = payload.get("password", "")
    if password != AUTH_PASSWORD:
        return JSONResponse({"ok": False, "authenticated": False}, status_code=401)
    response = JSONResponse({"ok": True, "authenticated": True})
    response.set_cookie(
        AUTH_COOKIE_NAME,
        AUTH_COOKIE_VALUE,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return response


@router.post("/logout")
async def logout():
    response = JSONResponse({"ok": True, "authenticated": False})
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response
