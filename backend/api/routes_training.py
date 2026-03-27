from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from backend.training.checkpoint import list_checkpoints
from backend.training.trainer import TrainingLoop

router = APIRouter(prefix="/api/training", tags=["training"])


def _require_trainer(app_state):
    if app_state.trainer is None:
        try:
            app_state.trainer = TrainingLoop(app_state)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return app_state.trainer


@router.post("/start")
async def start_training(request: Request):
    app_state = request.app.state.app_state
    _require_trainer(app_state)
    if app_state.training_task is None or app_state.training_task.done():
        app_state.training_task = asyncio.create_task(app_state.trainer.start())
    return {"ok": True, "status": app_state.training_status.to_dict()}


@router.post("/stop")
async def stop_training(request: Request):
    app_state = request.app.state.app_state
    if app_state.trainer is None:
        return {"ok": True, "status": app_state.training_status.to_dict()}
    await app_state.trainer.stop()
    if app_state.training_task is not None:
        try:
            await asyncio.wait_for(app_state.training_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    return {"ok": True, "status": app_state.training_status.to_dict()}


@router.get("/status")
async def get_training_status(request: Request):
    app_state = request.app.state.app_state
    return {
        "training_status": app_state.training_status.to_dict(include_history=True),
        "eval_metrics": app_state.eval_metrics.to_dict(),
        "leaderboard": (
            app_state.trainer.leaderboard()
            if app_state.trainer is not None
            else []
        ),
        "turn_number_distribution": (
            app_state.trainer.replay_buffer.turn_number_histogram()
            if app_state.trainer is not None
            else []
        ),
    }


@router.post("/checkpoint")
async def force_save_checkpoint(request: Request):
    app_state = request.app.state.app_state
    if app_state.trainer is None:
        raise HTTPException(status_code=400, detail="trainer not initialized")
    path = await asyncio.to_thread(app_state.trainer.save_checkpoint, False)
    return {"ok": True, "path": path}


@router.get("/checkpoints")
async def get_checkpoints(request: Request):
    return {"checkpoints": list_checkpoints()}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    app_state = request.app.state.app_state
    _require_trainer(app_state)
    return {"leaderboard": app_state.trainer.leaderboard()}


@router.get("/spectate")
async def get_current_spectate(request: Request):
    app_state = request.app.state.app_state
    return {
        "spectate": app_state.current_spectate,
        "games": sorted(app_state.spectate_games.values(), key=lambda game: game["game_id"]),
        "population_size": app_state.training_status.population_size,
    }


@router.post("/load")
async def load_checkpoint_route(request: Request):
    payload = await request.json()
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    app_state = request.app.state.app_state
    _require_trainer(app_state)
    await asyncio.to_thread(app_state.trainer.load_checkpoint, path)
    return {"ok": True, "status": app_state.training_status.to_dict()}


@router.post("/reset")
async def reset_training(request: Request):
    app_state = request.app.state.app_state
    if app_state.trainer is not None:
        await app_state.trainer.stop()
    if app_state.training_task is not None:
        try:
            await asyncio.wait_for(app_state.training_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        app_state.training_task = None
    _require_trainer(app_state)
    app_state.trainer.reset()
    app_state.current_spectate = None
    app_state.spectate_games = {}
    return {"ok": True, "status": app_state.training_status.to_dict()}
