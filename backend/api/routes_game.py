from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request

from backend.game.game_state import GameState
from backend.game.rules import apply_moves
from backend.model.inference import ModelAgent
from backend.state.app_state import HumanGameSession
from backend.training.baselines import HeuristicBaseline
from backend.training.trainer import TrainingLoop

router = APIRouter(prefix="/api/game", tags=["game"])


def _ensure_trainer(app_state):
    if app_state.trainer is None:
        try:
            app_state.trainer = TrainingLoop(app_state)
        except RuntimeError:
            return None
    return app_state.trainer


def _select_ai_move(app_state, state: GameState, opponent_model_name: str):
    trainer = _ensure_trainer(app_state)
    if trainer is None:
        return HeuristicBaseline().select_move(state)
    try:
        _, _, agent = trainer.get_opponent_descriptor(opponent_model_name)
        if isinstance(agent, ModelAgent):
            return agent.select_move(state, temperature=0.1, greedy=True).coord
        return agent.select_move(state)
    except Exception:
        return HeuristicBaseline().select_move(state)


async def _play_ai_turn(request: Request, session: HumanGameSession):
    while (
        not session.state.is_terminal
        and session.ai_color is not None
        and session.state.current_player == session.ai_color
        and session.state.placements_remaining_this_turn > 0
    ):
        coord = await asyncio.to_thread(
            _select_ai_move,
            request.app.state.app_state,
            session.state,
            session.opponent_model_name,
        )
        session.state = apply_moves(session.state, [coord])
        await request.app.state.app_state.ws_manager.broadcast(
            {
                "type": "game_move",
                "game_id": session.game_id,
                "player": session.ai_color,
                "hexes": [list(coord)],
                "turn_number": session.state.turn_number,
                "is_terminal": session.state.is_terminal,
                "winner": session.state.winner,
            },
            channels=[f"game:{session.game_id}"],
        )


@router.post("/new")
async def new_game(request: Request):
    payload = await request.json()
    human_color = payload.get("human_color", "red")
    opponent_model = payload.get("opponent_model", "leader")
    if human_color not in {"red", "blue", "both"}:
        raise HTTPException(status_code=400, detail="human_color must be red, blue, or both")
    game_id = uuid.uuid4().hex[:8]
    ai_color = None if human_color == "both" else ("blue" if human_color == "red" else "red")
    trainer = _ensure_trainer(request.app.state.app_state)
    if trainer is None:
        opponent_name = "heuristic"
        opponent_elo = 1000.0
    else:
        opponent_name, opponent_elo, _ = trainer.get_opponent_descriptor(opponent_model)
    session = HumanGameSession(
        game_id=game_id,
        human_color=human_color,
        ai_color=ai_color,
        opponent_model_name=opponent_name,
        opponent_model_elo=opponent_elo,
        state=GameState(),
    )
    request.app.state.app_state.human_games[game_id] = session
    if session.ai_color == session.state.current_player:
        await _play_ai_turn(request, session)
    return session.to_dict()


@router.get("/opponents")
async def list_opponents(request: Request):
    trainer = _ensure_trainer(request.app.state.app_state)
    if trainer is None:
        return {
            "opponents": [
                {"id": "heuristic", "label": "Heuristic", "elo": 1000.0, "kind": "baseline"},
                {"id": "random", "label": "Random", "elo": 800.0, "kind": "baseline"},
            ]
        }
    return {"opponents": trainer.available_opponents()}


@router.post("/{game_id}/move")
async def submit_move(game_id: str, request: Request):
    payload = await request.json()
    hexes = payload.get("hexes")
    if not isinstance(hexes, list) or not hexes:
        raise HTTPException(status_code=400, detail="hexes must be a non-empty list")
    session = request.app.state.app_state.human_games.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="game not found")
    if session.human_color != "both" and session.state.current_player != session.human_color:
        raise HTTPException(status_code=400, detail="it is not the human's turn")

    try:
        coords = [tuple(map(int, item)) for item in hexes]
        moving_player = session.state.current_player
        session.state = apply_moves(session.state, coords)
    except ValueError as exc:
        await request.app.state.app_state.ws_manager.broadcast(
            {"type": "game_error", "game_id": game_id, "error": str(exc)},
            channels=[f"game:{game_id}"],
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await request.app.state.app_state.ws_manager.broadcast(
        {
            "type": "game_move",
            "game_id": game_id,
            "player": moving_player,
            "hexes": [list(coord) for coord in coords],
            "turn_number": session.state.turn_number,
            "is_terminal": session.state.is_terminal,
            "winner": session.state.winner,
        },
        channels=[f"game:{game_id}"],
    )
    await _play_ai_turn(request, session)
    return session.to_dict()


@router.get("/{game_id}/state")
async def get_game_state(game_id: str, request: Request):
    session = request.app.state.app_state.human_games.get(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="game not found")
    return session.to_dict()


@router.delete("/{game_id}")
async def abandon_game(game_id: str, request: Request):
    removed = request.app.state.app_state.human_games.pop(game_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="game not found")
    return {"ok": True}
