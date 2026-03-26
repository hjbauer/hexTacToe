# Hex Self-Play Training System

Local FastAPI + vanilla JavaScript web app for experimenting with sparse-graph self-play training on an infinite-style hex board constrained by a distance-8 legal placement zone.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python /Users/hermannbauer/Desktop/codexHexTacToe/run_dev.py
```

Open [http://localhost:8000](http://localhost:8000).

## Environment Notes

- Use Python 3.11 or 3.12 for the ML stack. Python 3.13 is currently a poor fit for `torch-geometric` installs on CPU-only local setups.
- Use the same interpreter for both package install and app startup. If you install into `.venv`, launch with `python -m uvicorn ...` from that same activated environment.
- The dev runner only watches [`backend/`](/Users/hermannbauer/Desktop/codexHexTacToe/backend) and [`frontend/`](/Users/hermannbauer/Desktop/codexHexTacToe/frontend), so package installs inside `.venv` do not trigger endless reloads.
- The app shell and human-vs-AI fallback play can start without PyTorch installed, but training routes require `torch` and `torch-geometric`.

## Included

- Hex game state, legality checks, move application, and six-in-a-row win detection
- Sparse graph builder with color-swap augmentation support
- PyTorch Geometric GATv2 policy-value model
- Replay buffer, opponent pool, checkpointing, self-play training loop, and evaluation gating
- FastAPI REST and WebSocket APIs
- Frontend panels for training, evaluation, spectating, and human play

## Notes

- MCTS is stubbed and disabled by default.
- The frontend expects Chart.js from CDN but degrades if it is unavailable.
- Checkpoints are stored in [`checkpoints/`](/Users/hermannbauer/Desktop/codexHexTacToe/checkpoints).
