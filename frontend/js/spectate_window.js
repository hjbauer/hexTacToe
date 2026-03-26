class SpectateWindow {
  constructor() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    this.wsClient = new WSClient(`${protocol}://${window.location.host}/ws`);
    this.gridEl = document.getElementById("spectateGrid");
    this.populationEl = document.getElementById("spectatePopulation");
    this.activeCountEl = document.getElementById("spectateActiveCount");
    this.countSelectEl = document.getElementById("spectateCountSelect");
    this.annotateToggleBtn = document.getElementById("spectateAnnotateToggleBtn");
    this.paintRedBtn = document.getElementById("spectatePaintRedBtn");
    this.paintBlueBtn = document.getElementById("spectatePaintBlueBtn");
    this.slots = [];
    this.maxSlots = 16;
    this.lastGames = [];
    this.lastPopulationSize = 0;
    this.pollTimer = null;
    this.resizeFrame = null;
    this.annotateEnabled = false;
    this.paintColor = "red";
    this.overlayByGameId = new Map();

    this.restoreSelection();
    this.bindControls();
    this.bindSocket();
    this.fetchCurrent();
    this.startPolling();
    window.addEventListener("resize", () => this.scheduleResize());
  }

  restoreSelection() {
    const stored = window.localStorage.getItem("spectate-count");
    if (stored && Array.from(this.countSelectEl.options).some((option) => option.value === stored)) {
      this.countSelectEl.value = stored;
    }
  }

  bindControls() {
    this.countSelectEl.onchange = () => {
      window.localStorage.setItem("spectate-count", this.countSelectEl.value);
      this.renderState(this.lastGames, this.lastPopulationSize);
    };
    this.annotateToggleBtn.onclick = () => {
      this.annotateEnabled = !this.annotateEnabled;
      this.syncToolbar();
    };
    this.paintRedBtn.onclick = () => {
      this.paintColor = "red";
      this.syncToolbar();
    };
    this.paintBlueBtn.onclick = () => {
      this.paintColor = "blue";
      this.syncToolbar();
    };
    document.getElementById("spectateClearMarksBtn").onclick = () => {
      this.overlayByGameId.clear();
      this.renderState(this.lastGames, this.lastPopulationSize);
    };
    document.getElementById("fitAllBoardsBtn").onclick = () => {
      this.slots.forEach((slot) => {
        if (!slot.hidden) {
          slot.canvas.fitToBoard(slot.currentBoard);
        }
      });
    };
    this.syncToolbar();
  }

  bindSocket() {
    this.wsClient.subscribe(["spectate"]);
    this.wsClient.on("spectate_state", (message) => {
      this.renderState(message.games || [], message.population_size || 0);
    });
  }

  async fetchCurrent() {
    const response = await fetch("/api/training/spectate");
    const data = await response.json();
    this.renderState(data.games || [], data.population_size || 0);
  }

  startPolling() {
    this.pollTimer = window.setInterval(() => this.fetchCurrent(), 1000);
  }

  renderState(games, populationSize) {
    this.lastGames = games;
    this.lastPopulationSize = populationSize;
    this.populationEl.textContent = `Population: ${populationSize || 0}`;
    this.activeCountEl.textContent = `Active Games: ${games.length || 0}`;

    const visibleCount = Math.max(1, Number(this.countSelectEl.value || 8));
    const visibleGames = games.slice(0, visibleCount);
    this.ensureSlots(visibleCount);

    visibleGames.forEach((game, index) => this.renderSlot(this.slots[index], game));
    for (let index = visibleGames.length; index < visibleCount; index += 1) {
      this.renderSlot(this.slots[index], null);
    }
    for (let index = visibleCount; index < this.slots.length; index += 1) {
      this.slots[index].root.classList.add("hidden");
      this.slots[index].hidden = true;
    }

    this.gridEl.style.setProperty("--spectate-columns", this.columnsForCount(visibleCount));
    this.scheduleResize();
  }

  columnsForCount(count) {
    if (count <= 1) {
      return 1;
    }
    if (count <= 4) {
      return 2;
    }
    if (count <= 6) {
      return 3;
    }
    return 4;
  }

  ensureSlots(count) {
    while (this.slots.length < Math.min(count, this.maxSlots)) {
      const slot = this.createSlot(this.slots.length);
      this.slots.push(slot);
      this.gridEl.appendChild(slot.root);
    }
    for (let index = 0; index < count; index += 1) {
      this.slots[index].root.classList.remove("hidden");
      this.slots[index].hidden = false;
    }
  }

  createSlot(index) {
    const root = document.createElement("article");
    root.className = "spectate-card";

    const header = document.createElement("div");
    header.className = "spectate-card-header";

    const title = document.createElement("strong");
    title.textContent = `Game ${index + 1}`;
    const candidate = document.createElement("span");
    candidate.textContent = "Candidate: waiting";
    const opponent = document.createElement("span");
    opponent.textContent = "Opponent: waiting";
    const move = document.createElement("span");
    move.textContent = "Move: 0 / 0";

    header.appendChild(title);
    header.appendChild(candidate);
    header.appendChild(opponent);
    header.appendChild(move);

    const frame = document.createElement("div");
    frame.className = "spectate-board-frame";
    const canvasEl = document.createElement("canvas");
    frame.appendChild(canvasEl);

    root.appendChild(header);
    root.appendChild(frame);

    const slot = {
      root,
      header,
      title,
      candidate,
      opponent,
      move,
      frame,
      canvasEl,
      canvas: new HexCanvas(canvasEl, { radius: 20, minRadius: 5, maxRadius: 64 }),
      currentGameId: null,
      baseBoard: { red: [], blue: [] },
      currentBoard: { red: [], blue: [] },
      hidden: false,
    };
    slot.canvas.onHexClick((coord) => this.handleBoardClick(slot, coord));
    return slot;
  }

  handleBoardClick(slot, coord) {
    if (!this.annotateEnabled || !slot.currentGameId) {
      return;
    }
    const overlay = this.overlayFor(slot.currentGameId);
    if (this.boardHasCoord(overlay, coord)) {
      overlay.red = overlay.red.filter(([q, r]) => q !== coord[0] || r !== coord[1]);
      overlay.blue = overlay.blue.filter(([q, r]) => q !== coord[0] || r !== coord[1]);
    } else if (this.boardHasCoord(slot.baseBoard, coord)) {
      return;
    } else {
      overlay[this.paintColor].push(coord);
    }
    this.overlayByGameId.set(slot.currentGameId, overlay);
    this.renderSlot(slot, this.lastGames.find((game) => game.game_id === slot.currentGameId) || null);
  }

  overlayFor(gameId) {
    if (!this.overlayByGameId.has(gameId)) {
      this.overlayByGameId.set(gameId, { red: [], blue: [] });
    }
    return this.overlayByGameId.get(gameId);
  }

  boardHasCoord(board, coord) {
    return [...(board.red || []), ...(board.blue || [])].some(
      ([q, r]) => q === coord[0] && r === coord[1]
    );
  }

  mergedBoard(baseBoard, overlayBoard) {
    return {
      red: [...(baseBoard.red || []), ...(overlayBoard.red || [])],
      blue: [...(baseBoard.blue || []), ...(overlayBoard.blue || [])],
    };
  }

  syncToolbar() {
    this.annotateToggleBtn.textContent = this.annotateEnabled ? "Annotate On" : "Annotate Off";
    this.annotateToggleBtn.classList.toggle("active-toggle", this.annotateEnabled);
    this.paintRedBtn.classList.toggle("active-toggle", this.paintColor === "red");
    this.paintBlueBtn.classList.toggle("active-toggle", this.paintColor === "blue");
  }

  renderSlot(slot, game) {
    if (!game) {
      slot.currentGameId = null;
      slot.currentBoard = { red: [], blue: [] };
      slot.title.textContent = "Game: waiting";
      slot.candidate.textContent = "Candidate: waiting";
      slot.opponent.textContent = "Opponent: waiting";
      slot.move.textContent = "Move: 0 / 0";
      slot.root.classList.remove("spectate-terminal");
      slot.canvas.render(slot.currentBoard, { message: "Waiting for game" });
      return;
    }

    const isNewGame = slot.currentGameId !== game.game_id;
    slot.currentGameId = game.game_id;
    const baseBoard = game.board_snapshot || { red: [], blue: [] };
    slot.baseBoard = baseBoard;
    const overlayBoard = this.overlayFor(game.game_id);
    slot.currentBoard = this.mergedBoard(baseBoard, overlayBoard);
    slot.title.textContent = `Game: ${game.game_id}`;
    slot.candidate.textContent =
      `Candidate: ${game.candidate_model || "unknown"} (${game.candidate_color || "?"})`;
    slot.opponent.textContent =
      `Opponent: ${game.opponent_type || "unknown"} (${game.opponent_color || "?"})`;
    slot.move.textContent = `Move: ${game.move_count || 0} / ${game.max_turns || 0}`;
    slot.root.classList.toggle("spectate-terminal", Boolean(game.is_terminal));
    slot.canvas.render(
      slot.currentBoard,
      game.is_terminal ? { message: `Winner: ${game.winner || "draw"}` } : {}
    );
    if (isNewGame && (!slot.currentBoard.red.length && !slot.currentBoard.blue.length)) {
      slot.canvas.render(slot.currentBoard, { message: "Opening..." });
    }
  }

  scheduleResize() {
    if (this.resizeFrame !== null) {
      window.cancelAnimationFrame(this.resizeFrame);
    }
    this.resizeFrame = window.requestAnimationFrame(() => {
      this.resizeFrame = null;
      this.slots.forEach((slot) => {
        if (slot.hidden) {
          return;
        }
        const width = Math.max(Math.floor(slot.frame.clientWidth), 240);
        const height = Math.max(Math.floor(slot.frame.clientHeight), 240);
        slot.canvas.resize(width, height);
      });
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new SpectateWindow();
});
