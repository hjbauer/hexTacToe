class HumanGamePanel {
  constructor(wsClient) {
    this.wsClient = wsClient;
    this.canvas = new HexCanvas(document.getElementById("playCanvas"));
    this.gameId = null;
    this.state = null;
    this.pendingTurnMoves = [];
    this.boundSocketOff = [];
    this.opponents = [];
    this.bindControls();
    this.bindSocket();
    this.fetchOpponents();
  }

  bindControls() {
    document.getElementById("newGameBtn").onclick = () => this.newGame();
    document.getElementById("opponentModelSelect").onchange = () => this.updateOpponentBadge();
    this.canvas.onHexClick((coord) => this.handleHexClick(coord));
  }

  bindSocket() {
    this.boundSocketOff.push(this.wsClient.on("game_move", (message) => {
      if (message.game_id !== this.gameId) {
        return;
      }
      this.fetchState();
    }));
    this.boundSocketOff.push(this.wsClient.on("game_error", (message) => {
      if (message.game_id === this.gameId) {
        document.getElementById("humanGameStatus").textContent = message.error;
      }
    }));
  }

  async newGame() {
    await this.fetchOpponents();
    const human_color = document.getElementById("humanColorSelect").value;
    const opponent_model = document.getElementById("opponentModelSelect").value;
    const response = await fetch("/api/game/new", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ human_color, opponent_model }),
    });
    const data = await response.json();
    this.gameId = data.game_id;
    this.wsClient.subscribe([`game:${this.gameId}`]);
    this.state = data.state;
    this.pendingTurnMoves = [];
    this.render();
  }

  async fetchOpponents() {
    const response = await fetch("/api/game/opponents");
    const data = await response.json();
    const select = document.getElementById("opponentModelSelect");
    const previousValue = select.value;
    this.opponents = data.opponents || [];
    select.innerHTML = "";
    this.opponents.forEach((opponent) => {
      const option = document.createElement("option");
      option.value = opponent.id;
      option.textContent = `${opponent.label} (${Math.round(opponent.elo || 0)})`;
      select.appendChild(option);
    });
    if (this.opponents.some((opponent) => opponent.id === previousValue)) {
      select.value = previousValue;
    } else if (this.opponents.length) {
      select.value = this.opponents[0].id;
    }
    this.updateOpponentBadge();
  }

  updateOpponentBadge() {
    const selectedId = document.getElementById("opponentModelSelect").value;
    const opponent = this.opponents.find((item) => item.id === selectedId);
    document.getElementById("opponentEloBadge").textContent = opponent
      ? `Opponent Elo: ${Math.round(opponent.elo || 0)}`
      : "Opponent Elo: -";
  }

  async fetchState() {
    if (!this.gameId) {
      return;
    }
    const response = await fetch(`/api/game/${this.gameId}/state`);
    const data = await response.json();
    this.state = data.state;
    this.pendingTurnMoves = [];
    this.render();
  }

  async handleHexClick(coord) {
    if (!this.gameId || !this.state || this.state.is_terminal) {
      return;
    }
    if (this.pendingTurnMoves.some(([q, r]) => q === coord[0] && r === coord[1])) {
      return;
    }
    this.pendingTurnMoves.push(coord);
    const required = this.state.placements_remaining_this_turn;
    if (this.pendingTurnMoves.length < required) {
      this.render();
      return;
    }

    const response = await fetch(`/api/game/${this.gameId}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hexes: this.pendingTurnMoves }),
    });
    if (response.ok) {
      const data = await response.json();
      this.state = data.state;
    } else {
      this.fetchState();
    }
    this.pendingTurnMoves = [];
    this.render();
  }

  render() {
    if (!this.state) {
      this.canvas.render({ red: [], blue: [] });
      return;
    }
    const pendingBoard = {
      red: [...this.state.red],
      blue: [...this.state.blue],
    };
    if (!this.state.is_terminal && this.pendingTurnMoves.length) {
      const target = this.state.current_player === "red" ? pendingBoard.red : pendingBoard.blue;
      this.pendingTurnMoves.forEach((coord) => target.push(coord));
    }
    this.canvas.render(pendingBoard);
    document.getElementById("humanGameStatus").textContent = this.state.is_terminal
      ? `Winner: ${this.state.winner || "draw"}`
      : `Game ${this.gameId} vs ${this.state.opponent_model_name} (${Math.round(this.state.opponent_model_elo || 0)})`;
    const pending = this.pendingTurnMoves.length
      ? ` | selected ${this.pendingTurnMoves.length}/${this.state.placements_remaining_this_turn}`
      : "";
    document.getElementById("humanTurnStatus").textContent =
      `Turn: ${this.state.current_player} | remaining ${this.state.placements_remaining_this_turn}${pending}`;
  }

  destroy() {
    this.boundSocketOff.forEach((off) => off());
    this.boundSocketOff = [];
    this.canvas.onHexClick(null);
  }
}
