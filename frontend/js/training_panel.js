class TrainingPanel {
  constructor(wsClient, spectatePanel) {
    this.wsClient = wsClient;
    this.spectatePanel = spectatePanel;
    this.chart = null;
    this.chartPromise = null;
    this.statsOpen = false;
    this.boundSocketOff = [];

    this.headerTrainingState = document.getElementById("headerTrainingState");
    this.headerIteration = document.getElementById("headerIteration");
    this.phaseLabelEl = document.getElementById("trainingPhaseLabel");
    this.statusMessageEl = document.getElementById("trainingStatusMessage");
    this.trainingBadgeEl = document.getElementById("trainingBadge");
    this.progressFillEl = document.getElementById("trainingProgressFill");
    this.progressCaptionEl = document.getElementById("trainingProgressCaption");
    this.summaryEl = document.getElementById("trainingSummary");
    this.eventLogEl = document.getElementById("trainingEventLog");
    this.trainingStatsEl = document.getElementById("trainingStats");
    this.evalStatsEl = document.getElementById("evalStats");
    this.poolWinRatesEl = document.getElementById("poolWinRates");
    this.promotionHistoryEl = document.getElementById("promotionHistory");
    this.bufferDistributionEl = document.getElementById("bufferDistribution");
    this.leaderboardTableEl = document.getElementById("leaderboardTable");
    this.statsOverlay = document.getElementById("trainStatsOverlay");

    this.bindControls();
    this.bindSocket();
    this.fetchStatus();
    this.fetchCheckpoints();
  }

  bindControls() {
    document.getElementById("startTrainingBtn").onclick = () => this.post("/api/training/start");
    document.getElementById("stopTrainingBtn").onclick = () => this.post("/api/training/stop");
    document.getElementById("freshRunBtn").onclick = () => this.post("/api/training/reset");
    document.getElementById("saveCheckpointBtn").onclick = () => this.post("/api/training/checkpoint");
    document.getElementById("refreshCheckpointBtn").onclick = () => this.fetchCheckpoints();
    document.getElementById("loadCheckpointBtn").onclick = async () => {
      const path = document.getElementById("checkpointSelect").value;
      if (!path) {
        return;
      }
      await this.post("/api/training/load", { path });
    };

    document.getElementById("openStatsWindowBtn").onclick = () => this.openStats();
    document.getElementById("closeStatsWindowBtn").onclick = () => this.closeStats();
    document.getElementById("openSpectateWindowBtn").onclick = () => this.spectatePanel.open();
    this.statsOverlay.onclick = (event) => {
      if (event.target === this.statsOverlay) {
        this.closeStats();
      }
    };
  }

  bindSocket() {
    this.boundSocketOff.push(this.wsClient.on("training_status", (message) => {
      this.renderProgress(message);
    }));
    this.boundSocketOff.push(this.wsClient.on("checkpoint_saved", () => {
      this.fetchCheckpoints();
      if (this.statsOpen) {
        this.fetchStatus();
      }
    }));
    this.boundSocketOff.push(this.wsClient.on("eval_result", () => {
      if (this.statsOpen) {
        this.fetchStatus();
      }
    }));
    this.boundSocketOff.push(this.wsClient.on("model_promoted", () => {
      if (this.statsOpen) {
        this.fetchStatus();
      }
    }));
    this.wsClient.subscribe(["training"]);
  }

  async fetchStatus() {
    const response = await fetch("/api/training/status");
    const data = await response.json();
    this.renderProgress(data.training_status);
    if (this.statsOpen) {
      this.renderDetailedStats(
        data.training_status,
        data.eval_metrics,
        data.turn_number_distribution || [],
        data.leaderboard || []
      );
    }
  }

  async fetchCheckpoints() {
    const response = await fetch("/api/training/checkpoints");
    const data = await response.json();
    const select = document.getElementById("checkpointSelect");
    const previousValue = select.value;
    select.innerHTML = "";
    (data.checkpoints || []).forEach((path) => {
      const option = document.createElement("option");
      option.value = path;
      option.textContent = path.split("/").pop();
      select.appendChild(option);
    });
    if ((data.checkpoints || []).includes(previousValue)) {
      select.value = previousValue;
    } else if (data.checkpoints && data.checkpoints.length) {
      select.value = data.checkpoints[data.checkpoints.length - 1];
    }
  }

  async post(url, body) {
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    await this.fetchStatus();
    await this.fetchCheckpoints();
  }

  renderProgress(status) {
    const phase = status.current_phase || "idle";
    const progress = Number(status.phase_progress || 0);
    const total = Number(status.phase_total || 0);
    const percent = total > 0 ? Math.min(100, (progress / total) * 100) : 0;

    this.headerTrainingState.textContent = status.is_training ? "Training" : "Idle";
    this.headerIteration.textContent = `Iteration ${status.iteration ?? 0}`;
    this.phaseLabelEl.textContent = this.formatPhase(phase);
    this.statusMessageEl.textContent = status.status_message || "Waiting to start";
    this.trainingBadgeEl.textContent = status.is_training ? "Running" : "Stopped";
    this.progressFillEl.style.width = `${percent}%`;
    this.progressCaptionEl.textContent =
      total > 0 ? `${progress} of ${total} complete` : "No active work";

    const summary = [
      ["Iteration", status.iteration ?? 0],
      ["Games", status.games_played ?? 0],
      ["Population", status.population_size ?? 1],
      ["Replay Buffer", status.replay_buffer_size ?? 0],
      ["Blunder Rate", this.asPercent(status.forced_override_rate)],
      ["Leader", `${status.leader_model_name || "model-1"} (${this.formatElo(status.leader_model_elo)})`],
    ];
    this.summaryEl.innerHTML = summary
      .map(
        ([label, value]) =>
          `<div class="summary-card"><span class="label">${label}</span><span class="value">${value}</span></div>`
      )
      .join("");

    const events = status.recent_events || [];
    this.eventLogEl.innerHTML = this.renderRows(
      [...events].reverse(),
      "No activity yet",
      (row) => [row, ""],
      "event-row"
    );
  }

  async openStats() {
    this.statsOpen = true;
    this.statsOverlay.classList.remove("hidden");
    await this.fetchStatus();
  }

  closeStats() {
    this.statsOpen = false;
    this.statsOverlay.classList.add("hidden");
  }

  renderDetailedStats(status, metrics, distribution, leaderboard) {
    const trainingItems = [
      ["Episode", status.episode ?? 0],
      ["Policy Loss", this.formatFloat(status.loss_policy)],
      ["Value Loss", this.formatFloat(status.loss_value)],
      ["Temperature", this.formatFloat(status.current_temperature)],
      ["Opponent Pool", status.opponent_pool_size ?? 0],
      ["Population", status.population_size ?? 1],
      ["Leader", `${status.leader_model_name || "model-1"} (${this.formatElo(status.leader_model_elo)})`],
      ["Entropy Warning", status.entropy_warning ? "Yes" : "No"],
      ["Blunders", status.forced_override_count ?? 0],
      ["Model Decisions", status.model_decision_count ?? 0],
      ["Blunder Rate", this.asPercent(status.forced_override_rate)],
    ];
    this.trainingStatsEl.innerHTML = trainingItems
      .map(([label, value]) => this.statCard(label, value))
      .join("");

    const evalItems = [
      ["Last Eval", metrics.last_eval_iteration ?? 0],
      ["Win vs Ref", this.asPercent(metrics.win_rate_vs_reference)],
      ["Loss vs Ref", this.asPercent(metrics.loss_rate_vs_reference)],
      ["Win vs Random", this.asPercent(metrics.win_rate_vs_random)],
      ["Red Win Rate", this.asPercent(metrics.win_rate_as_red)],
      ["Blue Win Rate", this.asPercent(metrics.win_rate_as_blue)],
      ["Avg Length", this.formatFloat(metrics.avg_game_length)],
      ["Promoted", metrics.was_promoted ? "Yes" : "No"],
    ];
    this.evalStatsEl.innerHTML = evalItems.map(([label, value]) => this.statCard(label, value)).join("");
    this.poolWinRatesEl.innerHTML = this.renderRows(
      metrics.pool_win_rates || [],
      "No pool matches yet",
      (row) => [this.basename(row.checkpoint), this.asPercent(row.win_rate)]
    );
    this.promotionHistoryEl.innerHTML = this.renderRows(
      metrics.promotion_history || [],
      "No promotions yet",
      (row) => [`Iter ${row.iteration}`, this.basename(row.checkpoint)]
    );
    this.bufferDistributionEl.innerHTML = this.renderRows(
      distribution,
      "No buffer samples yet",
      (row) => [`Turn bucket ${row.bucket}`, row.count]
    );
    this.leaderboardTableEl.innerHTML = this.renderRows(
      leaderboard,
      "No league results yet",
      (row) => [
        `${row.name} | ${row.lineage} | ${row.role} | gen ${row.generation}`,
        `Elo ${this.formatElo(row.elo)} | ${row.wins}-${row.losses}-${row.draws}`
      ]
    );
    this.updateChart(status.loss_history || []);
  }

  updateChart(lossHistory) {
    this.ensureChart().then((ready) => {
      if (!ready || !this.chart) {
        return;
      }
      this.chart.data.labels = lossHistory.map((item) => item.iter);
      this.chart.data.datasets[0].data = lossHistory.map((item) => item.lp);
      this.chart.data.datasets[1].data = lossHistory.map((item) => item.lv);
      this.chart.update("none");
    });
  }

  async ensureChart() {
    if (!this.statsOpen) {
      return false;
    }
    if (this.chart) {
      return true;
    }
    if (!window.Chart && !this.chartPromise) {
      this.chartPromise = new Promise((resolve) => {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/chart.js";
        script.onload = () => resolve(true);
        script.onerror = () => resolve(false);
        document.head.appendChild(script);
      });
    }
    const ready = window.Chart ? true : await this.chartPromise;
    if (!ready || this.chart) {
      return ready;
    }
    const ctx = document.getElementById("lossChart").getContext("2d");
    this.chart = new window.Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "Policy", data: [], borderColor: "#c2472d", tension: 0.2, pointRadius: 0 },
          { label: "Value", data: [], borderColor: "#275d8b", tension: 0.2, pointRadius: 0 },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: { x: { display: false } },
      },
    });
    return true;
  }

  destroy() {
    this.boundSocketOff.forEach((off) => off());
    this.boundSocketOff = [];
    this.closeStats();
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  renderRows(rows, emptyText, renderRow, extraClass = "") {
    if (!rows.length) {
      return `<div class="table-row ${extraClass}"><span>${emptyText}</span></div>`;
    }
    return rows
      .map((row) => {
        const [left, right] = renderRow(row);
        return `<div class="table-row ${extraClass}"><span>${left}</span><span>${right}</span></div>`;
      })
      .join("");
  }

  statCard(label, value) {
    return `<div class="stat-card"><span class="label">${label}</span><span class="value">${value}</span></div>`;
  }

  formatPhase(phase) {
    return phase.replaceAll("_", " ");
  }

  basename(path) {
    return path ? path.split("/").pop() : "";
  }

  asPercent(value) {
    return `${((value || 0) * 100).toFixed(1)}%`;
  }

  formatFloat(value) {
    return Number(value || 0).toFixed(3);
  }

  formatElo(value) {
    return Math.round(Number(value || 0));
  }
}
