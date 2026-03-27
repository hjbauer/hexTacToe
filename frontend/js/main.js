window.addEventListener("DOMContentLoaded", async () => {
  const authenticated = await ensureAuthenticated();
  if (!authenticated) {
    return;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsClient = new WSClient(`${protocol}://${window.location.host}/ws`);
  const params = new URLSearchParams(window.location.search);
  const requestedMode = params.get("mode");
  const startupMenu = document.getElementById("startupMenu");
  const trainWorkspace = document.getElementById("trainWorkspace");
  const playWorkspace = document.getElementById("playWorkspace");
  const heroMetrics = document.getElementById("heroMetrics");

  let trainingPanel = null;
  let spectatePanel = null;
  let humanGamePanel = null;

  const destroyTrainMode = () => {
    if (trainingPanel) {
      trainingPanel.destroy();
      trainingPanel = null;
    }
    if (spectatePanel) {
      spectatePanel.destroy();
      spectatePanel = null;
    }
  };

  const destroyPlayMode = () => {
    if (humanGamePanel) {
      humanGamePanel.destroy();
      humanGamePanel = null;
    }
  };

  const showMenu = () => {
    destroyTrainMode();
    destroyPlayMode();
    startupMenu.classList.remove("hidden");
    trainWorkspace.classList.add("hidden");
    playWorkspace.classList.add("hidden");
    heroMetrics.classList.add("hidden");
  };

  const showTrain = () => {
    destroyPlayMode();
    startupMenu.classList.add("hidden");
    trainWorkspace.classList.remove("hidden");
    playWorkspace.classList.add("hidden");
    heroMetrics.classList.remove("hidden");
    if (!spectatePanel) {
      spectatePanel = new SpectatePanel(wsClient);
    }
    if (!trainingPanel) {
      trainingPanel = new TrainingPanel(wsClient, spectatePanel);
    }
  };

  const showPlay = () => {
    destroyTrainMode();
    startupMenu.classList.add("hidden");
    trainWorkspace.classList.add("hidden");
    playWorkspace.classList.remove("hidden");
    heroMetrics.classList.add("hidden");
    if (!humanGamePanel) {
      humanGamePanel = new HumanGamePanel(wsClient);
    }
  };

  document.getElementById("openTrainMode").onclick = showTrain;
  document.getElementById("openPlayMode").onclick = showPlay;
  document.getElementById("backToMenuFromTrain").onclick = showMenu;
  document.getElementById("backToMenuFromPlay").onclick = showMenu;

  if (requestedMode === "train") {
    showTrain();
  } else if (requestedMode === "play") {
    showPlay();
  } else {
    showMenu();
  }
});
