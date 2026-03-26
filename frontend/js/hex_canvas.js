class HexCanvas {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.radius = options.radius || 20;
    this.minRadius = options.minRadius || 8;
    this.maxRadius = options.maxRadius || 56;
    this.origin = {
      x: canvas.width / 2,
      y: canvas.height / 2,
    };
    this.lastBoardSnapshot = { red: [], blue: [] };
    this.lastOverlay = {};
    this.clickHandler = null;
    this.dragState = null;
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    canvas.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    canvas.addEventListener("pointerup", (event) => this.handlePointerUp(event));
    canvas.addEventListener("pointercancel", () => this.handlePointerCancel());
    canvas.addEventListener("wheel", (event) => this.handleWheel(event), { passive: false });
  }

  axialToPixel(q, r) {
    const x = this.origin.x + this.radius * Math.sqrt(3) * (q + r / 2);
    const y = this.origin.y + this.radius * 1.5 * r;
    return { x, y };
  }

  pixelToHex(x, y) {
    const q = ((x - this.origin.x) * Math.sqrt(3) / 3 - (y - this.origin.y) / 3) / this.radius;
    const r = ((y - this.origin.y) * 2 / 3) / this.radius;
    return this.roundHex(q, r);
  }

  roundHex(q, r) {
    let x = q;
    let z = r;
    let y = -x - z;

    let rx = Math.round(x);
    let ry = Math.round(y);
    let rz = Math.round(z);

    const xDiff = Math.abs(rx - x);
    const yDiff = Math.abs(ry - y);
    const zDiff = Math.abs(rz - z);

    if (xDiff > yDiff && xDiff > zDiff) {
      rx = -ry - rz;
    } else if (yDiff > zDiff) {
      ry = -rx - rz;
    } else {
      rz = -rx - ry;
    }
    return [rx, rz];
  }

  drawHex(q, r, fill, stroke = "rgba(32,49,37,0.12)") {
    const { x, y } = this.axialToPixel(q, r);
    this.ctx.beginPath();
    for (let index = 0; index < 6; index += 1) {
      const angle = ((60 * index - 30) * Math.PI) / 180;
      const px = x + this.radius * Math.cos(angle);
      const py = y + this.radius * Math.sin(angle);
      if (index === 0) {
        this.ctx.moveTo(px, py);
      } else {
        this.ctx.lineTo(px, py);
      }
    }
    this.ctx.closePath();
    this.ctx.fillStyle = fill;
    this.ctx.fill();
    this.ctx.strokeStyle = stroke;
    this.ctx.stroke();
  }

  setRadius(radius) {
    this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, radius));
    this.render(this.lastBoardSnapshot, this.lastOverlay);
  }

  zoomBy(delta) {
    this.setRadius(this.radius + delta);
  }

  zoomAt(clientX, clientY, scale) {
    const rect = this.canvas.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * this.canvas.width;
    const y = ((clientY - rect.top) / rect.height) * this.canvas.height;
    const qBefore = ((x - this.origin.x) * Math.sqrt(3) / 3 - (y - this.origin.y) / 3) / this.radius;
    const rBefore = ((y - this.origin.y) * 2 / 3) / this.radius;
    const nextRadius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius * scale));
    this.radius = nextRadius;
    this.origin = {
      x: x - (Math.sqrt(3) * this.radius * (qBefore + rBefore / 2)),
      y: y - (1.5 * this.radius * rBefore),
    };
    this.render(this.lastBoardSnapshot, this.lastOverlay);
  }

  panBy(dx, dy) {
    this.origin = {
      x: this.origin.x + dx,
      y: this.origin.y + dy,
    };
    this.render(this.lastBoardSnapshot, this.lastOverlay);
  }

  resize(width, height) {
    const previousWidth = this.canvas.width || width;
    const previousHeight = this.canvas.height || height;
    const originXRatio = previousWidth ? this.origin.x / previousWidth : 0.5;
    const originYRatio = previousHeight ? this.origin.y / previousHeight : 0.5;
    this.canvas.width = width;
    this.canvas.height = height;
    this.origin = {
      x: width * originXRatio,
      y: height * originYRatio,
    };
    this.render(this.lastBoardSnapshot, this.lastOverlay);
  }

  fitToBoard(boardSnapshot = this.lastBoardSnapshot, padding = 80) {
    const all = [...boardSnapshot.red, ...boardSnapshot.blue];
    const bounds = this.getBounds(all);
    const widthCells = Math.max(bounds.maxQ - bounds.minQ + 1, 1);
    const heightCells = Math.max(bounds.maxR - bounds.minR + 1, 1);
    const maxRadiusX = (this.canvas.width - padding * 2) / (Math.sqrt(3) * (widthCells + 1));
    const maxRadiusY = (this.canvas.height - padding * 2) / (1.5 * (heightCells + 1));
    this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, maxRadiusX, maxRadiusY));
    const centerQ = (bounds.minQ + bounds.maxQ) / 2;
    const centerR = (bounds.minR + bounds.maxR) / 2;
    this.origin = {
      x: (this.canvas.width / 2) - (this.radius * Math.sqrt(3) * (centerQ + centerR / 2)),
      y: (this.canvas.height / 2) - (this.radius * 1.5 * centerR),
    };
    this.render(boardSnapshot, this.lastOverlay);
  }

  render(boardSnapshot = { red: [], blue: [] }, overlay = {}) {
    this.lastBoardSnapshot = boardSnapshot;
    this.lastOverlay = overlay;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const all = [...boardSnapshot.red, ...boardSnapshot.blue];
    const bounds = this.getBounds(all);
    for (let q = bounds.minQ; q <= bounds.maxQ; q += 1) {
      for (let r = bounds.minR; r <= bounds.maxR; r += 1) {
        this.drawHex(q, r, "rgba(255,255,255,0.34)");
      }
    }
    boardSnapshot.red.forEach(([q, r]) => this.drawHex(q, r, "#c2472d"));
    boardSnapshot.blue.forEach(([q, r]) => this.drawHex(q, r, "#275d8b"));

    if (overlay.message) {
      this.ctx.fillStyle = "rgba(32,49,37,0.88)";
      this.ctx.font = "bold 24px Avenir Next";
      this.ctx.fillText(overlay.message, 24, 36);
    }
  }

  getBounds(coords) {
    if (!coords.length) {
      return { minQ: -4, maxQ: 4, minR: -4, maxR: 4 };
    }
    const qs = coords.map(([q]) => q);
    const rs = coords.map(([, r]) => r);
    return {
      minQ: Math.min(...qs) - 2,
      maxQ: Math.max(...qs) + 2,
      minR: Math.min(...rs) - 2,
      maxR: Math.max(...rs) + 2,
    };
  }

  onHexClick(handler) {
    this.clickHandler = handler;
  }

  handlePointerDown(event) {
    this.dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      moved: false,
    };
    this.canvas.setPointerCapture(event.pointerId);
  }

  handlePointerMove(event) {
    if (!this.dragState || event.pointerId !== this.dragState.pointerId) {
      return;
    }
    const dx = event.clientX - this.dragState.lastX;
    const dy = event.clientY - this.dragState.lastY;
    const distance = Math.abs(event.clientX - this.dragState.startX) + Math.abs(event.clientY - this.dragState.startY);
    if (distance > 4) {
      this.dragState.moved = true;
    }
    this.dragState.lastX = event.clientX;
    this.dragState.lastY = event.clientY;
    if (this.dragState.moved) {
      this.panBy(dx * (this.canvas.width / this.canvas.getBoundingClientRect().width), dy * (this.canvas.height / this.canvas.getBoundingClientRect().height));
    }
  }

  handlePointerUp(event) {
    if (!this.dragState || event.pointerId !== this.dragState.pointerId) {
      return;
    }
    const shouldClick = !this.dragState.moved && this.clickHandler;
    this.canvas.releasePointerCapture(event.pointerId);
    this.dragState = null;
    if (!shouldClick) {
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * this.canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * this.canvas.height;
    const coord = this.pixelToHex(x, y);
    this.clickHandler(coord);
  }

  handlePointerCancel() {
    this.dragState = null;
  }

  handleWheel(event) {
    event.preventDefault();
    if (event.ctrlKey || event.metaKey) {
      const scale = Math.exp(-event.deltaY * 0.0025);
      this.zoomAt(event.clientX, event.clientY, scale);
      return;
    }
    this.panBy(-event.deltaX, -event.deltaY);
  }
}
