class WSClient {
  constructor(url) {
    this.url = url;
    this.handlers = new Map();
    this.pendingSubscriptions = new Set();
    this.connect();
  }

  connect() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener("open", () => {
      if (this.pendingSubscriptions.size) {
        this.subscribe([...this.pendingSubscriptions]);
      }
      this.pingTimer = setInterval(() => this.send({ type: "ping" }), 15000);
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const handlers = this.handlers.get(message.type) || new Set();
      handlers.forEach((handler) => handler(message));
    });
    this.socket.addEventListener("close", () => {
      clearInterval(this.pingTimer);
      setTimeout(() => this.connect(), 1000);
    });
  }

  send(message) {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  subscribe(channels) {
    channels.forEach((channel) => this.pendingSubscriptions.add(channel));
    this.send({ type: "subscribe", channels });
  }

  on(type, handler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type).add(handler);
    return () => this.off(type, handler);
  }

  off(type, handler) {
    const handlers = this.handlers.get(type);
    if (!handlers) {
      return;
    }
    handlers.delete(handler);
    if (!handlers.size) {
      this.handlers.delete(type);
    }
  }
}
