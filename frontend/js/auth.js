async function ensureAuthenticated() {
  const check = await fetch("/api/auth/status", { credentials: "same-origin" });
  const status = await check.json();
  if (status.authenticated) {
    return true;
  }

  while (true) {
    const password = window.prompt("Enter password");
    if (password === null) {
      return false;
    }
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ password }),
    });
    if (response.ok) {
      return true;
    }
    window.alert("Wrong password.");
  }
}
