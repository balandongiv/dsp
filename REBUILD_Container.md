## ✅ Rebuild the Codespace container (recommended)

### In VS Code (inside the Codespace)

1. Open **Command Palette**:

   - **Windows/Linux:** `Ctrl + Shift + P`
   - **Mac:** `Cmd + Shift + P`

2. Run one of these commands (depending on what you see):

   - **Codespaces: Rebuild Container**
   - **Dev Containers: Rebuild Container**
   - **Dev Containers: Rebuild and Reopen in Container**

This will recreate the container using your `.devcontainer` config.

---

## ✅ Rebuild “clean” (ignore cached layers)

If your container keeps breaking the same way, do a clean rebuild:

1. Command Palette → run:

   - **Dev Containers: Rebuild Container Without Cache**
     (or similar wording)

This forces a full rebuild from scratch (no Docker build cache).

---

## ✅ Reset EVERYTHING (delete the Codespace and recreate)

If you want a **full reinstall of the entire environment**, deleting the Codespace is the cleanest fix.

### From GitHub (web)

1. Go to: **GitHub → Code → Codespaces**
2. Find your Codespace → click **…** → **Delete**
3. Create a **new Codespace**

✅ This wipes containers, volumes, installed packages, caches—everything.

---

## ✅ If you have multiple containers (docker-compose in devcontainer)

Rebuild normally (above), or also clear containers from terminal:

```bash
docker ps -a
docker compose down -v
```

> `-v` removes volumes too (fresh database/cache/etc).

---

## Quick “which one should I use?”

- **Just broken tools/packages?** → **Rebuild Container**
- **Container rebuild doesn’t fix it?** → **Rebuild Without Cache**
- **Want 100% fresh start?** → **Delete Codespace + recreate**
