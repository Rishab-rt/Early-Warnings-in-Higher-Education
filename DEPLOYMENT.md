# Deployment Guide

This covers the "Deployment UI" item from our team meeting: getting `app.py`
running behind a shareable link, in two ways. Neither option changes any
existing project file - the app runs exactly as it does locally.

## Option A: Streamlit Community Cloud (fastest, free, public link)

Best for sharing a live demo link in the presentation.

1. Push this repo to GitHub (already done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick this repo, branch `main`, and set the main file
   path to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically and
   picks up `.streamlit/config.toml` for settings.
5. Since the trained `.pkl` files aren't committed to git (by design, see
   README), add this as the app's **"Advanced settings" > startup command**,
   or simplest: keep `model.py`'s training step triggered automatically by
   adding a one-time check in a `packages.txt`/build step. In practice,
   the easiest fix is to run `python model.py` once via the Cloud app's
   terminal (Manage app > Terminal) right after the first deploy, then
   reboot the app.

## Option B: Docker (portable, runs anywhere - a server, a teammate's machine, etc.)

Best if we want a container we can hand off or deploy to any cloud host
(Render, Railway, AWS, etc.) later.

### Run locally with Docker Compose
```bash
docker compose up --build
```
Then open http://localhost:8501

### Or with plain Docker
```bash
docker build -t early-warnings-app .
docker run -p 8501:8501 early-warnings-app
```

The Dockerfile trains the models (`python model.py`) at build time, since
the `.pkl` artifacts are gitignored - so the image is self-contained and
works out of the box, no manual setup steps needed.

### Deploying the container to a free host (optional, for a public link)
Any of these accept a Dockerfile directly and give a public URL:
- [Render](https://render.com) - free tier, connect the GitHub repo, pick "Docker" as the environment.
- [Railway](https://railway.app) - similar flow, auto-detects the Dockerfile.

## Files added for this

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the app image, trains models at build time |
| `.dockerignore` | Keeps the image lean (excludes `.git`, `plots/`, etc.) |
| `docker-compose.yml` | One-command local run: `docker compose up --build` |
| `.streamlit/config.toml` | Server + theme settings used by both Docker and Streamlit Cloud |
| `DEPLOYMENT.md` | This file |
