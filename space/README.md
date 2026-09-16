---
title: Hamlet Translation API
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Hamlet Translation API

Lightweight FastAPI service for Hamlet English/German comparison data.

Endpoints:

- `/`
- `/ui/comparison`
- `/hamlet/4x4-comparison`

Set `OPENAI_API_KEY` in Space secrets for LLM endpoints.
