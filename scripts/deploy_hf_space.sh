#!/usr/bin/env bash
#
# Deploy the FastAPI backend to a Hugging Face Space (Docker SDK).
#
# HF Spaces free CPU tier = 16 GB RAM / 2 vCPU — enough for Whisper `medium`
# (and pyannote diarization if HF_TOKEN is set).
#
# Prerequisites:
#   1. Create a Space on huggingface.co: New → Space → SDK: Docker → Blank.
#   2. Have an HF access token with WRITE scope (huggingface.co/settings/tokens).
#
# Usage:
#   ./scripts/deploy_hf_space.sh <hf-username> <space-name>
#
# When git asks for credentials on push: username = your HF username,
# password = your HF write token (or run `huggingface-cli login` first).
#
set -euo pipefail

HF_USER="${1:?Usage: deploy_hf_space.sh <hf-username> <space-name>}"
SPACE="${2:?Usage: deploy_hf_space.sh <hf-username> <space-name>}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> Cloning Space repo..."
git clone "https://huggingface.co/spaces/${HF_USER}/${SPACE}" "$WORK"

echo ">> Copying API code (Dockerfile + app) into the Space..."
cp -r "$ROOT/api/." "$WORK/"

# Drop artefacts that must not ship
find "$WORK" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -f "$WORK/.env" 2>/dev/null || true

echo ">> Writing HF Space README (frontmatter: Docker SDK, port 8000)..."
cat > "$WORK/README.md" <<EOF
---
title: MTBank Speech Analytics
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# MTBank Speech Analytics — Live Demo (FastAPI backend)

Full project & documentation: https://github.com/${HF_USER}/MTBank-AI-Engineer-

- Swagger UI: \`/docs\`
- Health: \`/health\`
- Full analysis: \`POST /analyze\` (multipart \`file=<audio>\` or form \`url=...\`)
EOF

echo ">> Committing and pushing..."
cd "$WORK"
git add -A
git commit -m "Deploy MTBank Speech Analytics API to HF Spaces" || echo "(nothing to commit)"
git push

echo ""
echo ">> Done. Space is building:"
echo "   https://huggingface.co/spaces/${HF_USER}/${SPACE}"
echo ">> Live URL once built:"
echo "   https://${HF_USER}-${SPACE}.hf.space/docs"
echo ""
echo ">> Remember to set these in Space → Settings → Variables and secrets:"
echo "     OPENAI_API_KEY (secret), OPENAI_API_BASE_URL, LLM_MODEL,"
echo "     WHISPER_MODEL=medium, WHISPER_COMPUTE_TYPE=int8, WHISPER_LANGUAGE=ru"
echo "     HF_TOKEN (secret, optional — enables pyannote diarization)"
