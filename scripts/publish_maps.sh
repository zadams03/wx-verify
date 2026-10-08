#!/usr/bin/env bash
# Publish data/dashboard/maps/ as assets of a GitHub Release (SPEC 4.4 item 5).
#
#   scripts/publish_maps.sh
#
# Reads the release tag and repository from config.yaml (dashboard.maps_release_tag,
# dashboard.maps_repo), checks the committed manifest against the files on disk, prints
# what it will do and asks before uploading. Needs the GitHub CLI (`gh auth login`).
# Run by hand; the pipeline never calls it.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python
[ -x venv/bin/python ] && PY=venv/bin/python

read -r TAG REPO < <($PY -c "
from src.utils.config import load_config
d = load_config().dashboard
print(d.maps_release_tag, d.maps_repo)")

command -v gh >/dev/null || { echo "gh (GitHub CLI) is not installed"; exit 1; }

# Rebuild the manifest from the files on disk so it cannot disagree with what is uploaded.
$PY -c "
from src.data.map_store import build_manifest
from src.utils.config import load_config
m = build_manifest(load_config())
print(f\"manifest: {len(m['files'])} files, {m['total_bytes']/1e6:.1f} MB, release {m['release_tag']}\")"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
$PY - "$STAGE" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
stage = Path(sys.argv[1])
manifest = json.loads(Path("data/dashboard/maps_manifest.json").read_text())
for f in manifest["files"]:
    shutil.copy2(Path("data/dashboard/maps") / f["path"], stage / f["asset"])
PYEOF

COUNT=$(ls "$STAGE" | wc -l | tr -d ' ')
SIZE=$(du -sh "$STAGE" | cut -f1)
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  ACTION="UPLOAD $COUNT files ($SIZE) to the EXISTING release $TAG, replacing assets with the same name"
else
  ACTION="CREATE release $TAG on $REPO and upload $COUNT files ($SIZE)"
fi
echo
echo "About to: $ACTION"
read -r -p "Continue? [y/N] " ANSWER
[ "$ANSWER" = "y" ] || { echo "Aborted."; exit 1; }

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "$STAGE"/*.nc --repo "$REPO" --clobber
else
  gh release create "$TAG" "$STAGE"/*.nc --repo "$REPO" \
    --title "Map data $TAG" \
    --notes "Yearly grid-point statistics for the wx-verify spatial maps. Files are listed, with SHA-256 checksums, in data/dashboard/maps_manifest.json; the dashboard downloads them on first use."
fi
echo "Done. Commit data/dashboard/maps_manifest.json (it names this release) and push."
