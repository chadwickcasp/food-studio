#!/usr/bin/env bash
# Render development prompts, then stop this Compute Engine VM so GPU billing ends.
# Keeps the boot disk (same as scripts/gcp.sh stop). Safe to run off-GCP: it
# renders and leaves the machine running if metadata is not available.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

if command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi

echo "==== development job start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) python=${PYTHON} args=${*} ===="
status=0
"${PYTHON}" -m src.development --config config.yaml "$@" || status=$?
echo "==== development renders exit ${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
sync

# Stops this VM the same way as scripts/train_then_stop.sh.
stop_this_vm() {
  local meta="http://metadata.google.internal/computeMetadata/v1"
  local name zone project
  if ! name="$(curl -sf --max-time 2 -H "Metadata-Flavor: Google" "${meta}/instance/name")"; then
    echo "Not running on Compute Engine; leaving this machine up."
    return 0
  fi
  zone="$(curl -sf -H "Metadata-Flavor: Google" "${meta}/instance/zone")"
  zone="${zone##*/}"
  project="$(curl -sf -H "Metadata-Flavor: Google" "${meta}/project/project-id")"
  echo "Stopping Compute Engine instance ${name} (zone=${zone}) to end GPU billing."
  if gcloud compute instances stop "${name}" --project="${project}" --zone="${zone}" --quiet; then
    return 0
  fi
  echo "gcloud stop failed; trying sudo shutdown -h now." >&2
  sudo -n shutdown -h now
}

stop_this_vm
exit "${status}"
