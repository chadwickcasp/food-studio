#!/usr/bin/env bash
# Run the LoRA trainer, then stop this Compute Engine VM so GPU billing ends.
# Keeps the boot disk (same as scripts/gcp.sh stop). Safe to run off-GCP: it
# will train and leave the machine running if metadata is not available.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

echo "==== train start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) ===="
status=0
if [[ ! -f "${ROOT}/outputs/samples/baseline/manifest.json" ]]; then
  echo "==== held-out Base baseline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  python -m src.inference --config config.yaml || status=$?
  echo "==== held-out Base baseline exit ${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
else
  echo "Reusing preserved held-out Base baseline manifest."
fi
if [[ "${status}" -eq 0 ]]; then
  python -m src.train --config config.yaml "$@" || status=$?
fi
echo "==== train exit ${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
if [[ "${status}" -eq 0 ]]; then
  echo "==== development renders start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  python -m src.development --config config.yaml || status=$?
  echo "==== development renders exit ${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
fi
sync

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
