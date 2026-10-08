#!/usr/bin/env bash
# Render held-out eval prompts, then stop this Compute Engine VM so GPU billing ends.
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

step=""
force=0
status=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      if [[ $# -lt 2 || ! "${2}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Usage: $(basename "$0") --step <n> [--force]" >&2
        status=1
        break
      fi
      step="${2}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    *)
      echo "Usage: $(basename "$0") --step <n> [--force]" >&2
      status=1
      break
      ;;
  esac
done

echo "==== eval job start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) python=${PYTHON} step=${step} force=${force} ===="

run_eval_inference() {
  local output_dir="$1"
  local lora_path="${2:-}"
  local -a command
  command=("${PYTHON}" -m src.inference --config config.yaml --output-dir "${output_dir}")
  if [[ "${force}" -eq 0 ]]; then
    command+=(--reuse-existing)
  fi
  if [[ -n "${lora_path}" ]]; then
    command+=(--lora "${lora_path}")
  fi
  "${command[@]}"
}

weights=""
if [[ "${status}" -eq 0 && -z "${step}" ]]; then
  echo "Missing --step." >&2
  status=1
fi
if [[ "${status}" -eq 0 ]]; then
  weights="${ROOT}/outputs/checkpoints/checkpoint-${step}/pytorch_lora_weights.safetensors"
  if [[ ! -f "${weights}" ]]; then
    echo "Missing LoRA weights: ${weights}" >&2
    status=1
  fi
fi
if [[ "${status}" -eq 0 ]]; then
  run_eval_inference "outputs/eval/base" || status=$?
fi
if [[ "${status}" -eq 0 ]]; then
  run_eval_inference "outputs/eval/${step}" "outputs/checkpoints/checkpoint-${step}" || status=$?
fi
echo "==== eval renders exit ${status} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
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
