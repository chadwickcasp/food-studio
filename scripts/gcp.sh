#!/usr/bin/env bash
# Create a Compute Engine Spot VM (g2-standard-8 / 1x L4) and sync this repo onto it.
# No Vertex AI or extra orchestration: gcloud create + tar sync, as in docs/flux2_klein_lora_mvp.md.

PROJECT="${GCP_PROJECT:-food-studio-509220}"
ZONE="${GCP_ZONE:-us-central1-a}"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE="${GCP_INSTANCE:-food-studio}"
MACHINE="${GCP_MACHINE:-g2-standard-8}"
DISK_SIZE="${GCP_DISK_SIZE:-100GB}"
REMOTE_DIR="${GCP_REMOTE_DIR:-food-studio}"
IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <create|sync|ssh|train|stop|start|pull|delete>

  create   Start a Spot ${MACHINE} VM with a ${DISK_SIZE} boot disk (1x L4).
  sync     Copy this repository onto the VM, including local training images.
  ssh      Open an SSH session.
  train    Detach baseline, training, and 50-step dev renders; then stop the VM.
  stop     Stop the VM and keep the persistent disk.
  start    Start a stopped VM (after you stop it, or after Spot STOP preemption).
  pull     Copy outputs/ back to this machine.
  delete   Delete the VM and its boot disk.

  train [--resume latest|<checkpoint-dir>]

Training preserves the held-out Base baseline, renders development prompts for
Base and every saved checkpoint, then stops the VM on success or failure so the
L4 does not keep billing.
The boot disk is kept. Start the VM later to inspect outputs/train.log or pull
checkpoints. For an interactive run that leaves the VM up, use ssh instead.

Edit PROJECT and ZONE at the top of this script, or override with GCP_PROJECT / GCP_ZONE.
Other environment: GCP_INSTANCE GCP_MACHINE GCP_DISK_SIZE
Pick a zone with L4 Spot capacity if create fails.
EOF
}

need_project() {
  if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
    echo "Set PROJECT at the top of this script, or export GCP_PROJECT" >&2
    exit 1
  fi
}

gcloud_vm() {
  gcloud compute instances "$@" --project="${PROJECT}" --zone="${ZONE}"
}

ssh_cmd() {
  gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" "$@"
}

create() {
  need_project
  gcloud compute instances create "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE}" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure \
    --boot-disk-size="${DISK_SIZE}" \
    --boot-disk-type=pd-balanced \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --metadata=install-nvidia-driver=True \
    --scopes=cloud-platform
  echo
  echo "VM created. First boot may install the NVIDIA driver and reboot."
  echo "Next: $(basename "$0") sync && $(basename "$0") ssh"
}

sync() {
  need_project
  ssh_cmd --command="mkdir -p \${HOME}/${REMOTE_DIR}"
  tar -C "${ROOT}" \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    -czf - . \
    | gcloud compute ssh "${INSTANCE}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      --command="tar -xzf - -C \${HOME}/${REMOTE_DIR}"
  echo "Synced ${ROOT} -> ${INSTANCE}:~/${REMOTE_DIR}"
}

ssh() {
  need_project
  gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}"
}

train() {
  need_project
  local resume_arg=""
  case "${1:-}" in
    "")
      ;;
    --resume)
      if [[ -z "${2:-}" || $# -ne 2 ]]; then
        echo "Usage: $(basename "$0") train [--resume latest|<checkpoint-dir>]" >&2
        exit 1
      fi
      if [[ ! "${2}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "Unsafe --resume value: ${2}" >&2
        exit 1
      fi
      resume_arg="--resume ${2}"
      ;;
    *)
      echo "Usage: $(basename "$0") train [--resume latest|<checkpoint-dir>]" >&2
      exit 1
      ;;
  esac

  ssh_cmd --command="bash -lc \"mkdir -p \\\$HOME/${REMOTE_DIR}/outputs && cd \\\$HOME/${REMOTE_DIR} && nohup bash scripts/train_then_stop.sh ${resume_arg} >> outputs/train.log 2>&1 < /dev/null & echo Detached training pid \\\$! && echo Log: \\\$HOME/${REMOTE_DIR}/outputs/train.log && echo The VM will stop when training exits.\""
}

stop() {
  need_project
  gcloud_vm stop "${INSTANCE}"
}

start() {
  need_project
  gcloud_vm start "${INSTANCE}"
}

pull() {
  need_project
  mkdir -p "${ROOT}/outputs"
  gcloud compute scp --recurse \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${INSTANCE}:~/${REMOTE_DIR}/outputs/." \
    "${ROOT}/outputs/"
  echo "Copied VM outputs/ -> ${ROOT}/outputs/"
}

delete() {
  need_project
  gcloud_vm delete "${INSTANCE}"
}

cmd="${1:-}"
shift || true
case "${cmd}" in
  create | sync | ssh | stop | start | pull | delete) "${cmd}" ;;
  train) train "$@" ;;
  -h | --help | help | "") usage ;;
  *)
    usage >&2
    exit 1
    ;;
esac
