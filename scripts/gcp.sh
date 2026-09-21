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
Usage: $(basename "$0") <create|sync|ssh|stop|start|pull|delete>

  create   Start a Spot ${MACHINE} VM with a ${DISK_SIZE} boot disk (1x L4).
  sync     Copy this repository onto the VM, including local training images.
  ssh      Open an SSH session.
  stop     Stop the VM and keep the persistent disk.
  start    Start a stopped VM (after you stop it, or after Spot STOP preemption).
  pull     Copy outputs/ back to this machine.
  delete   Delete the VM and its boot disk.

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
case "${cmd}" in
  create | sync | ssh | stop | start | pull | delete) "${cmd}" ;;
  -h | --help | help | "") usage ;;
  *)
    usage >&2
    exit 1
    ;;
esac
