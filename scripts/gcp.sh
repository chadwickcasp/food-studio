#!/usr/bin/env bash
# Create a Compute Engine Spot GPU VM and sync this repo onto it.
# No Vertex AI or extra orchestration: gcloud create + tar sync, as in docs/flux2_klein_lora_mvp.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${GCP_PROJECT:-food-studio-509220}"
PROFILE="${GCP_PROFILE:-}"
if [[ -z "${PROFILE}" ]]; then
  case "${GCP_MACHINE:-}" in
    g4-*) PROFILE=g4 ;;
    *) PROFILE=l4 ;;
  esac
fi
DISK_SIZE="${GCP_DISK_SIZE:-100GB}"
REMOTE_DIR="${GCP_REMOTE_DIR:-food-studio}"
IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

configure_profile() {
  local default_zone default_instance default_machine default_disk_type
  case "${PROFILE}" in
    l4)
      default_zone=us-central1-a
      default_instance=food-studio
      default_machine=g2-standard-8
      default_disk_type=pd-balanced
      ;;
    g4)
      default_zone=us-central1-b
      default_instance=food-studio-g4
      default_machine=g4-standard-48
      default_disk_type=hyperdisk-balanced
      ;;
    *)
      echo "Unknown GCP profile: ${PROFILE}. Choose l4 or g4." >&2
      return 1
      ;;
  esac
  ZONE="${GCP_ZONE:-${default_zone}}"
  INSTANCE="${GCP_INSTANCE:-${default_instance}}"
  MACHINE="${GCP_MACHINE:-${default_machine}}"
  DISK_TYPE="${GCP_DISK_TYPE:-${default_disk_type}}"
}

configure_profile

usage() {
  cat <<EOF
Usage: $(basename "$0") [g4] <create|sync|sync-inference|push-lora|ssh|train|watch|stop|start|pull|delete>

  g4       Select the separate G4 instance (RTX PRO 6000, 96 GB VRAM).
  create   Create a Spot ${MACHINE} VM with a ${DISK_SIZE} ${DISK_TYPE} boot disk.
  sync     Copy this repository onto the VM, including local training images.
           Prints pack/upload/extract progress; skips venvs, UI build trees,
           and generated outputs.
  sync-inference
           Copy only inference code, config, requirements, and prompt files.
  push-lora [path]
           Copy one local LoRA weight file or checkpoint directory to the VM.
           Defaults to outputs/checkpoints/final/pytorch_lora_weights.safetensors.
  ssh      Open an SSH session.
  train    Detach baseline, training, and 50-step dev renders; then stop the VM.
           Stays attached. If the host preempts or kills the VM before the job
           finishes, start it again and resume from the latest checkpoint.
  watch    Stream GPU, process, and outputs/train.log until you Ctrl-C.
  stop     Stop the VM and keep the persistent disk.
  start    Start a stopped VM (after you stop it, or after Spot STOP preemption).
           If the VM is still STOPPING, wait until it is fully stopped first.
           If the zone has no GPU capacity, retry until one is free. Ctrl-C stops waiting.
  pull     Copy outputs/ back to this machine.
  delete   Delete the VM and its boot disk.

  train [--resume latest|<checkpoint-dir>]

Training preserves the held-out Base baseline, renders development prompts for
Base and every saved checkpoint, then stops the VM on success or failure so the
GPU does not keep billing.
The boot disk is kept. Start the VM later to inspect outputs/train.log or pull
checkpoints. For an interactive run that leaves the VM up, use ssh instead.

train stays in the foreground after detaching the job. When Compute Engine
preempts the Spot VM, or a host error stops it, before that job finishes, train
starts the instance again and resumes from the latest checkpoint. A stop made
by the training script or by stop is left stopped. Ctrl-C ends this watcher
and leaves the VM as it is.

Default: L4 in us-central1-a. G4: RTX PRO 6000 in us-central1-b with Hyperdisk.
Override with GCP_PROJECT, GCP_PROFILE, GCP_ZONE, GCP_INSTANCE, GCP_MACHINE,
GCP_DISK_TYPE, GCP_DISK_SIZE, or GCP_START_RETRY_SECONDS.
GCP_WATCH_SECONDS GCP_WATCH_HEARTBEAT_SECONDS
Pick a zone with Spot capacity for the selected GPU if create fails.
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
    --boot-disk-type="${DISK_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --metadata=install-nvidia-driver=True \
    --scopes=cloud-platform
  echo
  echo "VM created. First boot may install the NVIDIA driver and reboot."
  if [[ "${PROFILE}" == "g4" ]]; then
    echo "Next: $(basename "$0") g4 sync-inference && $(basename "$0") g4 ssh"
  else
    echo "Next: $(basename "$0") sync && $(basename "$0") ssh"
  fi
}

archive_bytes() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    stat -f%z "$1"
  else
    stat -c%s "$1"
  fi
}

sync() (
  need_project
  local mode="${1:-full}"
  local archive remote_archive size_bytes size_h file_count started
  archive="$(mktemp "${TMPDIR:-/tmp}/food-studio-sync.XXXXXX.tar.gz")"
  remote_archive="/tmp/food-studio-sync.tar.gz"
  started="${SECONDS}"
  trap 'rm -f "${archive}"' EXIT

  echo "Sync 1/4  Packing ${ROOT} (${mode})"
  # COPYFILE_DISABLE skips macOS xattrs that otherwise flood tar with warnings.
  if [[ "${mode}" == "inference" ]]; then
    COPYFILE_DISABLE=1 tar -C "${ROOT}" \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf "${archive}" \
      config.yaml requirements.txt src data/development_prompts.json data/validation_prompts.json
  else
    echo "          Skipping .git, virtualenvs, ui/node_modules, ui/dist, and generated outputs."
    COPYFILE_DISABLE=1 tar -C "${ROOT}" \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='venv' \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='.DS_Store' \
      --exclude='*.pyc' \
      --exclude='ui/node_modules' \
      --exclude='ui/dist' \
      --exclude='outputs/checkpoints' \
      --exclude='outputs/samples' \
      --exclude='outputs/comparisons' \
      --exclude='outputs/development' \
      --exclude='outputs/train.log' \
      --exclude='*.safetensors' \
      -czf "${archive}" .
  fi

  size_bytes="$(archive_bytes "${archive}")"
  size_h="$(du -h "${archive}" | awk '{print $1}')"
  file_count="$(COPYFILE_DISABLE=1 tar -tzf "${archive}" | wc -l | tr -d ' ')"
  echo "Sync 2/4  Packed ${size_h} (${file_count} files, ${size_bytes} bytes). Connecting to ${INSTANCE} (${ZONE})."
  ssh_cmd --command="mkdir -p \${HOME}/${REMOTE_DIR}"

  echo "Sync 3/4  Uploading ${size_h} to ${INSTANCE}:${remote_archive}"
  gcloud compute scp \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${archive}" \
    "${INSTANCE}:${remote_archive}"

  echo "Sync 4/4  Extracting on the VM into ~/${REMOTE_DIR}"
  ssh_cmd --command="tar -xzf ${remote_archive} -C \${HOME}/${REMOTE_DIR} && rm -f ${remote_archive}"

  echo "Synced ${size_h} -> ${INSTANCE}:~/${REMOTE_DIR} in $((SECONDS - started))s"
)

push_lora() {
  need_project
  local source_path="${1:-${ROOT}/outputs/checkpoints/final/pytorch_lora_weights.safetensors}"
  local checkpoint
  if [[ $# -gt 1 ]]; then
    echo "Usage: $(basename "$0") [g4] push-lora [checkpoint-directory|weights-file]" >&2
    return 1
  fi
  if [[ -d "${source_path}" ]]; then
    source_path="${source_path}/pytorch_lora_weights.safetensors"
  fi
  if [[ ! -f "${source_path}" || "${source_path}" != *.safetensors ]]; then
    echo "LoRA weights file not found: ${source_path}" >&2
    return 1
  fi
  checkpoint="$(basename "$(dirname "${source_path}")")"
  if [[ ! "${checkpoint}" =~ ^[A-Za-z0-9._-]+$ || "${checkpoint}" == "." || "${checkpoint}" == ".." ]]; then
    echo "Unsupported checkpoint directory name: ${checkpoint}" >&2
    return 1
  fi
  ssh_cmd --command="mkdir -p \${HOME}/${REMOTE_DIR}/outputs/checkpoints/${checkpoint}"
  gcloud compute scp \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${source_path}" \
    "${INSTANCE}:~/${REMOTE_DIR}/outputs/checkpoints/${checkpoint}/pytorch_lora_weights.safetensors"
  echo "Copied LoRA weights -> ${INSTANCE}:~/${REMOTE_DIR}/outputs/checkpoints/${checkpoint}/"
}

ssh() {
  need_project
  gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}"
}

parse_resume_arg() {
  case "${1:-}" in
    "")
      printf ''
      ;;
    --resume)
      if [[ -z "${2:-}" || $# -ne 2 ]]; then
        echo "Usage: $(basename "$0") train [--resume latest|<checkpoint-dir>]" >&2
        return 1
      fi
      if [[ ! "${2}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "Unsafe --resume value: ${2}" >&2
        return 1
      fi
      printf -- '--resume %s' "${2}"
      ;;
    *)
      echo "Usage: $(basename "$0") train [--resume latest|<checkpoint-dir>]" >&2
      return 1
      ;;
  esac
}

# Newest-first operation lines: "operationType<TAB>statusMessage".
# restart: the host stopped the VM. leave: a caller stopped it. unknown: no
# terminal operation is visible yet.
classify_operation_lines() {
  local line op
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    op="${line%%$'\t'*}"
    case "${op}" in
      compute.instances.preempted | compute.instances.hostError | compute.instances.terminateOnHostMaintenance)
        echo restart
        return 0
        ;;
      stop)
        echo leave
        return 0
        ;;
    esac
    case "${line}" in
      *[Pp]reempted* | *"host error"* | *"Host error"* | *"host maintenance"* | *"Host maintenance"*)
        echo restart
        return 0
        ;;
    esac
  done
  echo unknown
}

# Log text, or a grep extract of its marker lines. Looks at the newest
# "train start" segment: complete, failed, or incomplete.
job_outcome_from_log() {
  local log="$1"
  awk '
    /==== train start / { segment = "" }
    { segment = segment $0 "\n" }
    END {
      if (segment ~ /==== development renders exit 0 /) print "complete"
      else if (segment ~ /==== development renders exit /) print "failed"
      else if (segment ~ /==== train exit / && segment !~ /==== train exit 0 /) print "failed"
      else print "incomplete"
    }
  ' "${log}"
}

detach_training() {
  local resume_arg="$1"
  ssh_cmd --command="bash -lc \"mkdir -p \\\$HOME/${REMOTE_DIR}/outputs && cd \\\$HOME/${REMOTE_DIR} && nohup bash scripts/train_then_stop.sh ${resume_arg} >> outputs/train.log 2>&1 < /dev/null & echo Detached training pid \\\$! && echo Log: \\\$HOME/${REMOTE_DIR}/outputs/train.log && echo The VM stops when the job exits. This command restarts it if the host terminates the VM first.\""
}

wait_for_ssh() {
  local attempt
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if ssh_cmd --command="true"; then
      return 0
    fi
    echo "SSH to ${INSTANCE} is not ready (attempt ${attempt}). Retrying in 5s." >&2
    sleep 5
  done
  echo "SSH to ${INSTANCE} did not become ready." >&2
  return 1
}

remote_training_running() {
  ssh_cmd --command="if pgrep -f '[b]ash scripts/train_then_stop' >/dev/null || pgrep -f '[p]ython3 -m src.train' >/dev/null || pgrep -f '[p]ython -m src.train' >/dev/null || pgrep -f '[p]ython3 -m src.inference' >/dev/null || pgrep -f '[p]ython -m src.inference' >/dev/null || pgrep -f '[p]ython3 -m src.development' >/dev/null || pgrep -f '[p]ython -m src.development' >/dev/null; then echo yes; else echo no; fi" | grep -qx yes
}

remote_job_outcome() {
  local tmp
  tmp="$(mktemp)"
  if ! ssh_cmd --command="if [[ -f \$HOME/${REMOTE_DIR}/outputs/train.log ]]; then grep -E '==== train start |==== train exit |==== development renders exit ' \$HOME/${REMOTE_DIR}/outputs/train.log; fi" >"${tmp}"; then
    rm -f "${tmp}"
    echo "Could not read the training log on ${INSTANCE}." >&2
    return 1
  fi
  job_outcome_from_log "${tmp}"
  rm -f "${tmp}"
}

latest_operation_lines() {
  gcloud compute operations list \
    --project="${PROJECT}" \
    --zones="${ZONE}" \
    --filter="targetLink:instances/${INSTANCE}" \
    --sort-by=~insertTime \
    --limit=20 \
    --format="value(operationType,statusMessage)"
}

stop_action() {
  local attempt lines action
  for attempt in 1 2 3 4 5 6; do
    lines="$(latest_operation_lines)" || lines=""
    action="$(printf '%s\n' "${lines}" | classify_operation_lines)"
    if [[ "${action}" != "unknown" ]]; then
      printf '%s\n' "${action}"
      return 0
    fi
    echo "Stop reason for ${INSTANCE} is not visible yet (attempt ${attempt}). Retrying." >&2
    sleep 5
  done
  echo unknown
}

launch_or_watch() {
  local resume_arg="$1"
  if remote_training_running; then
    echo "Training is already running on ${INSTANCE}. Watching for host termination."
    return 0
  fi
  echo "Starting training on ${INSTANCE}."
  detach_training "${resume_arg}"
}

resume_after_host_stop() {
  local outcome
  outcome="$(remote_job_outcome)"
  case "${outcome}" in
    complete)
      echo "The job had already finished. Stopping ${INSTANCE}."
      stop
      exit 0
      ;;
    failed)
      echo "The job had already exited with an error. Stopping ${INSTANCE}."
      stop
      exit 1
      ;;
    incomplete)
      launch_or_watch "--resume latest"
      ;;
    *)
      echo "Unexpected training-log outcome: ${outcome}" >&2
      exit 1
      ;;
  esac
}

wait_until_instance_stops() {
  local interval="${GCP_WATCH_SECONDS:-30}"
  local heartbeat="${GCP_WATCH_HEARTBEAT_SECONDS:-300}"
  local status="" previous="" elapsed=0
  if [[ ! "${interval}" =~ ^[1-9][0-9]*$ || ! "${heartbeat}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GCP_WATCH_SECONDS and GCP_WATCH_HEARTBEAT_SECONDS must be positive numbers of seconds." >&2
    return 1
  fi
  while true; do
    if ! status="$(instance_status)" || [[ -z "${status}" ]]; then
      echo "Could not read ${INSTANCE} status. Retrying in ${interval}s." >&2
      sleep "${interval}"
      continue
    fi
    case "${status}" in
      RUNNING | STAGING | PROVISIONING | REPAIRING | SUSPENDING | STOPPING)
        if [[ "${status}" != "${previous}" || "${elapsed}" -ge "${heartbeat}" ]]; then
          echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${INSTANCE} is ${status}. Watching for host termination."
          elapsed=0
        fi
        previous="${status}"
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        ;;
      *)
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${INSTANCE} is ${status}."
        return 0
        ;;
    esac
  done
}

train() {
  need_project
  local resume_arg seen_launch=0 action
  resume_arg="$(parse_resume_arg "$@")"
  while true; do
    start
    wait_for_ssh
    if [[ "${seen_launch}" -eq 1 ]]; then
      resume_after_host_stop
    else
      launch_or_watch "${resume_arg}"
    fi
    seen_launch=1
    wait_until_instance_stops
    action="$(stop_action)"
    if [[ "${action}" == "restart" ]]; then
      echo "Host terminated ${INSTANCE} before the job finished. Starting it again."
      continue
    fi
    if [[ "${action}" == "unknown" ]]; then
      echo "Could not tell why ${INSTANCE} stopped. Leaving it stopped." >&2
      return 1
    fi
    echo "${INSTANCE} stopped. Leaving it stopped."
    return 0
  done
}

instance_status() {
  gcloud_vm describe "${INSTANCE}" --format='get(status)'
}

wait_while_stopping() {
  local status elapsed=0
  local timeout="${GCP_START_WAIT_SECONDS:-600}"
  local interval=5
  status="$(instance_status)"
  if [[ "${status}" != "STOPPING" && "${status}" != "SUSPENDING" ]]; then
    return 0
  fi
  echo "Waiting for ${INSTANCE} to finish ${status} before start (timeout ${timeout}s)."
  while [[ "${status}" == "STOPPING" || "${status}" == "SUSPENDING" ]]; do
    if [[ "${elapsed}" -ge "${timeout}" ]]; then
      echo "Timed out after ${elapsed}s; ${INSTANCE} is still ${status}." >&2
      exit 1
    fi
    echo "  ${status}... ${elapsed}s"
    sleep "${interval}"
    elapsed=$((elapsed + interval))
    status="$(instance_status)"
  done
  echo "  ${status} after ${elapsed}s"
}

stop() {
  need_project
  gcloud_vm stop "${INSTANCE}"
}

start() {
  need_project
  wait_while_stopping
  local status
  status="$(instance_status)"
  if [[ "${status}" == "RUNNING" ]]; then
    echo "${INSTANCE} is already RUNNING."
    return 0
  fi
  if [[ "${status}" != "TERMINATED" && "${status}" != "STOPPED" ]]; then
    echo "Refusing to start ${INSTANCE} from status ${status}." >&2
    exit 1
  fi
  local interval="${GCP_START_RETRY_SECONDS:-60}"
  local attempt=1
  local output
  if [[ ! "${interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GCP_START_RETRY_SECONDS must be a positive number of seconds." >&2
    exit 1
  fi
  while true; do
    if output="$(gcloud_vm start "${INSTANCE}" 2>&1)"; then
      printf '%s\n' "${output}"
      return 0
    fi
    printf '%s\n' "${output}" >&2
    if [[ "${output}" != *"ZONE_RESOURCE_POOL_EXHAUSTED"* ]]; then
      return 1
    fi
    echo "No ${MACHINE} capacity in ${ZONE}. Retry ${attempt} in ${interval}s. Ctrl-C to stop." >&2
    attempt=$((attempt + 1))
    sleep "${interval}"
  done
}

watch() {
  need_project
  wait_while_stopping
  local vm_status
  vm_status="$(instance_status)"
  echo "Watch ${INSTANCE} (${ZONE}): ${vm_status}"
  if [[ "${vm_status}" != "RUNNING" ]]; then
    echo "The VM is not running; start it to stream the log." >&2
    exit 1
  fi
  echo "Ctrl-C stops this view only. Training on the VM continues."
  ssh_cmd --command="bash -lc 'set +e
echo \"=== GPU ===\"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo
echo \"=== process ===\"
pgrep -af \"src.(inference|train|development)|train_then_stop\" || echo \"No training process visible.\"
echo
LOG=\"\$HOME/${REMOTE_DIR}/outputs/train.log\"
if [[ ! -f \"\$LOG\" ]]; then
  echo \"Waiting for \$LOG ...\"
  while [[ ! -f \"\$LOG\" ]]; do sleep 2; done
fi
echo \"=== log ===\"
tail -n 80 -F \"\$LOG\"
'"
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

main() {
  local cmd="${1:-}"
  shift || true
  if [[ "${cmd}" == "g4" ]]; then
    PROFILE=g4
    configure_profile
    cmd="${1:-}"
    shift || true
  fi
  case "${cmd}" in
    create | sync | ssh | watch | stop | start | pull | delete) "${cmd}" ;;
    sync-inference) sync inference ;;
    push-lora) push_lora "$@" ;;
    train) train "$@" ;;
    -h | --help | help | "") usage ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
