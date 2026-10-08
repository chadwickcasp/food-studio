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
Usage: $(basename "$0") [g4] <create|sync|sync-inference|push-lora|ssh|train|develop|eval|watch|stop|start|pull|delete>

  g4       Select the separate G4 instance (RTX PRO 6000, 96 GB VRAM).
  create   Create a Spot ${MACHINE} VM with a ${DISK_SIZE} ${DISK_TYPE} boot disk.
  sync     Copy this repository onto the VM, including local training images.
           Prints pack/upload/extract progress; skips venvs, UI build trees,
           and generated outputs.
  sync-inference
           Copy only inference code, config, requirements, prompt files, and
           the development render script.
  push-lora [path]
           Copy one local LoRA weight file or checkpoint directory to the VM.
           Defaults to outputs/checkpoints/final/pytorch_lora_weights.safetensors.
  ssh      Open an SSH session.
  train    Detach baseline, training, and 50-step dev renders; then stop the VM.
           Stays attached. If the host preempts or kills the VM before the job
           finishes, start it again and resume from the latest checkpoint.
  develop [--name <run-name>] [--force]
           Upload every configured checkpoint, render development prompts for
           Base and each checkpoint, then stop the VM. Steps come from
           training.checkpoint_steps and training.max_train_steps. A later
           fine-tune uses the same command; pass --name to keep its review
           manifest separate. --name starts with a letter or number, then
           letters, numbers, dots, underscores, or hyphens. --force renders
           every sample again.
  eval [--step <n>] [--force]
           Render the held-out eval prompts for Base and one checkpoint, then
           stop the VM. The step defaults to selectedStep in
           outputs/reviews/checkpoint-selection.json. The checkpoint file must
           already be on the VM. --force renders every sample again on the
           first launch.
  watch    Stream GPU, process, and the running job log until you Ctrl-C.
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

develop stays in the foreground after detaching the render. It uploads
pytorch_lora_weights.safetensors for each configured checkpoint, then renders
the current development prompts. The VM script stops the instance when that
render exits. A host preemption starts the VM again and continues; saved
images are kept when the manifest still matches the prompt, seed, settings,
and adapter. --force renders every sample again on the first launch. A
restart does not repeat it. Ctrl-C ends this watcher and leaves the VM as it is.

eval stays in the foreground after detaching the render. It renders
data/validation_prompts.json for Base and the selected checkpoint, then the
VM script stops the instance. A host preemption starts the VM again and
continues; saved images are kept when the manifest still matches. --force
renders every sample again on the first launch. A restart does not repeat it.
Ctrl-C ends this watcher and leaves the VM as it is.

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
      config.yaml requirements.txt src scripts/develop_then_stop.sh \
      scripts/eval_then_stop.sh \
      data/development_prompts.json data/validation_prompts.json
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

# Same schedule as src.development.checkpoint_steps: every interval, plus the
# final step when it is not already on that interval.
checkpoint_step_list() {
  local max_steps="$1"
  local interval="$2"
  local step
  if [[ ! "${max_steps}" =~ ^[1-9][0-9]*$ || ! "${interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "training.max_train_steps and training.checkpoint_steps must be positive integers." >&2
    return 1
  fi
  step="${interval}"
  while [[ "${step}" -le "${max_steps}" ]]; do
    printf '%s\n' "${step}"
    step=$((step + interval))
  done
  if [[ $((max_steps % interval)) -ne 0 ]]; then
    printf '%s\n' "${max_steps}"
  fi
}

config_field() {
  local key="$1"
  local file="${ROOT}/config.yaml"
  local value=""
  if [[ ! -f "${file}" ]]; then
    echo "Config not found: ${file}" >&2
    return 1
  fi
  value="$(awk -v key="${key}:" '$1 == key { print $2; exit }' "${file}")"
  if [[ -z "${value}" ]]; then
    echo "Missing ${key} in ${file}." >&2
    return 1
  fi
  printf '%s' "${value}"
}

# Prints repo-relative pytorch_lora_weights.safetensors paths, one per line.
required_lora_weights() {
  local checkpoints_dir="$1"
  shift
  local step path rel missing=()
  if [[ "${checkpoints_dir}" != "${ROOT}/"* || "${checkpoints_dir}" == *..* ]]; then
    echo "Checkpoint directory must stay inside ${ROOT}." >&2
    return 1
  fi
  for step in "$@"; do
    path="${checkpoints_dir}/checkpoint-${step}/pytorch_lora_weights.safetensors"
    if [[ ! -f "${path}" ]]; then
      missing+=("${step}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing LoRA weights for checkpoint(s): ${missing[*]}" >&2
    echo "Expected pytorch_lora_weights.safetensors in ${checkpoints_dir}/checkpoint-<step>/." >&2
    echo "Pull the training checkpoints onto this machine before rendering." >&2
    return 1
  fi
  for step in "$@"; do
    path="${checkpoints_dir}/checkpoint-${step}/pytorch_lora_weights.safetensors"
    rel="${path#"${ROOT}/"}"
    printf '%s\n' "${rel}"
  done
}

load_develop_weights() {
  local max_steps="" interval="" checkpoints_dir="" steps_text=""
  max_steps="$(config_field max_train_steps)" || return 1
  interval="$(config_field checkpoint_steps)" || return 1
  checkpoints_dir="$(config_field checkpoints_dir)" || return 1
  if [[ "${checkpoints_dir}" == /* || "${checkpoints_dir}" == *..* ]]; then
    echo "output.checkpoints_dir must be a path inside the repository." >&2
    return 1
  fi
  steps_text="$(checkpoint_step_list "${max_steps}" "${interval}")" || return 1
  # Integers from checkpoint_step_list have no spaces or glob characters.
  # shellcheck disable=SC2086
  required_lora_weights "${ROOT}/${checkpoints_dir}" ${steps_text}
}

parse_develop_args() {
  local name="" force=0 rendered=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force)
        force=1
        shift
        ;;
      --name)
        if [[ $# -lt 2 || ! "${2}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
          echo "Usage: $(basename "$0") [g4] develop [--name <run-name>] [--force]" >&2
          echo "--name must be letters, numbers, dots, underscores, or hyphens." >&2
          return 1
        fi
        name="${2}"
        shift 2
        ;;
      *)
        echo "Usage: $(basename "$0") [g4] develop [--name <run-name>] [--force]" >&2
        return 1
        ;;
    esac
  done
  if [[ -n "${name}" ]]; then
    rendered="--name ${name}"
  fi
  if [[ "${force}" -eq 1 ]]; then
    if [[ -n "${rendered}" ]]; then
      rendered="${rendered} --force"
    else
      rendered="--force"
    fi
  fi
  printf '%s' "${rendered}"
}

# Drop --force so a host restart keeps finished images. The first launch still
# receives the original arguments from parse_develop_args.
develop_resume_args() {
  local args="$1"
  local part="" rendered=""
  if [[ -z "${args}" ]]; then
    printf ''
    return 0
  fi
  # Arguments are --name, a validated token, and --force. No spaces or globs.
  # shellcheck disable=SC2086
  for part in ${args}; do
    if [[ "${part}" == "--force" ]]; then
      continue
    fi
    if [[ -n "${rendered}" ]]; then
      rendered="${rendered} ${part}"
    else
      rendered="${part}"
    fi
  done
  printf '%s' "${rendered}"
}

selected_checkpoint_step() {
  local file="${ROOT}/outputs/reviews/checkpoint-selection.json"
  local step=""
  if [[ ! -f "${file}" ]]; then
    echo "No checkpoint selection at ${file}. Pass --step <n>." >&2
    return 1
  fi
  if ! step="$(python3 -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")).get("selectedStep"); print("" if value is None else value)' "${file}")"; then
    echo "Could not read selectedStep from ${file}." >&2
    return 1
  fi
  if [[ ! "${step}" =~ ^[1-9][0-9]*$ ]]; then
    echo "selectedStep in ${file} must be a positive integer." >&2
    return 1
  fi
  printf '%s' "${step}"
}

# Prints "<step>" or "<step> --force". The step is a positive integer.
parse_eval_args() {
  local step="" force=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force)
        force=1
        shift
        ;;
      --step)
        if [[ $# -lt 2 || ! "${2}" =~ ^[1-9][0-9]*$ ]]; then
          echo "Usage: $(basename "$0") [g4] eval [--step <n>] [--force]" >&2
          echo "--step must be a positive integer." >&2
          return 1
        fi
        step="${2}"
        shift 2
        ;;
      *)
        echo "Usage: $(basename "$0") [g4] eval [--step <n>] [--force]" >&2
        return 1
        ;;
    esac
  done
  if [[ -z "${step}" ]]; then
    step="$(selected_checkpoint_step)" || return 1
  fi
  if [[ "${force}" -eq 1 ]]; then
    printf '%s --force' "${step}"
  else
    printf '%s' "${step}"
  fi
}

# Drop --force so a host restart keeps finished eval images.
eval_resume_args() {
  local args="$1"
  local part="" rendered=""
  if [[ -z "${args}" ]]; then
    printf ''
    return 0
  fi
  # Arguments are a positive integer and optional --force.
  # shellcheck disable=SC2086
  for part in ${args}; do
    if [[ "${part}" == "--force" ]]; then
      continue
    fi
    if [[ -n "${rendered}" ]]; then
      rendered="${rendered} ${part}"
    else
      rendered="${part}"
    fi
  done
  printf '%s' "${rendered}"
}

checkpoint_weight_rel_ok() {
  local rel="$1"
  local parent base
  parent="$(basename "$(dirname "${rel}")")"
  base="$(basename "${rel}")"
  if [[ "${rel}" == /* || "${rel}" == *..* ]]; then
    return 1
  fi
  if [[ ! "${parent}" =~ ^checkpoint-[0-9]+$ ]]; then
    return 1
  fi
  [[ "${base}" == "pytorch_lora_weights.safetensors" ]]
}

push_checkpoint_weights() (
  need_project
  local rel archive remote_archive size_h started
  if [[ $# -lt 1 ]]; then
    echo "No checkpoint weights to upload." >&2
    return 1
  fi
  for rel in "$@"; do
    if ! checkpoint_weight_rel_ok "${rel}"; then
      echo "Refusing to upload unexpected checkpoint path: ${rel}" >&2
      return 1
    fi
    if [[ ! -f "${ROOT}/${rel}" ]]; then
      echo "LoRA weights file not found: ${ROOT}/${rel}" >&2
      return 1
    fi
  done
  archive="$(mktemp "${TMPDIR:-/tmp}/food-studio-loras.XXXXXX.tar.gz")"
  remote_archive="/tmp/food-studio-loras.tar.gz"
  trap 'rm -f "${archive}"' EXIT
  started="${SECONDS}"
  echo "Packing $# checkpoint weight file(s)."
  COPYFILE_DISABLE=1 tar -C "${ROOT}" -czf "${archive}" "$@"
  size_h="$(du -h "${archive}" | awk '{print $1}')"
  echo "Uploading ${size_h} of LoRA weights to ${INSTANCE}."
  ssh_cmd --command="mkdir -p \${HOME}/${REMOTE_DIR}"
  gcloud compute scp \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    "${archive}" \
    "${INSTANCE}:${remote_archive}"
  ssh_cmd --command="tar -xzf ${remote_archive} -C \${HOME}/${REMOTE_DIR} && rm -f ${remote_archive}"
  echo "Uploaded $# checkpoint weight file(s) in $((SECONDS - started))s."
)

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

remote_develop_running() {
  ssh_cmd --command="if pgrep -f '[b]ash scripts/develop_then_stop' >/dev/null; then echo yes; else echo no; fi" | grep -qx yes
}

remote_eval_running() {
  ssh_cmd --command="if pgrep -f '[b]ash scripts/eval_then_stop' >/dev/null; then echo yes; else echo no; fi" | grep -qx yes
}

remote_training_running() {
  ssh_cmd --command="if pgrep -f '[b]ash scripts/train_then_stop' >/dev/null || pgrep -f '[b]ash scripts/develop_then_stop' >/dev/null || pgrep -f '[b]ash scripts/eval_then_stop' >/dev/null || pgrep -f '[p]ython3 -m src.train' >/dev/null || pgrep -f '[p]ython -m src.train' >/dev/null || pgrep -f '[p]ython3 -m src.inference' >/dev/null || pgrep -f '[p]ython -m src.inference' >/dev/null || pgrep -f '[p]ython3 -m src.development' >/dev/null || pgrep -f '[p]ython -m src.development' >/dev/null; then echo yes; else echo no; fi" | grep -qx yes
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

# Newest "development job start" segment: complete, failed, or incomplete.
development_outcome_from_log() {
  local log="$1"
  awk '
    /==== development job start / { segment = "" }
    { segment = segment $0 "\n" }
    END {
      if (segment ~ /==== development renders exit 0 /) print "complete"
      else if (segment ~ /==== development renders exit /) print "failed"
      else print "incomplete"
    }
  ' "${log}"
}

detach_development() {
  local extra="$1"
  ssh_cmd --command="bash -lc \"mkdir -p \\\$HOME/${REMOTE_DIR}/outputs && cd \\\$HOME/${REMOTE_DIR} && nohup bash scripts/develop_then_stop.sh ${extra} >> outputs/development.log 2>&1 < /dev/null & echo Detached development pid \\\$! && echo Log: \\\$HOME/${REMOTE_DIR}/outputs/development.log && echo The VM stops when the job exits. This command restarts it if the host terminates the VM first.\""
}

remote_development_outcome() {
  local tmp
  tmp="$(mktemp)"
  if ! ssh_cmd --command="if [[ -f \$HOME/${REMOTE_DIR}/outputs/development.log ]]; then grep -E '==== development job start |==== development renders exit ' \$HOME/${REMOTE_DIR}/outputs/development.log; fi" >"${tmp}"; then
    rm -f "${tmp}"
    echo "Could not read the development log on ${INSTANCE}." >&2
    return 1
  fi
  development_outcome_from_log "${tmp}"
  rm -f "${tmp}"
}

cli_profile() {
  if [[ "${PROFILE}" == "g4" ]]; then
    printf ' g4'
  fi
}

print_retrieve_renders() {
  echo "Retrieve renders with: scripts/gcp.sh$(cli_profile) start && scripts/gcp.sh$(cli_profile) pull && scripts/gcp.sh$(cli_profile) stop"
}

report_develop_stopped() {
  echo "${INSTANCE} stopped. Leaving it stopped."
  echo "Check outputs/development.log for 'development renders exit 0' before reviewing the images."
  print_retrieve_renders
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
    echo "A GPU job is already running on ${INSTANCE}. Watching for host termination."
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

refuse_other_gpu_job() {
  echo "A training or inference job is running on ${INSTANCE}. Wait for it to finish before rendering." >&2
}

resume_development() {
  local extra="$1"
  local outcome
  outcome="$(remote_development_outcome)"
  case "${outcome}" in
    complete)
      echo "The development render had already finished. Stopping ${INSTANCE}."
      stop
      print_retrieve_renders
      exit 0
      ;;
    failed)
      echo "The development render had already exited with an error. Stopping ${INSTANCE}."
      stop
      exit 1
      ;;
    incomplete)
      if remote_develop_running; then
        echo "Development rendering is already running on ${INSTANCE}. Watching for host termination."
        return 0
      fi
      if remote_training_running; then
        refuse_other_gpu_job
        return 1
      fi
      echo "Resuming development renders on ${INSTANCE}."
      detach_development "${extra}"
      ;;
    *)
      echo "Unexpected development-log outcome: ${outcome}" >&2
      exit 1
      ;;
  esac
}

rotate_development_log() {
  ssh_cmd --command="log=\$HOME/${REMOTE_DIR}/outputs/development.log; if [[ -f \"\$log\" ]]; then mv \"\$log\" \"\$log.prev\"; fi"
}

upload_develop_inputs() {
  local weights_text="$1"
  echo "Syncing inference files to ${INSTANCE}."
  sync inference
  echo "Uploading checkpoint weights to ${INSTANCE}."
  # Paths from load_develop_weights are relative and contain no spaces.
  # shellcheck disable=SC2086
  push_checkpoint_weights ${weights_text}
}

begin_development() {
  local render_args="$1"
  local weights_text="$2"
  if remote_develop_running; then
    echo "Development rendering is already running on ${INSTANCE}. Watching for host termination."
    return 0
  fi
  if remote_training_running; then
    refuse_other_gpu_job
    return 1
  fi
  # Drop a previous run's log before upload. A preemption during upload must
  # not treat that older "exit 0" as this launch finishing.
  rotate_development_log
  upload_develop_inputs "${weights_text}"
  echo "Starting development renders on ${INSTANCE}."
  detach_development "${render_args}"
}

develop() {
  need_project
  local render_args="" resume_args="" weights_text="" seen_launch=0 action status=""
  render_args="$(parse_develop_args "$@")" || return 1
  resume_args="$(develop_resume_args "${render_args}")"
  weights_text="$(load_develop_weights)" || return 1
  while true; do
    start
    wait_for_ssh
    if [[ "${seen_launch}" -eq 1 ]]; then
      resume_development "${resume_args}"
    elif ! begin_development "${render_args}" "${weights_text}"; then
      status="$(instance_status)" || status="RUNNING"
      if [[ "${status}" != "RUNNING" ]]; then
        echo "Host terminated ${INSTANCE} before the development render started. Starting it again."
        continue
      fi
      return 1
    else
      seen_launch=1
    fi
    wait_until_instance_stops
    action="$(stop_action)"
    if [[ "${action}" == "restart" ]]; then
      echo "Host terminated ${INSTANCE} before the development render finished. Starting it again."
      continue
    fi
    if [[ "${action}" == "unknown" ]]; then
      echo "Could not tell why ${INSTANCE} stopped. Leaving it stopped." >&2
      return 1
    fi
    report_develop_stopped
    return 0
  done
}

detach_eval() {
  local step="$1"
  local force_arg="${2:-}"
  ssh_cmd --command="bash -lc \"mkdir -p \\\$HOME/${REMOTE_DIR}/outputs && cd \\\$HOME/${REMOTE_DIR} && nohup bash scripts/eval_then_stop.sh --step ${step} ${force_arg} >> outputs/eval.log 2>&1 < /dev/null & echo Detached eval pid \\\$! && echo Log: \\\$HOME/${REMOTE_DIR}/outputs/eval.log && echo The VM stops when the job exits. This command restarts it if the host terminates the VM first.\""
}

# Newest "eval job start" segment: complete, failed, or incomplete.
eval_outcome_from_log() {
  local log="$1"
  awk '
    /==== eval job start / { segment = "" }
    { segment = segment $0 "\n" }
    END {
      if (segment ~ /==== eval renders exit 0 /) print "complete"
      else if (segment ~ /==== eval renders exit /) print "failed"
      else print "incomplete"
    }
  ' "${log}"
}

remote_eval_outcome() {
  local tmp
  tmp="$(mktemp)"
  if ! ssh_cmd --command="if [[ -f \$HOME/${REMOTE_DIR}/outputs/eval.log ]]; then grep -E '==== eval job start |==== eval renders exit ' \$HOME/${REMOTE_DIR}/outputs/eval.log; fi" >"${tmp}"; then
    rm -f "${tmp}"
    echo "Could not read the eval log on ${INSTANCE}." >&2
    return 1
  fi
  eval_outcome_from_log "${tmp}"
  rm -f "${tmp}"
}

remote_eval_checkpoint_present() {
  local step="$1"
  ssh_cmd --command="test -f \$HOME/${REMOTE_DIR}/outputs/checkpoints/checkpoint-${step}/pytorch_lora_weights.safetensors"
}

rotate_eval_log() {
  ssh_cmd --command="log=\$HOME/${REMOTE_DIR}/outputs/eval.log; if [[ -f \"\$log\" ]]; then mv \"\$log\" \"\$log.prev\"; fi"
}

report_eval_stopped() {
  echo "${INSTANCE} stopped. Leaving it stopped."
  echo "Check outputs/eval.log for 'eval renders exit 0' before reviewing the images."
  print_retrieve_renders
}

resume_eval() {
  local step="$1"
  local outcome
  outcome="$(remote_eval_outcome)"
  case "${outcome}" in
    complete)
      echo "The eval render had already finished. Stopping ${INSTANCE}."
      stop
      print_retrieve_renders
      exit 0
      ;;
    failed)
      echo "The eval render had already exited with an error. Stopping ${INSTANCE}."
      stop
      exit 1
      ;;
    incomplete)
      if remote_eval_running; then
        echo "Eval rendering is already running on ${INSTANCE}. Watching for host termination."
        return 0
      fi
      if remote_training_running; then
        refuse_other_gpu_job
        return 1
      fi
      echo "Resuming eval renders on ${INSTANCE}."
      detach_eval "${step}"
      ;;
    *)
      echo "Unexpected eval-log outcome: ${outcome}" >&2
      exit 1
      ;;
  esac
}

# Returns 2 when the caller must stop without treating the failure as a host preemption.
begin_eval() {
  local step="$1"
  local force_arg="${2:-}"
  if remote_eval_running; then
    echo "Eval rendering is already running on ${INSTANCE}. Watching for host termination."
    return 0
  fi
  if remote_training_running; then
    refuse_other_gpu_job
    return 2
  fi
  if ! remote_eval_checkpoint_present "${step}"; then
    echo "Missing outputs/checkpoints/checkpoint-${step}/pytorch_lora_weights.safetensors on ${INSTANCE}." >&2
    echo "Stopping ${INSTANCE}." >&2
    stop || true
    return 2
  fi
  # Drop a previous run's log before sync. A preemption during sync must not
  # treat that older "exit 0" as this launch finishing.
  rotate_eval_log
  echo "Syncing inference files to ${INSTANCE}."
  sync inference
  echo "Starting eval renders on ${INSTANCE}."
  detach_eval "${step}" "${force_arg}"
}

render_eval() {
  need_project
  local eval_args="" resume_args="" step="" force_arg="" seen_launch=0 action begin_status=0 status=""
  eval_args="$(parse_eval_args "$@")" || return 1
  resume_args="$(eval_resume_args "${eval_args}")"
  step="${resume_args}"
  if [[ "${eval_args}" == *" --force" ]]; then
    force_arg="--force"
  fi
  while true; do
    start
    wait_for_ssh
    if [[ "${seen_launch}" -eq 1 ]]; then
      resume_eval "${step}"
    else
      begin_status=0
      begin_eval "${step}" "${force_arg}" || begin_status=$?
      if [[ "${begin_status}" -eq 2 ]]; then
        return 1
      fi
      if [[ "${begin_status}" -ne 0 ]]; then
        status="$(instance_status)" || status="RUNNING"
        if [[ "${status}" != "RUNNING" ]]; then
          echo "Host terminated ${INSTANCE} before the eval render started. Starting it again."
          continue
        fi
        return 1
      fi
      seen_launch=1
    fi
    wait_until_instance_stops
    action="$(stop_action)"
    if [[ "${action}" == "restart" ]]; then
      echo "Host terminated ${INSTANCE} before the eval render finished. Starting it again."
      continue
    fi
    if [[ "${action}" == "unknown" ]]; then
      echo "Could not tell why ${INSTANCE} stopped. Leaving it stopped." >&2
      return 1
    fi
    report_eval_stopped
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
  echo "Ctrl-C stops this view only. The VM job continues."
  ssh_cmd --command="bash -lc 'set +e
echo \"=== GPU ===\"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo
echo \"=== process ===\"
pgrep -af \"src.(inference|train|development)|train_then_stop|develop_then_stop|eval_then_stop\" || echo \"No job process visible.\"
echo
if pgrep -f \"[b]ash scripts/eval_then_stop\" >/dev/null; then
  LOG=\"\$HOME/${REMOTE_DIR}/outputs/eval.log\"
elif pgrep -f \"[b]ash scripts/develop_then_stop\" >/dev/null; then
  LOG=\"\$HOME/${REMOTE_DIR}/outputs/development.log\"
else
  LOG=\"\$HOME/${REMOTE_DIR}/outputs/train.log\"
fi
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
    develop) develop "$@" ;;
    eval) render_eval "$@" ;;
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
