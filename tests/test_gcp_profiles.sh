#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/gcp.sh"

gcloud() {
  printf '<%s>\n' "$@"
  if [[ "${1:-}" == "compute" && "${2:-}" == "scp" ]]; then
    local arg
    for arg in "$@"; do
      if [[ "${arg}" == *.tar.gz && -f "${arg}" ]]; then
        tar -tzf "${arg}" | sed 's/^/archive:/'
      fi
    done
  fi
}

assert_contains() {
  if [[ "$1" != *"$2"* ]]; then
    echo "Expected '$2' in mocked gcloud call" >&2
    exit 1
  fi
}

l4_create="$(main create)"
assert_contains "${l4_create}" '<food-studio>'
assert_contains "${l4_create}" '<--zone=us-central1-a>'
assert_contains "${l4_create}" '<--machine-type=g2-standard-8>'
assert_contains "${l4_create}" '<--boot-disk-type=pd-balanced>'

g4_create="$(main g4 create)"
assert_contains "${g4_create}" '<food-studio-g4>'
assert_contains "${g4_create}" '<--zone=us-central1-b>'
assert_contains "${g4_create}" '<--machine-type=g4-standard-48>'
assert_contains "${g4_create}" '<--boot-disk-type=hyperdisk-balanced>'
assert_contains "${g4_create}" 'g4 sync'

inference_sync="$(main g4 sync-inference)"
assert_contains "${inference_sync}" 'archive:config.yaml'
assert_contains "${inference_sync}" 'archive:requirements.txt'
assert_contains "${inference_sync}" 'archive:src/inference.py'
assert_contains "${inference_sync}" 'archive:scripts/develop_then_stop.sh'
assert_contains "${inference_sync}" 'archive:scripts/eval_then_stop.sh'
assert_contains "${inference_sync}" 'archive:data/development_prompts.json'
assert_contains "${inference_sync}" 'archive:data/validation_prompts.json'
if [[ "${inference_sync}" == *'archive:data/train/'* || "${inference_sync}" == *'.jpg'* ]]; then
  echo "Inference sync included training images" >&2
  exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -f "${temp_dir}/checkpoint-500/pytorch_lora_weights.safetensors"; rmdir "${temp_dir}/checkpoint-500" "${temp_dir}"' EXIT
mkdir "${temp_dir}/checkpoint-500"
touch "${temp_dir}/checkpoint-500/pytorch_lora_weights.safetensors"
upload="$(main g4 push-lora "${temp_dir}/checkpoint-500")"
assert_contains "${upload}" '<scp>'
assert_contains "${upload}" '<--zone=us-central1-b>'
assert_contains "${upload}" '<food-studio-g4:~/food-studio/outputs/checkpoints/checkpoint-500/pytorch_lora_weights.safetensors>'

assert_eq() {
  if [[ "$1" != "$2" ]]; then
    echo "Expected '$2' but got '$1'" >&2
    exit 1
  fi
}

assert_eq "$(checkpoint_step_list 1000 50 | paste -sd' ' -)" "$(seq 50 50 1000 | paste -sd' ' -)"
assert_eq "$(checkpoint_step_list 1000 300 | paste -sd' ' -)" "300 600 900 1000"
assert_eq "$(checkpoint_step_list 40 50 | paste -sd' ' -)" "40"
if checkpoint_step_list 0 50 >/dev/null 2>&1; then
  echo "checkpoint_step_list accepted a non-positive max step" >&2
  exit 1
fi

assert_eq "$(parse_develop_args)" ""
assert_eq "$(parse_develop_args --force)" "--force"
assert_eq "$(parse_develop_args --name run-2 --force)" "--name run-2 --force"
assert_eq "$(develop_resume_args "--name run-2 --force")" "--name run-2"
assert_eq "$(develop_resume_args "--force")" ""
assert_eq "$(develop_resume_args "--name run-2")" "--name run-2"
assert_eq "$(develop_resume_args "")" ""
if parse_develop_args --name "bad name" >/dev/null 2>&1; then
  echo "parse_develop_args accepted a name with a space" >&2
  exit 1
fi
if parse_develop_args --name "--force" >/dev/null 2>&1; then
  echo "parse_develop_args accepted a flag as a run name" >&2
  exit 1
fi

fixture="$(mktemp -d)"
saved_root="${ROOT}"
ROOT="${fixture}"
mkdir -p "${fixture}/outputs/checkpoints/checkpoint-50" "${fixture}/outputs/checkpoints/checkpoint-100"
printf 'weights' > "${fixture}/outputs/checkpoints/checkpoint-50/pytorch_lora_weights.safetensors"
printf 'weights' > "${fixture}/outputs/checkpoints/checkpoint-100/pytorch_lora_weights.safetensors"
cat > "${fixture}/config.yaml" <<'EOF'
training:
  max_train_steps: 100
  checkpoint_steps: 50
output:
  checkpoints_dir: outputs/checkpoints
EOF
weights="$(load_develop_weights)"
assert_eq "${weights}" $'outputs/checkpoints/checkpoint-50/pytorch_lora_weights.safetensors\noutputs/checkpoints/checkpoint-100/pytorch_lora_weights.safetensors'
if push_checkpoint_weights "../outputs/checkpoints/checkpoint-50/pytorch_lora_weights.safetensors" >/dev/null 2>&1; then
  echo "push_checkpoint_weights accepted a path outside the repository" >&2
  exit 1
fi
PROFILE=g4
configure_profile
upload_all="$(push_checkpoint_weights outputs/checkpoints/checkpoint-50/pytorch_lora_weights.safetensors outputs/checkpoints/checkpoint-100/pytorch_lora_weights.safetensors)"
assert_contains "${upload_all}" "archive:outputs/checkpoints/checkpoint-50/pytorch_lora_weights.safetensors"
assert_contains "${upload_all}" "archive:outputs/checkpoints/checkpoint-100/pytorch_lora_weights.safetensors"
assert_contains "${upload_all}" "<food-studio-g4:/tmp/food-studio-loras.tar.gz>"
rm -f "${fixture}/outputs/checkpoints/checkpoint-100/pytorch_lora_weights.safetensors"
if load_develop_weights >/dev/null 2>"${fixture}/missing.txt"; then
  echo "load_develop_weights accepted a missing checkpoint" >&2
  exit 1
fi
assert_contains "$(cat "${fixture}/missing.txt")" "Missing LoRA weights for checkpoint(s): 100"
ROOT="${saved_root}"
rm -rf "${fixture}"

log_file="$(mktemp)"
printf '%s\n' "==== development job start " "==== development renders exit 0 " > "${log_file}"
assert_eq "$(development_outcome_from_log "${log_file}")" "complete"
printf '%s\n' "==== development job start " "==== development renders exit 1 " > "${log_file}"
assert_eq "$(development_outcome_from_log "${log_file}")" "failed"
printf '%s\n' "==== development job start older" "==== development renders exit 0 " "==== development job start newer" > "${log_file}"
assert_eq "$(development_outcome_from_log "${log_file}")" "incomplete"
rm -f "${log_file}"

assert_eq "$(parse_eval_args --step 700)" "700"
assert_eq "$(parse_eval_args --step 700 --force)" "700 --force"
assert_eq "$(eval_resume_args "700 --force")" "700"
assert_eq "$(eval_resume_args "700")" "700"
if parse_eval_args --step 0 >/dev/null 2>&1; then
  echo "parse_eval_args accepted step 0" >&2
  exit 1
fi
if parse_eval_args --step abc >/dev/null 2>&1; then
  echo "parse_eval_args accepted a non-numeric step" >&2
  exit 1
fi

selection_dir="$(mktemp -d)"
saved_selection_root="${ROOT}"
ROOT="${selection_dir}"
if selected_checkpoint_step >/dev/null 2>&1; then
  echo "selected_checkpoint_step accepted a missing selection file" >&2
  exit 1
fi
mkdir -p "${selection_dir}/outputs/reviews"
printf '%s\n' '{"selectedStep": "700"}' > "${selection_dir}/outputs/reviews/checkpoint-selection.json"
assert_eq "$(selected_checkpoint_step)" "700"
assert_eq "$(parse_eval_args)" "700"
assert_eq "$(parse_eval_args --force)" "700 --force"
ROOT="${saved_selection_root}"
rm -rf "${selection_dir}"

eval_log="$(mktemp)"
printf '%s\n' "==== eval job start " "==== eval renders exit 0 " > "${eval_log}"
assert_eq "$(eval_outcome_from_log "${eval_log}")" "complete"
printf '%s\n' "==== eval job start " "==== eval renders exit 1 " > "${eval_log}"
assert_eq "$(eval_outcome_from_log "${eval_log}")" "failed"
printf '%s\n' "==== eval job start older" "==== eval renders exit 0 " "==== eval job start newer" > "${eval_log}"
assert_eq "$(eval_outcome_from_log "${eval_log}")" "incomplete"
rm -f "${eval_log}"

help_text="$(main help)"
assert_contains "${help_text}" "develop [--name <run-name>] [--force]"
assert_contains "${help_text}" "eval [--step <n>] [--force]"

echo "gcp profile tests passed"
