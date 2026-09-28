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

echo "gcp profile tests passed"
