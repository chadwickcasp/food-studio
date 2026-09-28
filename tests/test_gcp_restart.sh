#!/usr/bin/env bash
# Decision tests for host-termination restart. No gcloud calls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/gcp.sh"

assert_eq() {
  local got="$1" want="$2" label="$3"
  if [[ "${got}" != "${want}" ]]; then
    echo "${label}: got '${got}', want '${want}'" >&2
    exit 1
  fi
}

action="$(printf '%s\n' $'compute.instances.preempted\tInstance was preempted.' $'start\t' | classify_operation_lines)"
assert_eq "${action}" restart "newest preemption"

action="$(printf '%s\n' $'start\t' $'compute.instances.preempted\tInstance was preempted.' | classify_operation_lines)"
assert_eq "${action}" restart "skip newer start operations"

action="$(printf '%s\n' $'stop\t' $'compute.instances.preempted\tInstance was preempted.' | classify_operation_lines)"
assert_eq "${action}" leave "voluntary stop is newer than preemption"

action="$(printf '%s\n' $'compute.instances.hostError\tInstance terminated due to host error.' | classify_operation_lines)"
assert_eq "${action}" restart "host error"

action="$(printf '%s\n' $'compute.instances.migrate\tTerminated for host maintenance.' | classify_operation_lines)"
assert_eq "${action}" restart "host maintenance message"

action="$(printf '%s\n' $'start\t' | classify_operation_lines)"
assert_eq "${action}" unknown "start alone"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

cat >"${tmp}" <<'EOF'
==== train start old ====
==== train exit 0 old ====
==== development renders exit 0 old ====
==== train start new ====
Steps: 501/1000
EOF
assert_eq "$(job_outcome_from_log "${tmp}")" incomplete "killed mid-run"

cat >"${tmp}" <<'EOF'
==== train start new ====
==== train exit 0 new ====
==== development renders exit 0 new ====
EOF
assert_eq "$(job_outcome_from_log "${tmp}")" complete "finished job"

cat >"${tmp}" <<'EOF'
==== train start new ====
==== train exit 1 new ====
EOF
assert_eq "$(job_outcome_from_log "${tmp}")" failed "train error"

cat >"${tmp}" <<'EOF'
==== train start new ====
==== train exit 0 new ====
EOF
assert_eq "$(job_outcome_from_log "${tmp}")" incomplete "renders not finished"

cat >"${tmp}" <<'EOF'
==== train start new ====
==== train exit 0 new ====
==== development renders exit 1 new ====
EOF
assert_eq "$(job_outcome_from_log "${tmp}")" failed "render error"

assert_eq "$(parse_resume_arg)" "" "default resume"
assert_eq "$(parse_resume_arg --resume latest)" "--resume latest" "resume latest"
if parse_resume_arg --resume 'latest;rm'; then
  echo "unsafe resume was accepted" >&2
  exit 1
fi

echo "gcp restart decision tests passed"
