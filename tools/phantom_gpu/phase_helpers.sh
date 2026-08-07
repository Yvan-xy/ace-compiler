#!/usr/bin/env bash

# Run a phase with errexit enabled inside an isolated shell while preserving the
# caller's ability to record the failing status before returning it.
run_timed_phase() {
  local timings_file="$1"
  local phase_name="$2"
  shift 2
  local phase_started phase_ended phase_exit
  local caller_errexit=false
  [[ $- == *e* ]] && caller_errexit=true
  phase_started="$(date +%s)"
  set +e
  (
    set -e
    "$@"
  )
  phase_exit=$?
  phase_ended="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${phase_name}" "${phase_started}" "${phase_ended}" \
    "$((phase_ended - phase_started))" "${phase_exit}" >>"${timings_file}"
  if [[ "${caller_errexit}" == true ]]; then
    set -e
  else
    set +e
  fi
  return "${phase_exit}"
}
