#!/usr/bin/env bash

shell_join() {
  if [[ $# -lt 2 ]]; then
    echo "shell_join requires an output variable and at least one argument" >&2
    return 2
  fi
  local output_variable="$1"
  local joined
  shift
  printf -v joined '%q ' "$@"
  printf -v "${output_variable}" '%s' "${joined% }"
}
