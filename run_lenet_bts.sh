#!/usr/bin/env bash
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BTS_SUITE_NAME="LeNet"
export BTS_MODEL_CHOICES="lenet lenet_x2"
export BTS_DEFAULT_WORK_DIR="tmp_lenet_bts"
export BTS_ENTRYPOINT="run_lenet_bts.sh"
exec "${SCRIPT_DIR}/run_model_and_main_bts.sh" "$@"
