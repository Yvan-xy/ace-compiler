#!/bin/bash
set -euo pipefail
set -x

APP_ROOT=/app
DATASET_DIR="${APP_ROOT}/fhe-cmplr/rtlib/ant/dataset"
RESNET_GEN_C="${APP_ROOT}/resnet20_cifar10_pre.c"
RESNET_DATASET_INC="${DATASET_DIR}/resnet20_cifar10_pre.onnx.inc"
WORK_DIR="${APP_ROOT}/tmp_a_sh_resnet20"
BACKUP_RESNET_INC="${WORK_DIR}/resnet20_cifar10_pre.onnx.inc.orig"

mkdir -p "${WORK_DIR}"

if [[ -f "${RESNET_DATASET_INC}" ]]; then
  cp "${RESNET_DATASET_INC}" "${BACKUP_RESNET_INC}"
fi

cleanup() {
  if [[ -f "${BACKUP_RESNET_INC}" ]]; then
    cp "${BACKUP_RESNET_INC}" "${RESNET_DATASET_INC}"
  fi
}
trap cleanup EXIT

cp "${RESNET_GEN_C}" "${RESNET_DATASET_INC}"

c++ /app/fhe-cmplr/rtlib/ant/dataset/resnet20_cifar10.cxx -DRTLIB_SUPPORT_LINUX -I /usr/local/include -I /usr/local/rtlib/include/ -I /usr/local/rtlib/include/ant/ -O3 -DNDEBUG -std=gnu++17 -fopenmp /usr/local/rtlib/lib/libFHErt_ant.a /usr/local/rtlib/lib/libFHErt_common.a /usr/local/lib/libAIRutil.a -lgmp -lm -o /app/resnet20_cifar10_pre.ace -lgomp
time /app/resnet20_cifar10_pre.ace /app/dataset/test_batch.bin 0 0

