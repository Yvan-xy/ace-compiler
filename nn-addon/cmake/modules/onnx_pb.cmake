#=============================================================================
#
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

# Build external onnx project dependent function
function(build_external_proto)

  set(REPO_ONNX_URL "https://github.com/onnx/onnx.git")
  set(ONNX_GIT_TAG "1089b9e8045a3a2882d7bb6a1dbaeaf9cae131da")

  message(STATUS "Cloning External Repository   : ${REPO_ONNX_URL}")

  include(FetchContent)
  FetchContent_Declare(
    onnx
    GIT_REPOSITORY  ${REPO_ONNX_URL}
    GIT_TAG         ${ONNX_GIT_TAG}
    GIT_SUBMODULES  ""
  )

  FetchContent_GetProperties(onnx)
  if(NOT onnx_POPULATED)
      FetchContent_Populate(onnx)
  endif()

  set(ONNX_PATH   ${onnx_SOURCE_DIR}/onnx/)
  set(ONNX_PROTO  ${onnx_SOURCE_DIR}/onnx/onnx.proto)
  set(ONNX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx)

  execute_process(
    COMMAND mkdir -p ${ONNX_OUTPUT}
    DEPEND  onnx
    COMMAND protoc --proto_path=${ONNX_PATH} --cpp_out=${ONNX_OUTPUT} ${ONNX_PROTO}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    message(STATUS "Generating onnx proto with Protobuf")

  add_library(onnx_objects OBJECT
    ${ONNX_OUTPUT}/onnx.pb.cc
  )
  set_property(TARGET onnx_objects PROPERTY POSITION_INDEPENDENT_CODE 1)

  include_directories (${ONNX_OUTPUT})
endfunction()

if(NOT TARGET onnx_objects)
  build_external_proto()
endif()
