#=============================================================================
#
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

function(fetch_blake2)
  set(REPO_BLAKE2_URL "https://github.com/BLAKE2/BLAKE2.git")
  set(BLAKE2_GIT_TAG "ed1974ea83433eba7b2d95c5dcd9ac33cb847913")

  message(STATUS "Cloning External Repository   : ${REPO_BLAKE2_URL}")

  include(FetchContent)

  FetchContent_Declare(
      blake2
      GIT_REPOSITORY ${REPO_BLAKE2_URL}
      GIT_TAG ${BLAKE2_GIT_TAG}
  )
  FetchContent_MakeAvailable(blake2)

  set (BLAKE2_DIR "${blake2_SOURCE_DIR}/ref/")
  file (GLOB_RECURSE BLAKE2_SRC_FILES 
        CONFIGURE_DEPENDS ${BLAKE2_DIR}/blake2b-ref.c ${BLAKE2_DIR}/blake2xb-ref.c)

  include_directories(${BLAKE2_DIR})

  set(BLAKE2_SRC_FILES ${BLAKE2_SRC_FILES} PARENT_SCOPE)
endfunction()

if(NOT TARGET blake2)
  fetch_blake2()
endif()
