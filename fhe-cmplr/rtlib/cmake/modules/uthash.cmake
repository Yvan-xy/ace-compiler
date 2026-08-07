#=============================================================================
#
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

function(fetch_uthash)

  set(REPO_UTHASH_URL "https://github.com/troydhanson/uthash.git")
  set(UTHASH_GIT_TAG "a49bed0b4abb7dff16c73906dcdc8a9718d582d2")

  message(STATUS "Cloning External Repository   : ${REPO_UTHASH_URL}")

  include(FetchContent)
  FetchContent_Declare(
      uthash
      GIT_REPOSITORY ${REPO_UTHASH_URL}
      GIT_TAG ${UTHASH_GIT_TAG}
  )
  FetchContent_MakeAvailable(uthash)

  include_directories(${uthash_SOURCE_DIR}/src)

  install(FILES ${uthash_SOURCE_DIR}/src/uthash.h DESTINATION include/rtlib)
endfunction()

if(NOT TARGET uthash)
  fetch_uthash()
endif()

