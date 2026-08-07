#=============================================================================
#
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

# Build external unittest project dependent function
function(build_external_unittest)

  set(REPO_UNITTEST_URL "https://github.com/google/googletest.git")
  set(UNITTEST_GIT_TAG "f8d7d77c06936315286eb55f8de22cd23c188571")

  message(STATUS "Cloning External Repository    : ${REPO_UNITTEST_URL}")

  include(ExternalProject)
  ExternalProject_Add(
    unittest
    GIT_REPOSITORY ${REPO_UNITTEST_URL}
    GIT_TAG ${UNITTEST_GIT_TAG}
    PREFIX ${CMAKE_BINARY_DIR}/external
    CMAKE_ARGS  -DCMAKE_BUILD_TYPE=Release
    UPDATE_COMMAND ""
    BUILD_ALWAYS OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/external/src/unittest-build/lib/libgtest.a
  )
  ExternalProject_Get_Property(unittest SOURCE_DIR BINARY_DIR)

  add_library(gtest IMPORTED STATIC GLOBAL)
  set_target_properties(gtest PROPERTIES
    IMPORTED_LOCATION ${BINARY_DIR}/lib/libgtest.a
  )
  add_inc_list("${SOURCE_DIR}/googletest/include")
  add_dependencies(gtest unittest)

  set(gtest gtest PARENT_SCOPE)
  set(ENV{UNITTEST_INCLUDE_DIR} ${SOURCE_DIR}/googletest/include)

  install(DIRECTORY ${SOURCE_DIR}/googletest/include/ DESTINATION include)
  install(FILES ${BINARY_DIR}/lib/libgtest.a DESTINATION lib)
endfunction()

if(NOT TARGET gtest)
	build_external_unittest()
  add_dependencies(air_depend gtest)
  set(UT_LIBS "gtest;pthread" CACHE STRING "Global common libs of unittest")
endif()
