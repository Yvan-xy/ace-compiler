#=============================================================================
#
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

# Build external Phantom project dependent function
function(build_external_phantom)
  set(PHANTOM_SOURCE_DIR "" CACHE PATH
      "Local Git repository used to clone the pinned Phantom source")
  set(PHANTOM_GIT_TAG "92acb4a2661f4a59c74545cb90d88b248c5c8b07"
      CACHE STRING "Exact Phantom Git commit")
  option(PHANTOM_SOURCE_SNAPSHOT
         "Use a preverified source-only Phantom snapshot" OFF)

  string(LENGTH "${PHANTOM_GIT_TAG}" PHANTOM_GIT_TAG_LENGTH)
  if(NOT PHANTOM_GIT_TAG_LENGTH EQUAL 40 OR
     NOT PHANTOM_GIT_TAG MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "PHANTOM_GIT_TAG must be an exact 40-character commit")
  endif()
  if(PHANTOM_SOURCE_SNAPSHOT)
    foreach(PHANTOM_REQUIRED_PATH CMakeLists.txt include src)
      if(NOT EXISTS "${PHANTOM_SOURCE_DIR}/${PHANTOM_REQUIRED_PATH}")
        message(FATAL_ERROR
          "Phantom source snapshot is missing ${PHANTOM_REQUIRED_PATH}")
      endif()
    endforeach()
    if(EXISTS "${PHANTOM_SOURCE_DIR}/.git")
      message(FATAL_ERROR "Phantom source snapshot must not contain .git")
    endif()
  else()
    if(NOT IS_DIRECTORY "${PHANTOM_SOURCE_DIR}/.git")
      message(FATAL_ERROR
        "PHANTOM_SOURCE_DIR must name a mounted local Git repository")
    endif()
    execute_process(
      COMMAND git -C "${PHANTOM_SOURCE_DIR}" cat-file -e
              "${PHANTOM_GIT_TAG}^{commit}"
      RESULT_VARIABLE PHANTOM_COMMIT_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)
    if(NOT PHANTOM_COMMIT_RESULT EQUAL 0)
      message(FATAL_ERROR
        "Pinned Phantom commit is absent from PHANTOM_SOURCE_DIR")
    endif()
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "80")
    message(FATAL_ERROR
      "Phantom qualification requires CMAKE_CUDA_ARCHITECTURES=80")
  endif()

  message(STATUS "Phantom source repository     : ${PHANTOM_SOURCE_DIR}")
  message(STATUS "Phantom pinned commit         : ${PHANTOM_GIT_TAG}")
  message(STATUS "Phantom source snapshot       : ${PHANTOM_SOURCE_SNAPSHOT}")

  include(ExternalProject)
  set(PHANTOM_EXTERNAL_ARGUMENTS
      PREFIX ${CMAKE_BINARY_DIR}/external
      UPDATE_COMMAND ""
      BUILD_ALWAYS OFF
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
                 -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
                 -DCMAKE_CXX_STANDARD=17
                 -DCMAKE_CXX_STANDARD_REQUIRED=ON
                 -DCMAKE_CUDA_STANDARD=17
                 -DCMAKE_CUDA_STANDARD_REQUIRED=ON
                 -DPHANTOM_BUILD_EXAMPLES=OFF
                 -DPHANTOM_BUILD_TESTS=OFF
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target phantom_ordinary
      # This argument list is expanded into ExternalProject_Add.  An empty
      # string is dropped during list expansion and silently restores the
      # default `cmake --build . --target install`, which pulls the native-BTS
      # and CNN partitions into an ordinary/retained qualification build.
      INSTALL_COMMAND ${CMAKE_COMMAND} -E true
      BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/external/src/phantom_external-build/lib/libphantom_ordinary.a)
  if(PHANTOM_SOURCE_SNAPSHOT)
    ExternalProject_Add(
      phantom_external
      SOURCE_DIR ${PHANTOM_SOURCE_DIR}
      DOWNLOAD_COMMAND ""
      ${PHANTOM_EXTERNAL_ARGUMENTS})
  else()
    ExternalProject_Add(
      phantom_external
      GIT_REPOSITORY ${PHANTOM_SOURCE_DIR}
      GIT_TAG ${PHANTOM_GIT_TAG}
      GIT_SHALLOW OFF
      ${PHANTOM_EXTERNAL_ARGUMENTS})
  endif()
  ExternalProject_Get_Property(phantom_external SOURCE_DIR BINARY_DIR)

  find_package(CUDAToolkit REQUIRED)
  find_library(CUDA_DEVICE_RUNTIME_LIBRARY NAMES cudadevrt
    HINTS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_ROOT}/lib64"
    REQUIRED)
  find_library(NTL_LIBRARY NAMES ntl REQUIRED)
  find_library(GMP_LIBRARY NAMES gmp REQUIRED)
  find_library(GMPXX_LIBRARY NAMES gmpxx REQUIRED)

  add_library(phantom_ordinary IMPORTED STATIC GLOBAL)
  set_target_properties(phantom_ordinary PROPERTIES
    IMPORTED_LOCATION ${BINARY_DIR}/lib/libphantom_ordinary.a
    INTERFACE_LINK_LIBRARIES
      "${NTL_LIBRARY};${GMPXX_LIBRARY};${GMP_LIBRARY};${CUDA_DEVICE_RUNTIME_LIBRARY};CUDA::cudart"
  )
  include_directories(${SOURCE_DIR}/include)
  add_dependencies(phantom_ordinary phantom_external)

  set(phantom phantom_ordinary PARENT_SCOPE)
  set(ENV{PHANTOM_INCLUDE_DIR} ${SOURCE_DIR}/include)
  set(PHANTOM_LIBS
      phantom_ordinary ${NTL_LIBRARY} ${GMPXX_LIBRARY} ${GMP_LIBRARY}
      ${CUDA_DEVICE_RUNTIME_LIBRARY} CUDA::cudart PARENT_SCOPE)
endfunction()
