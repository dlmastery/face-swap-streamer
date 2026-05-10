cmake_minimum_required(VERSION 3.20)
project(faceswap-cli LANGUAGES CXX)

# ---- Project-wide settings -----------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:__cplusplus /MP)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Oi /Ot /GL /DNDEBUG)
        add_link_options(/LTCG)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG -ffast-math)
    endif()
endif()

# ---- Dependencies --------------------------------------------------------
include(FetchContent)

# CLI11 — argparse, single header, MIT
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(CLI11)

# fmt — formatting library, used for status output
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fmt)

# OpenCV — image I/O + affine warp + paste_back blending. Located via
# either OpenCV_DIR env var (vcpkg integration sets this), or a manual
# path passed at configure time via -DOpenCV_DIR=...
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio calib3d dnn)
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_INCLUDE_DIRS}")

# ONNX Runtime — neural network inference. We don't use vcpkg's onnxruntime
# port because it doesn't ship the CUDA EP DLLs. Instead we point at the
# DLLs already present in the conda dlc env (or a manual download).
# Configure with -DORT_ROOT=<path> if it can't auto-detect.
set(ORT_ROOT "" CACHE PATH "ONNX Runtime install root (containing include/ and lib/)")
if(NOT ORT_ROOT)
    file(GLOB _ort_candidates
        "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-gpu-*"
        "$ENV{USERPROFILE}/onnxruntime")
    foreach(c ${_ort_candidates})
        if(EXISTS "${c}/include/onnxruntime_cxx_api.h")
            set(ORT_ROOT "${c}")
            break()
        endif()
    endforeach()
endif()
if(NOT ORT_ROOT)
    message(FATAL_ERROR "ONNX Runtime not found. Pass -DORT_ROOT=<path> or run scripts/setup.ps1")
endif()
message(STATUS "ORT_ROOT: ${ORT_ROOT}")

# ---- Targets --------------------------------------------------------------
add_executable(faceswap
    src/main.cpp
    src/face_analyser.cpp
    src/inswapper.cpp
    src/reference.cpp
    src/pipeline.cpp
    src/ffmpeg_io.cpp
    src/onnx_session.cpp
)
target_include_directories(faceswap PRIVATE
    include
    "${ORT_ROOT}/include"        # if ORT_ROOT points at install root
    "${ORT_ROOT}"                # if ORT_ROOT points at conda capi/ which has the headers next to dlls? we'll handle in setup.ps1
)
target_link_libraries(faceswap PRIVATE
    CLI11::CLI11
    fmt::fmt
    ${OpenCV_LIBS}
)

# ONNX Runtime: link against onnxruntime.lib and copy the DLLs next to the exe.
target_link_directories(faceswap PRIVATE "${ORT_ROOT}/lib" "${ORT_ROOT}")
target_link_libraries(faceswap PRIVATE onnxruntime)

# Copy ORT DLLs + OpenCV DLLs alongside the exe so it runs from the build dir
# without PATH gymnastics.
if(WIN32)
    file(GLOB ORT_DLLS "${ORT_ROOT}/lib/*.dll" "${ORT_ROOT}/*.dll")
    foreach(_dll ${ORT_DLLS})
        add_custom_command(TARGET faceswap POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" $<TARGET_FILE_DIR:faceswap>)
    endforeach()

    # OpenCV runtime DLLs — official Windows pack lays them at
    # <OpenCV_DIR>/x64/vc16/bin/  (OpenCV_DIR is .../build/)
    file(GLOB CV_DLLS "${OpenCV_DIR}/x64/vc16/bin/opencv_world*.dll"
                      "${OpenCV_DIR}/x64/vc17/bin/opencv_world*.dll"
                      "${OpenCV_DIR}/x64/vc16/bin/opencv_videoio_*.dll"
                      "${OpenCV_DIR}/x64/vc17/bin/opencv_videoio_*.dll")
    foreach(_dll ${CV_DLLS})
        add_custom_command(TARGET faceswap POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" $<TARGET_FILE_DIR:faceswap>)
    endforeach()
endif()

# ---- Install --------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS faceswap RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
