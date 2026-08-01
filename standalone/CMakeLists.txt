cmake_minimum_required(VERSION 3.16)
project(yolo_aim_standalone VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(YA_BUILD_CAPTURE_SHARED "Build yolo_capture as shared library" ON)
option(YA_WITH_WGC "Build Windows Graphics Capture backend" ON)
option(YA_WITH_INFER "Link ONNX inference from parent repo" ON)
option(YA_WITH_AIM "Link full parent aim/mouse stack (OBS parity)" ON)
option(YA_WITH_CUDA_EP "Define HAVE_ONNXRUNTIME_CUDA_EP / TENSORRT_EP" ON)

if(NOT WIN32)
  message(FATAL_ERROR "standalone host currently targets Windows only")
endif()

set(YA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(YA_PARENT ${YA_ROOT}/..)
set(SC_DIR ${YA_ROOT}/third_party/screen_capture)

# ---- screen capture sources (vendored) ----
set(SC_SOURCES
  ${SC_DIR}/DXScreenCapture.cpp
  ${SC_DIR}/GDIScreenCapture.cpp
  ${SC_DIR}/ScreenCapture.cpp
  ${SC_DIR}/ScreenCaptureFactory.cpp
)
if(YA_WITH_WGC)
  list(APPEND SC_SOURCES ${SC_DIR}/WGCScreenCapture.cpp)
  add_compile_definitions(YA_WITH_WGC=1)
endif()

set(CAPTURE_SOURCES
  ${SC_SOURCES}
  ${YA_ROOT}/src/capture/ycap_c_api.cpp
)

if(YA_BUILD_CAPTURE_SHARED)
  add_library(yolo_capture SHARED ${CAPTURE_SOURCES})
  target_compile_definitions(yolo_capture PRIVATE YALO_CAPTURE_EXPORTS SCREENCAPTURE_EXPORTS)
else()
  add_library(yolo_capture STATIC ${CAPTURE_SOURCES})
  target_compile_definitions(yolo_capture PUBLIC YALO_CAPTURE_STATIC SCREENCAPTURE_EXPORTS)
  target_compile_definitions(yolo_capture INTERFACE YCAP_STATIC_LINK)
endif()

target_include_directories(yolo_capture
  PUBLIC
    ${YA_ROOT}/include
    ${SC_DIR}
)
target_link_libraries(yolo_capture PUBLIC d3d11 dxgi)
if(YA_WITH_WGC)
  target_link_libraries(yolo_capture PUBLIC windowsapp dwmapi)
  target_compile_options(yolo_capture PRIVATE /await)
endif()

# ---- host launcher (M1 new architecture: app/engine, core reuse) ----
set(HOST_SOURCES
  ${YA_ROOT}/src/app/main.cpp
  ${YA_ROOT}/src/engine/Engine.cpp
  ${YA_ROOT}/src/config/ConfigModel.cpp
  ${YA_ROOT}/src/config/ConfigKeyMap.cpp
  ${YA_ROOT}/src/config/ConfigStore.cpp
  ${YA_ROOT}/src/capture/FrameSource.cpp
  ${YA_ROOT}/src/core/obs_stubs.cpp
)

if(YA_WITH_INFER)
  list(APPEND HOST_SOURCES
    ${YA_ROOT}/src/core/InferEngine.cpp
    ${YA_PARENT}/src/models/ModelYOLO.cpp
    ${YA_PARENT}/src/models/DmlPreprocessor.cpp
  )

  set(OpenCV_DIR "${YA_PARENT}/libs/opencv")
  list(APPEND CMAKE_PREFIX_PATH "${OpenCV_DIR}")
  include("${OpenCV_DIR}/OpenCVConfig.cmake")

  set(ONNXRUNTIME_DIR "${YA_PARENT}/onnxruntime")
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
  )
endif()

if(YA_WITH_AIM)
  list(APPEND HOST_SOURCES
    ${YA_ROOT}/src/aim/FullAimBridge.cpp
    ${YA_PARENT}/src/AbstractMouseController.cpp
    ${YA_PARENT}/src/MouseControllerFactory.cpp
    ${YA_PARENT}/src/MouseController.cpp
    ${YA_PARENT}/src/MAKCUMouseController.cpp
    ${YA_PARENT}/src/LogiDriverMouseController.cpp
    ${YA_PARENT}/src/logi_driver.cpp
    ${YA_PARENT}/src/GvInputMouseController.cpp
    ${YA_PARENT}/src/NtUserSendInputMouseController.cpp
    ${YA_PARENT}/src/NtUserInjectMouseController.cpp
    ${YA_PARENT}/src/NtUserInjectPointerController.cpp
    ${YA_PARENT}/src/DerivativePredictor.cpp
    ${YA_PARENT}/src/SmithPredictor.cpp
    ${YA_PARENT}/src/SlewRateController.cpp
    ${YA_PARENT}/src/OneEuroFilter.cpp
    ${YA_PARENT}/src/aim_controller.cpp
    ${YA_PARENT}/src/mpid.cpp
    ${YA_PARENT}/src/curve.cpp
    ${YA_PARENT}/src/utils/GhostTracker.cpp
    ${YA_PARENT}/src/HungarianAlgorithm.cpp
    ${YA_PARENT}/src/RecoilPatternManager.cpp
    ${YA_PARENT}/src/CrosshairDetector.cpp
  )
  # GvInput uses std::wstring(DevicePath); needs UNICODE detail struct
  set_source_files_properties(
    ${YA_PARENT}/src/GvInputMouseController.cpp
    PROPERTIES COMPILE_DEFINITIONS "UNICODE;_UNICODE"
  )
endif()

add_executable(yolo_host ${HOST_SOURCES})

target_include_directories(yolo_host PRIVATE
  ${YA_ROOT}/include
  ${YA_ROOT}/src
  ${YA_ROOT}/src/app
  ${YA_ROOT}/src/engine
  ${YA_ROOT}/src/config
  ${YA_ROOT}/src/capture
  ${YA_ROOT}/src/host
  ${YA_ROOT}/src/web
  ${YA_ROOT}/src/core
  ${YA_ROOT}/src/aim
  ${YA_ROOT}/third_party
)

target_link_libraries(yolo_host PRIVATE yolo_capture ws2_32 shell32 user32)
target_compile_options(yolo_host PRIVATE /utf-8)

if(YA_WITH_INFER)
  target_include_directories(yolo_host PRIVATE
    ${YA_PARENT}/src
    ${YA_PARENT}/src/models
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
  )
  target_include_directories(yolo_host BEFORE PRIVATE ${YA_ROOT}/include)

  target_compile_definitions(yolo_host PRIVATE YA_WITH_INFER=1)
  if(YA_WITH_CUDA_EP)
    if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.lib")
      target_compile_definitions(yolo_host PRIVATE HAVE_ONNXRUNTIME_CUDA_EP)
      message(STATUS "CUDA EP enabled for standalone")
    endif()
    if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_tensorrt.lib")
      target_compile_definitions(yolo_host PRIVATE HAVE_ONNXRUNTIME_TENSORRT_EP)
      message(STATUS "TensorRT EP enabled for standalone")
    endif()
  endif()

  # GPU preprocess path (same as parent OBS plugin HAVE_CUDA):
  # compile CudaPreprocessor.cu with nvcc + define HAVE_CUDA so ModelYOLO can
  # cudaMalloc device input and skip pure-CPU preprocess where possible.
  option(YA_WITH_CUDA_PRE "Enable HAVE_CUDA GPU preprocess (nvcc CudaPreprocessor.cu)" ON)
  if(YA_WITH_CUDA_PRE)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(_ya_cuda_src "${YA_PARENT}/src/models/CudaPreprocessor.cu")
      set(_ya_cuda_obj "${CMAKE_CURRENT_BINARY_DIR}/CudaPreprocessor.cuda.obj")
      get_filename_component(_ya_msvc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
      # Force MSVC 14.44 for nvcc (same workaround as parent CMakeLists)
      set(_ya_vc_tools "E:/VisualStudio/chanping/VC/Tools/MSVC/14.44.35207")
      if(NOT EXISTS "${_ya_vc_tools}")
        set(_ya_vc_tools "")
      endif()
      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_ya_cuda_compile.cmake"
        "if(EXISTS \"${_ya_vc_tools}\")\n"
        "  set(ENV{VCToolsVersion} \"14.44.35207\")\n"
        "  set(ENV{VCToolsInstallDir} \"${_ya_vc_tools}/\")\n"
        "endif()\n"
        "execute_process(\n"
        "  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}\n"
        "    -c \"${_ya_cuda_src}\"\n"
        "    -o \"${_ya_cuda_obj}\"\n"
        "    -I \"${YA_PARENT}/src/models\"\n"
        "    -I \"${CUDAToolkit_INCLUDE_DIRS}\"\n"
        "    -ccbin \"${_ya_msvc_bin}\"\n"
        "    -Xcompiler \"/MD /Zi\"\n"
        "    -O2\n"
        "  RESULT_VARIABLE _nvcc_result\n"
        ")\n"
        "if(NOT _nvcc_result EQUAL 0)\n"
        "  message(FATAL_ERROR \"nvcc failed: \${_nvcc_result}\")\n"
        "endif()\n"
      )
      add_custom_command(
        OUTPUT "${_ya_cuda_obj}"
        COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/_ya_cuda_compile.cmake"
        DEPENDS "${_ya_cuda_src}" "${YA_PARENT}/src/models/CudaPreprocessor.cuh"
        COMMENT "nvcc CudaPreprocessor.cu (standalone HAVE_CUDA)"
        VERBATIM
      )
      set_source_files_properties("${_ya_cuda_obj}" PROPERTIES EXTERNAL_OBJECT TRUE)
      target_sources(yolo_host PRIVATE "${_ya_cuda_obj}")
      target_compile_definitions(yolo_host PRIVATE HAVE_CUDA)
      target_include_directories(yolo_host PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_link_libraries(yolo_host PRIVATE CUDA::cudart)
      message(STATUS "HAVE_CUDA preprocess enabled (nvcc: ${CUDAToolkit_NVCC_EXECUTABLE})")
    else()
      message(WARNING "YA_WITH_CUDA_PRE=ON but CUDAToolkit not found — preprocess stays CPU")
    endif()
  endif()

  target_link_libraries(yolo_host PRIVATE
    onnxruntime::onnxruntime
    OpenCV::opencv_core
    OpenCV::opencv_imgproc
    d3d11 dxgi
  )

  file(GLOB YA_ORT_DLLS
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_*.dll"
  )
  foreach(_dll ${YA_ORT_DLLS})
    add_custom_command(TARGET yolo_host POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_dll}" $<TARGET_FILE_DIR:yolo_host>
      COMMENT "Copy ${_dll}"
    )
  endforeach()

  # Optional: stage CUDA/cuDNN next to host so CUDA EP works without system PATH.
  # Override with -DYA_CUDNN_BIN=... -DYA_CUDA_BIN=...
  if(NOT YA_CUDNN_BIN)
    foreach(_c
        "D:/CUDATool/cudnn/bin/12.9/x64"
        "D:/CUDATool/cudnn/bin/13.1/x64"
        "$ENV{CUDNN_PATH}/bin/12.9/x64"
        "$ENV{CUDNN_PATH}")
      if(EXISTS "${_c}/cudnn64_9.dll")
        set(YA_CUDNN_BIN "${_c}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT YA_CUDA_BIN)
    foreach(_c
        "D:/CUDATool/bin"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
        "$ENV{CUDA_PATH}/bin")
      if(EXISTS "${_c}/cudart64_12.dll")
        set(YA_CUDA_BIN "${_c}")
        break()
      endif()
    endforeach()
  endif()
  if(YA_CUDNN_BIN)
    message(STATUS "Staging cuDNN from ${YA_CUDNN_BIN}")
    file(GLOB YA_CUDNN_DLLS "${YA_CUDNN_BIN}/cudnn*.dll")
    foreach(_dll ${YA_CUDNN_DLLS})
      add_custom_command(TARGET yolo_host POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_dll}" $<TARGET_FILE_DIR:yolo_host>
        COMMENT "Copy cudnn ${_dll}"
      )
    endforeach()
  endif()
  if(YA_CUDA_BIN)
    message(STATUS "Staging CUDA runtime from ${YA_CUDA_BIN}")
    foreach(_name
        cudart64_12.dll cublas64_12.dll cublasLt64_12.dll
        cufft64_11.dll curand64_10.dll cusolver64_11.dll cusparse64_12.dll
        nvJitLink_120_0.dll nvrtc-builtins64_125.dll nvrtc64_120_0.dll)
      if(EXISTS "${YA_CUDA_BIN}/${_name}")
        add_custom_command(TARGET yolo_host POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${YA_CUDA_BIN}/${_name}" $<TARGET_FILE_DIR:yolo_host>
          COMMENT "Copy CUDA ${_name}"
        )
      endif()
    endforeach()
  endif()
endif()

if(YA_WITH_AIM)
  target_compile_definitions(yolo_host PRIVATE YA_WITH_AIM=1)
  target_include_directories(yolo_host PRIVATE
    ${YA_PARENT}/src
    ${YA_PARENT}/src/models
    ${YA_PARENT}/src/utils
  )
  # nlohmann json (RecoilPatternManager)
  if(EXISTS "${YA_PARENT}/.deps/obs-deps-2025-07-11-x64/include/nlohmann/json.hpp")
    target_include_directories(yolo_host PRIVATE
      "${YA_PARENT}/.deps/obs-deps-2025-07-11-x64/include")
  elseif(EXISTS "${YA_PARENT}/libs/nlohmann/json.hpp")
    target_include_directories(yolo_host PRIVATE "${YA_PARENT}/libs")
  endif()
  # Eigen for curve.cpp neural path
  if(EXISTS "${YA_PARENT}/libs/eigen3/eigen/Eigen/Dense")
    target_include_directories(yolo_host PRIVATE "${YA_PARENT}/libs/eigen3/eigen")
  elseif(EXISTS "${YA_PARENT}/libs/eigen3/Eigen/Dense")
    target_include_directories(yolo_host PRIVATE "${YA_PARENT}/libs/eigen3")
  endif()
  # stubs win over OBS
  target_include_directories(yolo_host BEFORE PRIVATE ${YA_ROOT}/include)

  target_link_libraries(yolo_host PRIVATE setupapi hid)

  # OpenCV still needed for Detection.h / CrosshairDetector if infer off
  if(NOT YA_WITH_INFER)
    set(OpenCV_DIR "${YA_PARENT}/libs/opencv")
    include("${OpenCV_DIR}/OpenCVConfig.cmake")
    target_include_directories(yolo_host PRIVATE ${OpenCV_INCLUDE_DIRS} ${YA_PARENT}/src)
    target_link_libraries(yolo_host PRIVATE OpenCV::opencv_core OpenCV::opencv_imgproc)
  else()
    # Crosshair uses imgproc
    target_link_libraries(yolo_host PRIVATE OpenCV::opencv_imgproc)
  endif()

  message(STATUS "YA_WITH_AIM=ON — full OBS mouse/aim stack linked")
endif()

# Copy webui + config next to exe after build
add_custom_command(TARGET yolo_host POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${YA_ROOT}/webui $<TARGET_FILE_DIR:yolo_host>/webui
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${YA_ROOT}/config $<TARGET_FILE_DIR:yolo_host>/config
  COMMENT "Copy webui/ and config/ beside yolo_host"
)

set(YA_DEFAULT_MODEL "${YA_PARENT}/NCNN/251218Val_Bing_red_v11s_320.onnx")
if(EXISTS "${YA_DEFAULT_MODEL}")
  add_custom_command(TARGET yolo_host POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:yolo_host>/models
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${YA_DEFAULT_MODEL}" $<TARGET_FILE_DIR:yolo_host>/models/yolo.onnx
    COMMENT "Copy default YOLO onnx model"
  )
endif()

message(STATUS "yolo_aim_standalone: capture=${YA_BUILD_CAPTURE_SHARED} wgc=${YA_WITH_WGC} infer=${YA_WITH_INFER} aim=${YA_WITH_AIM}")

# ---- M2 config layer unit test ----
if(EXISTS "${YA_PARENT}/.deps/obs-deps-2025-07-11-x64/include/nlohmann/json.hpp")
  set(_nlohmann_inc "${YA_PARENT}/.deps/obs-deps-2025-07-11-x64/include")
elseif(EXISTS "${YA_PARENT}/libs/nlohmann/json.hpp")
  set(_nlohmann_inc "${YA_PARENT}/libs")
endif()
add_executable(yolo_config_test
  ${YA_ROOT}/test/config_test.cpp
  ${YA_ROOT}/src/config/ConfigModel.cpp
  ${YA_ROOT}/src/config/ConfigKeyMap.cpp
  ${YA_ROOT}/src/config/ConfigStore.cpp
)
target_include_directories(yolo_config_test PRIVATE
  ${YA_ROOT}/src
  ${YA_ROOT}/src/config
  ${YA_PARENT}/src
  ${_nlohmann_inc}
  "${YA_PARENT}/libs/opencv/x64/include"
)
target_compile_options(yolo_config_test PRIVATE /utf-8)
