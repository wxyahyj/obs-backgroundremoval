cmake_minimum_required(VERSION 3.16)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" ON)
option(ENABLE_QT "Use Qt functionality" ON)
option(ENABLE_CUDA "Enable CUDA support for GPU texture preprocessing" ON)

set(VCPKG_TARGET_TRIPLET "" CACHE STRING "Vcpkg target triplet to use")
option(USE_PKGCONFIG "Use pkg-config to find dependencies" ON)
option(USE_SYSTEM_ONNXRUNTIME "Use system ONNX Runtime" ON)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core Svg REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets Qt6::Svg)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
  if(WIN32)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_QT)
  endif()
  
  # 闆嗘垚QtFluentWidgets UI搴?(MIT璁稿彲璇?
  set(FLUENT_BUILD_DESIGNER_PLUGIN OFF CACHE BOOL "Disable QtFluentWidgets Designer plugin" FORCE)
  add_subdirectory(${CMAKE_SOURCE_DIR}/QtFluentWidgets-master ${CMAKE_BINARY_DIR}/QtFluentWidgets)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE QtFluentWidgets)
  message(STATUS "QtFluentWidgets UI library integrated")
endif()

if(NOT VCPKG_TARGET_TRIPLET STREQUAL "")
  # Use vcpkg to find dependencies (Official binary)

  message(STATUS "Using vcpkg with target triplet: ${VCPKG_TARGET_TRIPLET}")

  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")

  find_package(CURL CONFIG REQUIRED)
  find_package(OpenCV CONFIG REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
  # Eigen3鐢眂reeper-qt鎻愪緵锛屼笉鍐嶅崟鐙煡鎵?
  if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG REQUIRED)
  endif()
elseif(USE_PKGCONFIG)
  # Use pkg-config to find dependencies (Arch Linux, etc.)

  message(STATUS "Using pkg-config to find dependencies")

  find_package(PkgConfig REQUIRED)

  # CURL
  pkg_check_modules(PC_CURL REQUIRED libcurl)
  add_library(CURL::libcurl INTERFACE IMPORTED)
  target_link_libraries(CURL::libcurl INTERFACE ${PC_CURL_LIBRARIES})
  target_include_directories(CURL::libcurl INTERFACE ${PC_CURL_INCLUDE_DIRS})
  target_compile_definitions(CURL::libcurl INTERFACE ${PC_CURL_CFLAGS_OTHER})
  target_link_directories(CURL::libcurl INTERFACE ${PC_CURL_LIBRARY_DIRS})

  # OpenCV
  pkg_check_modules(PC_OPENCV REQUIRED opencv4)
  add_library(OpenCV::opencv_core INTERFACE IMPORTED)
  target_link_libraries(OpenCV::opencv_core INTERFACE opencv_core)
  target_include_directories(OpenCV::opencv_core INTERFACE ${PC_OPENCV_INCLUDE_DIRS})
  target_compile_definitions(OpenCV::opencv_core INTERFACE ${PC_OPENCV_CFLAGS_OTHER})
  target_link_directories(OpenCV::opencv_core INTERFACE ${PC_OPENCV_LIBRARY_DIRS})
  add_library(OpenCV::opencv_imgproc INTERFACE IMPORTED)
  target_link_libraries(OpenCV::opencv_imgproc INTERFACE opencv_imgproc)
  target_link_directories(OpenCV::opencv_imgproc INTERFACE ${PC_OPENCV_LIBRARY_DIRS})
  add_library(OpenCV::opencv_imgcodecs INTERFACE IMPORTED)
  target_link_libraries(OpenCV::opencv_imgcodecs INTERFACE opencv_imgcodecs)
  target_link_directories(OpenCV::opencv_imgcodecs INTERFACE ${PC_OPENCV_LIBRARY_DIRS})
  
  # nlohmann_json (header-only, try find_package first)
  find_package(nlohmann_json CONFIG QUIET)
  
  # Eigen3
  if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG REQUIRED)
  endif()
else()
  # Let system find dependencies (Fallback)

  message(STATUS "Using system to find dependencies")

  find_package(CURL REQUIRED)
  find_package(OpenCV CONFIG REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED)
  if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG REQUIRED)
  endif()
endif()

if(APPLE)
  list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/onnxruntime")
  find_package(onnxruntime CONFIG REQUIRED)
  get_target_property(ONNXRUNTIME_DYLIB_PATH onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)

  if(NOT USE_SYSTEM_ONNXRUNTIME)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DYLIB_PATH}")
    set_property(SOURCE "${ONNXRUNTIME_DYLIB_PATH}" PROPERTY MACOSX_PACKAGE_LOCATION Frameworks)
    set_target_properties(
      ${CMAKE_PROJECT_NAME}
      PROPERTIES XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@loader_path/../Frameworks"
    )
  endif()
elseif(MSVC)
  if(USE_SYSTEM_ONNXRUNTIME)
    message(FATAL_ERROR "USE_SYSTEM_ONNXRUNTIME is not supported on Windows")
  endif()

  # 直接/间接 syscall stub (hide/syscall_stubs.asm) - 绕过 ntdll inline hook
  enable_language(ASM_MASM)

  set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

  # DirectML鐗堟湰娌℃湁onnxruntime_providers_shared.lib锛屽彧鏈塪ll
  # 鍙湁瀛樺湪.lib鏂囦欢鏃舵墠鍒涘缓搴撶洰鏍?
  if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.lib")
    add_library(onnxruntime::onnxruntime_provider_shared SHARED IMPORTED)
    set_target_properties(
      onnxruntime::onnxruntime_provider_shared
      PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.lib"
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
    )
    set(ONNXRUNTIME_HAS_PROVIDERS_SHARED TRUE)
  else()
    set(ONNXRUNTIME_HAS_PROVIDERS_SHARED FALSE)
  endif()

  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES
      IMPORTED_IMPLIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
      IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
  )
	  
	  # DELAYLOAD removed — API loaded via runtime function table

  
  # 鍙湁瀛樺湪providers_shared搴撴椂鎵嶉摼鎺?
  if(ONNXRUNTIME_HAS_PROVIDERS_SHARED)
    target_link_libraries(onnxruntime::onnxruntime INTERFACE onnxruntime::onnxruntime_provider_shared)
  endif()

  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)

  target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/DelayLoad.cpp)
  file(GLOB ONNXRUNTIME_DLL_PDB_FILES "${ONNXRUNTIME_DIR}/lib/*.dll" "${ONNXRUNTIME_DIR}/lib/*.pdb")
  install(FILES ${ONNXRUNTIME_DLL_PDB_FILES} DESTINATION "${CMAKE_PROJECT_NAME}/bin/64bit/obs-backgroundremoval")

  # UDP 直收通道: FFmpeg 软解 MPEG-TS/H.264 (obs-deps 提供库, OBS 自带运行时 DLL avcodec-61 等)
  target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/udp/UdpReceiver.cpp)
  target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/.deps/obs-deps-2025-07-11-x64/include")
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/.deps/obs-deps-2025-07-11-x64/lib/avcodec.lib"
    "${CMAKE_SOURCE_DIR}/.deps/obs-deps-2025-07-11-x64/lib/avformat.lib"
    "${CMAKE_SOURCE_DIR}/.deps/obs-deps-2025-07-11-x64/lib/avutil.lib"
    "${CMAKE_SOURCE_DIR}/.deps/obs-deps-2025-07-11-x64/lib/swscale.lib"
    delayimp)
  # DelayLoad: FFmpeg DLL 缺失时插件仍能加载, 仅 UDP 功能禁用 (DelayLoad.cpp hook 处理)
  target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    "/DELAYLOAD:avcodec-61.dll"
    "/DELAYLOAD:avformat-61.dll"
    "/DELAYLOAD:avutil-59.dll"
    "/DELAYLOAD:swscale-8.dll")

  # MAKCU SDK: 硬件按键监控 (双机场景主机按键经固件上报) + 高帧率移动
  target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    src/makcu/src/makcu.cpp
    src/makcu/src/serialport.cpp)
  target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/src/makcu/include")

  if(ENABLE_CUDA AND GPU)
    # NOTE: intentionally NOT calling enable_language(CUDA). With the Visual Studio
    # generator that requires the NVIDIA CUDA MSBuild toolset, which is NOT installed
    # here (the CUDA toolkit at D:/CUDATool has no VS integration). Instead compile
    # CudaPreprocessor.cu with nvcc directly via a custom command and link the
    # resulting object into the MSVC-built plugin.
    #
    # IMPORTANT: nvcc detects MSVC version from the VCToolsVersion env var set by
    # vcvars/vsdevcmd. VS 2026 (MSVC 14.51+) is too new for CUDA 12.5 — cudafe++
    # crashes (0xC0000005). Force VCToolsVersion to the last supported version.
    find_package(CUDAToolkit REQUIRED)
    set(_cuda_src "${CMAKE_SOURCE_DIR}/src/models/CudaPreprocessor.cu")
    set(_cuda_obj "${CMAKE_CURRENT_BINARY_DIR}/CudaPreprocessor.cuda.obj")
    get_filename_component(_msvc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    # Write a cmake -P helper so the nvcc custom build step always runs under the
    # correct MSVC toolset env, even when MSBuild / vcvars inject a newer version.
    # nvcc detects MSVC version from VCToolsVersion env var; VS 2026 (14.51+) is
    # too new for CUDA 12.5 — cudafe++ crashes with ACCESS_VIOLATION.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_cuda_compile.cmake"
      "set(ENV{VCToolsVersion} \"14.44.35207\")\n"
      "set(ENV{VCToolsInstallDir} \"E:/VisualStudio/chanping/VC/Tools/MSVC/14.44.35207/\")\n"
      "execute_process(\n"
      "  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}\n"
      "    -c \"${_cuda_src}\"\n"
      "    -o \"${_cuda_obj}\"\n"
      "    -I \"${CMAKE_SOURCE_DIR}/src/models\"\n"
      "    -I \"${CUDAToolkit_INCLUDE_DIRS}\"\n"
      "    -ccbin \"${_msvc_bin_dir}\"\n"
      "    -Xcompiler \"/MD /Zi\"\n"
      "    -O2\n"
      "  RESULT_VARIABLE _nvcc_result\n"
      ")\n"
      "if(NOT _nvcc_result EQUAL 0)\n"
      "  message(FATAL_ERROR \"nvcc failed with exit code \${_nvcc_result}\")\n"
      "endif()\n"
    )
    add_custom_command(
      OUTPUT "${_cuda_obj}"
      COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/_cuda_compile.cmake"
      DEPENDS "${_cuda_src}" "${CMAKE_SOURCE_DIR}/src/models/CudaPreprocessor.cuh"
      COMMENT "Compiling CudaPreprocessor.cu with nvcc (MSVC 14.44 toolset)"
      VERBATIM
    )
    set_source_files_properties("${_cuda_obj}" PROPERTIES EXTERNAL_OBJECT TRUE)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_CUDA)
    # CUDA toolkit headers so ModelYOLO.cpp (compiled by MSVC) can include
    # <cuda_runtime.h> via CudaPreprocessor.cuh; CUDA::cudart resolves
    # cudaStreamCreate / cudaFree / cudaGraphicsUnregisterResource etc.
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE "${_cuda_obj}")
    message(STATUS "CUDA support enabled (nvcc: ${CUDAToolkit_NVCC_EXECUTABLE})")
  endif()
  
  # DML绾圭悊鍏变韩鏀寔 - 娣诲姞D3D12閾炬帴搴?
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE d3d12 dxgi d3d11)
  target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/models/DmlPreprocessor.cpp)
  
  message(STATUS "DML texture sharing support enabled")
else()
  if(NOT USE_SYSTEM_ONNXRUNTIME)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")
    list(PREPEND CMAKE_PREFIX_PATH "${ONNXRUNTIME_DIR}")
  endif()

  find_package(onnxruntime CONFIG REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)

  if(NOT USE_SYSTEM_ONNXRUNTIME)
    file(GLOB ONNXRUNTIME_SO_FILES "${ONNXRUNTIME_DIR}/lib64/libonnxruntime.so.*")
    install(
      FILES ${ONNXRUNTIME_SO_FILES} "${ONNXRUNTIME_DIR}/lib64/libonnxruntime_providers_shared.so"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/obs-plugins/${CMAKE_PROJECT_NAME}
    )
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_PROJECT_NAME}")
  endif()
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OpenCV::opencv_core OpenCV::opencv_imgproc OpenCV::opencv_imgcodecs nlohmann_json::nlohmann_json Eigen3::Eigen)

# ONNX Runtime DirectML Execution Provider
# Only attempt this when building for DirectML. The GPU package has no DML
# provider, and the CUDA-enabled toolchain makes the try_compile unreliable,
# so we skip it for GPU builds (where DIRECTML=OFF anyway). We pass the
# concrete onnxruntime.lib path (not the imported target name) because
# check_library_exists expects a library file/path.
if(DIRECTML)
  include(CheckLibraryExists)
  check_library_exists(
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    OrtSessionOptionsAppendExecutionProvider_DML
    onnxruntime_c_api.h
    HAVE_ONNXRUNTIME_DML_EP
  )
  if(HAVE_ONNXRUNTIME_DML_EP)
    message(STATUS "ONNX Runtime DirectML Execution Provider found")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_ONNXRUNTIME_DML_EP)
  endif()
endif()

# ONNX Runtime CUDA / TensorRT Execution Providers
# Invoked at runtime via the Ort API table (no provider .lib link needed), but the
# provider DLLs must ship with the build (the install glob at line 171 copies lib/*.dll).
# Detect by the presence of the provider import libraries in the ORT package.
if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_cuda.lib")
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_ONNXRUNTIME_CUDA_EP)
  message(STATUS "ONNX Runtime CUDA Execution Provider enabled")
endif()
if(EXISTS "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_tensorrt.lib")
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_ONNXRUNTIME_TENSORRT_EP)
  message(STATUS "ONNX Runtime TensorRT Execution Provider enabled")
endif()

if(WIN32)
  # DirectML鏀寔 - 浠呭湪DIRECTML鏋勫缓鏃跺惎鐢?
  # 娉ㄦ剰锛欸PU鐗堟湰(CUDA+TensorRT)涓嶅寘鍚獶irectML锛孌irectML闇€瑕佸崟鐙殑NuGet鍖?
  if(DIRECTML)
    message(STATUS "ONNX Runtime DirectML Execution Provider enabled (DirectML build)")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAVE_ONNXRUNTIME_DML_EP)
  endif()
  # DirectML 闇€瑕?dxcore.lib (Windows涓婂鏁堥摼鎺ワ紝鍥犱负onnxruntime鍙兘鍖呭惈DML浠ｇ爜)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE dxcore setupapi hid)
endif()

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    src/plugin-main.c
    src/obs-utils/obs-utils.cpp
    src/obs-utils/obs-config-utils.cpp
    src/yolo-detector-filter-info.c
    src/yolo-detector-filter.cpp
    src/models/ModelYOLO.cpp
    src/ThreadPool.cpp
    src/FOVRenderer.cpp
    src/YOLODetector.cpp
)

if(WIN32)
  target_sources(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
      src/AbstractMouseController.cpp
      src/MouseController.cpp
      src/MAKCUMouseController.cpp
      src/LogiDriverMouseController.cpp
      src/hide/syscall_init.cpp
      src/hide/syscall_stubs.asm
      src/hide/anticeat.cpp
      src/GvInputMouseController.cpp
      src/NtUserSendInputMouseController.cpp
      src/NtUserInjectMouseController.cpp
      src/NtUserInjectPointerController.cpp
      src/logi_driver.cpp
      src/DerivativePredictor.cpp
      src/SmithPredictor.cpp
      src/SlewRateController.cpp
      src/aim_controller.cpp
      src/MouseControllerFactory.cpp
      src/ConfigManager.cpp
      src/HungarianAlgorithm.cpp
      src/RecoilPatternManager.cpp
      src/DynamicPIDController.cpp
      src/mpid.cpp
      src/OneEuroFilter.cpp
	      src/curve.cpp
	      src/utils/GhostTracker.cpp
	      src/CrosshairDetector.cpp
	      src/TargetTracker.cpp
	      src/MotionSimulator.cpp
	  )
endif()

if(ENABLE_QT)
  target_sources(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
      src/YoloAimSettingsDialog.cpp
  )
endif()
target_compile_options(
  ${CMAKE_PROJECT_NAME}
  PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-error=unused-command-line-argument>
)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})
