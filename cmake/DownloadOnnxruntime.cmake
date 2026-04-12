cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PLATFORM)
  if(APPLE)
    set(PLATFORM "macos")
  elseif(WIN32)
    set(PLATFORM "windows")
  else()
    set(PLATFORM "linux")
  endif()
endif()

if(NOT DEFINED GPU)
  set(GPU OFF)
endif()

if(NOT DEFINED DIRECTML)
  set(DIRECTML OFF)
endif()

file(REMOVE_RECURSE onnxruntime)

if(PLATFORM STREQUAL "macos")
  file(
    DOWNLOAD
      https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-universal2-1.23.2.tgz
      onnxruntime-osx-universal2-1.23.2.tgz
    EXPECTED_HASH SHA256=49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-osx-universal2-1.23.2.tgz
  )
  file(RENAME onnxruntime-osx-universal2-1.23.2 onnxruntime)
  message(STATUS "Downloaded ONNX Runtime for macOS")
elseif(PLATFORM STREQUAL "windows")
  # Windows平台：使用NuGet预编译包
  # 三个独立版本：GPU(CUDA/TensorRT)、DirectML、CPU
  
  if(DIRECTML)
    # DirectML版本 - 使用本地已解压的NuGet包
    # 包含: DirectML执行提供程序 + CPU执行提供程序
    message(STATUS "Setting up ONNX Runtime DirectML version...")
    
    # 设置工作目录
    set(WORK_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
    
    # 创建目标目录
    file(MAKE_DIRECTORY ${WORK_DIR}/onnxruntime/include ${WORK_DIR}/onnxruntime/lib)
    
    # 查找本地已解压的ONNX Runtime DirectML包目录
    set(ORT_LOCAL_DIR "")
    file(GLOB ORT_LOCAL_DIRS "${WORK_DIR}/microsoft.ml.onnxruntime.directml.*")
    if(ORT_LOCAL_DIRS)
      list(GET ORT_LOCAL_DIRS 0 ORT_LOCAL_DIR)
      message(STATUS "Found local ONNX Runtime DirectML package: ${ORT_LOCAL_DIR}")
    else()
      message(FATAL_ERROR "Cannot find local microsoft.ml.onnxruntime.directml.* directory. Please ensure the NuGet package is extracted.")
    endif()
    
    # 复制ONNX Runtime文件
    if(EXISTS "${ORT_LOCAL_DIR}/runtimes/win-x64/native")
      file(COPY "${ORT_LOCAL_DIR}/runtimes/win-x64/native/" DESTINATION ${WORK_DIR}/onnxruntime/lib)
      message(STATUS "Copied onnxruntime.dll and onnxruntime_providers_shared.dll")
    else()
      message(FATAL_ERROR "Cannot find runtimes/win-x64/native in ${ORT_LOCAL_DIR}")
    endif()
    
    # 复制头文件
    if(EXISTS "${ORT_LOCAL_DIR}/build/native/include")
      file(COPY "${ORT_LOCAL_DIR}/build/native/include/" DESTINATION ${WORK_DIR}/onnxruntime/include)
      message(STATUS "Copied ONNX Runtime headers")
    else()
      message(FATAL_ERROR "Cannot find build/native/include in ${ORT_LOCAL_DIR}")
    endif()
    
    # 查找本地已解压的DirectML包目录
    set(DML_LOCAL_DIR "")
    file(GLOB DML_LOCAL_DIRS "${WORK_DIR}/microsoft.ai.directml.*")
    if(DML_LOCAL_DIRS)
      list(GET DML_LOCAL_DIRS 0 DML_LOCAL_DIR)
      message(STATUS "Found local DirectML package: ${DML_LOCAL_DIR}")
    else()
      message(FATAL_ERROR "Cannot find local microsoft.ai.directml.* directory. Please ensure the NuGet package is extracted.")
    endif()
    
    # 复制DirectML.dll
    if(EXISTS "${DML_LOCAL_DIR}/bin/x64-win/DirectML.dll")
      file(COPY "${DML_LOCAL_DIR}/bin/x64-win/DirectML.dll" DESTINATION ${WORK_DIR}/onnxruntime/lib)
      message(STATUS "Copied DirectML.dll")
    else()
      message(FATAL_ERROR "Cannot find bin/x64-win/DirectML.dll in ${DML_LOCAL_DIR}")
    endif()
    
    # 创建CMake配置文件
    file(MAKE_DIRECTORY ${WORK_DIR}/onnxruntime/lib/cmake/onnxruntime)
    file(WRITE ${WORK_DIR}/onnxruntime/lib/cmake/onnxruntime/onnxruntimeConfig.cmake
"include(CMakeFindDependencyMacro)
find_dependency(Threads)
if(NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES \"\${CMAKE_CURRENT_LIST_DIR}/../../../include\"
    IMPORTED_LOCATION \"\${CMAKE_CURRENT_LIST_DIR}/../onnxruntime.dll\"
    IMPORTED_IMPLIB \"\${CMAKE_CURRENT_LIST_DIR}/../onnxruntime.lib\"
  )
endif()
")
    
    message(STATUS "Downloaded ONNX Runtime DirectML version")
    message(STATUS "  - onnxruntime.dll (with DirectML EP)")
    message(STATUS "  - onnxruntime_providers_shared.dll")
    message(STATUS "  - DirectML.dll (platform code)")
    
  elseif(GPU)
    # GPU版本 - 从GitHub Release下载（CUDA + TensorRT）
    message(STATUS "Downloading ONNX Runtime GPU version from GitHub Release...")
    
    file(
      DOWNLOAD
        https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-gpu-1.23.2.zip
        onnxruntime-win-x64-gpu-1.23.2.zip
      EXPECTED_HASH SHA256=e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-win-x64-gpu-1.23.2.zip
    )
    file(RENAME onnxruntime-win-x64-gpu-1.23.2 onnxruntime)
    message(STATUS "Downloaded ONNX Runtime GPU version (CUDA + TensorRT)")
    
  else()
    # CPU版本
    message(STATUS "Downloading ONNX Runtime CPU version...")
    
    file(
      DOWNLOAD
        https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip
        onnxruntime-win-x64-1.23.2.zip
      EXPECTED_HASH SHA256=0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-win-x64-1.23.2.zip
    )
    file(RENAME onnxruntime-win-x64-1.23.2 onnxruntime)
    message(STATUS "Downloaded ONNX Runtime CPU version")
  endif()
  
elseif(PLATFORM STREQUAL "linux")
  file(
    DOWNLOAD
      https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
      onnxruntime-linux-x64-1.23.2.tgz
    EXPECTED_HASH SHA256=1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-linux-x64-1.23.2.tgz
  )
  file(RENAME onnxruntime-linux-x64-1.23.2 onnxruntime)
  execute_process(COMMAND ln -s lib onnxruntime/lib64)
  message(STATUS "Downloaded ONNX Runtime for Linux")
endif()

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
