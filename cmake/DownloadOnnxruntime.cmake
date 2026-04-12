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
    # DirectML版本 - 使用 Microsoft.ML.OnnxRuntime.DirectML NuGet包
    # 包含: DirectML执行提供程序 + CPU执行提供程序
    # 注意: DirectML.dll需要单独提供
    message(STATUS "Downloading ONNX Runtime DirectML version from NuGet...")
    
    file(
      DOWNLOAD
        https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.2
        onnxruntime-directml.nupkg
      SHOW_PROGRESS
    )
    
    # 解压NuGet包（NuGet包实际上是ZIP格式）
    message(STATUS "Extracting NuGet package...")
    
    # 检查nupkg文件是否存在
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime-directml.nupkg")
      message(FATAL_ERROR "NuGet package not found at ${CMAKE_CURRENT_LIST_DIR}/../onnxruntime-directml.nupkg")
    endif()
    
    # 获取文件大小
    file(SIZE "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime-directml.nupkg" NUPKG_SIZE)
    message(STATUS "NuGet package size: ${NUPKG_SIZE} bytes")
    
    # 在Windows上使用PowerShell解压（更可靠）
    if(WIN32)
      message(STATUS "Using PowerShell to extract...")
      execute_process(
        COMMAND powershell -Command "Expand-Archive -Path '${CMAKE_CURRENT_LIST_DIR}/../onnxruntime-directml.nupkg' -DestinationPath '${CMAKE_CURRENT_LIST_DIR}/..' -Force"
        RESULT_VARIABLE EXTRACT_RESULT
        ERROR_VARIABLE EXTRACT_ERROR
      )
      if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "PowerShell extraction failed: ${EXTRACT_ERROR}")
      endif()
    else()
      # 其他平台使用CMake的tar命令
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-directml.nupkg
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
        RESULT_VARIABLE EXTRACT_RESULT
        ERROR_VARIABLE EXTRACT_ERROR
      )
      if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "tar extraction failed: ${EXTRACT_ERROR}")
      endif()
    endif()
    
    message(STATUS "Extraction completed successfully")
    
    # 设置工作目录
    set(WORK_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
    
    # 创建目标目录
    file(MAKE_DIRECTORY ${WORK_DIR}/onnxruntime/include ${WORK_DIR}/onnxruntime/lib)
    
    # 调试：列出工作目录内容
    message(STATUS "Checking directory structure after extraction...")
    message(STATUS "Work directory: ${WORK_DIR}")
    file(GLOB ALL_ITEMS RELATIVE ${WORK_DIR} "${WORK_DIR}/*")
    foreach(ITEM ${ALL_ITEMS})
      message(STATUS "  Found: ${ITEM}")
    endforeach()
    
    # NuGet包可能解压到当前目录或以包名命名的子目录
    set(ORT_RUNTIME_DIR "")
    set(ORT_INCLUDE_DIR "")

    if(EXISTS "${WORK_DIR}/runtimes/win-x64/native")
      set(ORT_RUNTIME_DIR "${WORK_DIR}/runtimes/win-x64/native")
      set(ORT_INCLUDE_DIR "${WORK_DIR}/build/native/include")
      message(STATUS "Found ONNX Runtime in current directory")
    else()
      file(GLOB_RECURSE ORT_RUNTIME_DIRS RELATIVE ${WORK_DIR}
        "${WORK_DIR}/runtimes/win-x64/native"
        "${WORK_DIR}/*/runtimes/win-x64/native"
        "${WORK_DIR}/*/*/runtimes/win-x64/native"
        "${WORK_DIR}/*/*/*/runtimes/win-x64/native"
      )
      if(ORT_RUNTIME_DIRS)
        list(GET ORT_RUNTIME_DIRS 0 ORT_RUNTIME_REL)
        set(ORT_RUNTIME_DIR "${WORK_DIR}/${ORT_RUNTIME_REL}")
        message(STATUS "Found nested ONNX Runtime runtime directory: ${ORT_RUNTIME_DIR}")
      else()
        file(GLOB ORT_PKG_DIRS "${WORK_DIR}/microsoft.ml.onnxruntime.directml.*")
        message(STATUS "Glob result for microsoft.ml.onnxruntime.directml.*: ${ORT_PKG_DIRS}")
        if(ORT_PKG_DIRS)
          list(GET ORT_PKG_DIRS 0 ORT_PKG_DIR)
          message(STATUS "Checking directory: ${ORT_PKG_DIR}")
          if(IS_DIRECTORY ${ORT_PKG_DIR})
            message(STATUS "Is directory: yes")
            if(EXISTS "${ORT_PKG_DIR}/runtimes/win-x64/native")
              set(ORT_RUNTIME_DIR "${ORT_PKG_DIR}/runtimes/win-x64/native")
            else()
              message(STATUS "runtimes/win-x64/native not found in ${ORT_PKG_DIR}")
              file(GLOB SUB_ITEMS "${ORT_PKG_DIR}/*")
              foreach(SUB ${SUB_ITEMS})
                message(STATUS "  Sub: ${SUB}")
              endforeach()
            endif()
          else()
            message(STATUS "Is directory: no, it's a file")
          endif()
        else()
          message(STATUS "No matching directories found for microsoft.ml.onnxruntime.directml.*")
        endif()
      endif()

      if(ORT_RUNTIME_DIR)
        file(GLOB_RECURSE ORT_INCLUDE_DIRS RELATIVE ${WORK_DIR}
          "${WORK_DIR}/build/native/include"
          "${WORK_DIR}/*/build/native/include"
          "${WORK_DIR}/*/*/build/native/include"
          "${WORK_DIR}/*/*/*/build/native/include"
        )
        if(ORT_INCLUDE_DIRS)
          list(GET ORT_INCLUDE_DIRS 0 ORT_INCLUDE_REL)
          set(ORT_INCLUDE_DIR "${WORK_DIR}/${ORT_INCLUDE_REL}")
          message(STATUS "Found nested ONNX Runtime include directory: ${ORT_INCLUDE_DIR}")
        else()
          get_filename_component(ORT_PKG_ROOT "${ORT_RUNTIME_DIR}" DIRECTORY)
          get_filename_component(ORT_PKG_ROOT "${ORT_PKG_ROOT}" DIRECTORY)
          get_filename_component(ORT_PKG_ROOT "${ORT_PKG_ROOT}" DIRECTORY)
          set(ORT_INCLUDE_DIR "${ORT_PKG_ROOT}/build/native/include")
          if(EXISTS "${ORT_INCLUDE_DIR}")
            message(STATUS "Found include directory by package root fallback: ${ORT_INCLUDE_DIR}")
          else()
            set(ORT_INCLUDE_DIR "")
          endif()
        endif()
      endif()
    endif()

    if(NOT ORT_RUNTIME_DIR OR NOT EXISTS "${ORT_RUNTIME_DIR}")
      message(FATAL_ERROR "Cannot find ONNX Runtime native libraries after NuGet extraction. ORT_RUNTIME_DIR=${ORT_RUNTIME_DIR}")
    endif()
    
    file(COPY ${ORT_RUNTIME_DIR}/ DESTINATION ${WORK_DIR}/onnxruntime/lib)
    file(COPY ${ORT_INCLUDE_DIR}/ DESTINATION ${WORK_DIR}/onnxruntime/include)
    
    # 清理临时文件
    if(DEFINED ORT_PKG_DIR AND ORT_PKG_DIR)
      file(REMOVE_RECURSE ${ORT_PKG_DIR})
    endif()
    file(REMOVE_RECURSE ${WORK_DIR}/runtimes ${WORK_DIR}/build ${WORK_DIR}/package ${WORK_DIR}/_rels)
    file(REMOVE ${WORK_DIR}/onnxruntime-directml.nupkg ${WORK_DIR}/[Content_Types].xml)
    file(REMOVE ${WORK_DIR}/Microsoft.ML.OnnxRuntime.DirectML.nuspec)
    file(REMOVE ${WORK_DIR}/LICENSE ${WORK_DIR}/README.md ${WORK_DIR}/ThirdPartyNotices.txt ${WORK_DIR}/Privacy.md ${WORK_DIR}/.signature.p7s ${WORK_DIR}/ORT_icon_for_light_bg.png)
    
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
