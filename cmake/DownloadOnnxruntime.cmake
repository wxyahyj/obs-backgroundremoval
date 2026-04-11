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
elseif(PLATFORM STREQUAL "windows")
  if(GPU)
    # GPU版本包含CUDA和TensorRT
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
  elseif(DIRECTML)
    # DirectML版本 - 从NuGet下载
    # NuGet包解压后可能创建以包名命名的根目录，也可能不创建
    
    # 下载ONNX Runtime DirectML包
    file(
      DOWNLOAD
        https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.2
        onnxruntime-directml.nupkg
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-directml.nupkg
    )
    
    # 创建目标目录结构
    file(MAKE_DIRECTORY onnxruntime/include onnxruntime/lib)
    
    # 查找解压后的目录结构
    # NuGet包可能直接解压到当前目录，也可能创建以包名命名的子目录
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/native/include")
      # 直接解压到当前目录
      file(COPY build/native/include/ DESTINATION onnxruntime/include)
      file(COPY runtimes/win-x64/native/ DESTINATION onnxruntime/lib)
    else()
      # 解压到以包名命名的子目录，使用GLOB查找
      file(GLOB ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/microsoft.ml.onnxruntime.directml.*")
      if(ORT_DIR)
        list(GET ORT_DIR 0 ORT_DIR)
        file(COPY "${ORT_DIR}/build/native/include/" DESTINATION onnxruntime/include)
        file(COPY "${ORT_DIR}/runtimes/win-x64/native/" DESTINATION onnxruntime/lib)
      else()
        message(FATAL_ERROR "Cannot find ONNX Runtime DirectML package directory")
      endif()
    endif()
    
    # 下载DirectML平台包 (DirectML.dll)
    # 注意：Microsoft.ML.OnnxRuntime.DirectML包不包含DirectML.dll
    file(
      DOWNLOAD
        https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
        directml.nupkg
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf directml.nupkg
    )
    
    # 查找DirectML.dll的位置
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/runtimes/win-x64/native/DirectML.dll")
      file(COPY runtimes/win-x64/native/DirectML.dll DESTINATION onnxruntime/lib)
    else()
      file(GLOB DML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/microsoft.ai.directml.*")
      if(DML_DIR)
        list(GET DML_DIR 0 DML_DIR)
        file(COPY "${DML_DIR}/runtimes/win-x64/native/DirectML.dll" DESTINATION onnxruntime/lib)
      else()
        message(FATAL_ERROR "Cannot find DirectML.dll")
      endif()
    endif()
    
    # 清理所有NuGet包的临时文件
    file(GLOB ORT_CLEANUP "${CMAKE_CURRENT_SOURCE_DIR}/microsoft.ml.onnxruntime.directml.*")
    file(GLOB DML_CLEANUP "${CMAKE_CURRENT_SOURCE_DIR}/microsoft.ai.directml.*")
    if(ORT_CLEANUP)
      file(REMOVE_RECURSE ${ORT_CLEANUP})
    endif()
    if(DML_CLEANUP)
      file(REMOVE_RECURSE ${DML_CLEANUP})
    endif()
    file(REMOVE onnxruntime-directml.nupkg directml.nupkg [Content_Types].xml Microsoft.AI.DirectML.nuspec)
    
    message(STATUS "Downloaded ONNX Runtime DirectML version from NuGet")
    message(STATUS "  - onnxruntime.dll (with DirectML EP)")
    message(STATUS "  - DirectML.dll (platform code)")
    
    # 创建CMake配置文件（NuGet包没有提供）
    file(MAKE_DIRECTORY onnxruntime/lib/cmake/onnxruntime)
    file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeConfig.cmake
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
else()
  message(FATAL_ERROR "Unsupported platform: ${PLATFORM}")
endif()

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
