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
    # NuGet包解压后直接在当前目录创建build/和runtimes/目录
    
    # 下载ONNX Runtime DirectML包
    file(
      DOWNLOAD
        https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/1.23.2
        onnxruntime-directml.nupkg
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf onnxruntime-directml.nupkg
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # 创建目标目录结构
    file(MAKE_DIRECTORY onnxruntime/include onnxruntime/lib)
    
    # NuGet包直接解压到当前目录，复制文件
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/build/native/include/ DESTINATION onnxruntime/include)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/win-x64/native/ DESTINATION onnxruntime/lib)
    
    # 下载DirectML平台包 (DirectML.dll)
    file(
      DOWNLOAD
        https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
        directml.nupkg
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf directml.nupkg
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    # 复制DirectML.dll到lib目录
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/runtimes/win-x64/native/DirectML.dll DESTINATION onnxruntime/lib)
    
    # 清理所有NuGet包的临时文件
    file(REMOVE_RECURSE ${CMAKE_CURRENT_SOURCE_DIR}/build ${CMAKE_CURRENT_SOURCE_DIR}/runtimes ${CMAKE_CURRENT_SOURCE_DIR}/package ${CMAKE_CURRENT_SOURCE_DIR}/_rels)
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
  else()
    # CPU版本
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

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
