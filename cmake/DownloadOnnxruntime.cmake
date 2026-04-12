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
    
    # 如果开启DIRECTML，就额外下载并复制DirectML.dll
    if(DIRECTML)
      message(STATUS "Downloaded ONNX Runtime GPU version (CUDA + TensorRT) with DirectML support")
      message(STATUS "Adding DirectML support to GPU package...")
      
      # 下载DirectML平台包 (DirectML.dll)
      file(
        DOWNLOAD
          https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4
          directml.nupkg
      )
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf directml.nupkg
      )
      
      # 查找DirectML.dll的位置
      # NuGet包解压后的实际结构: bin/x64-win/DirectML.dll
      set(DML_DLL_PATH "")
      
      # 路径1: bin/x64-win/ (NuGet包实际结构)
      if(EXISTS "bin/x64-win/DirectML.dll")
        set(DML_DLL_PATH "bin/x64-win/DirectML.dll")
        message(STATUS "Found DirectML.dll in bin/x64-win/")
      # 路径2: runtimes/win-x64/native/ (某些NuGet包结构)
      elseif(EXISTS "runtimes/win-x64/native/DirectML.dll")
        set(DML_DLL_PATH "runtimes/win-x64/native/DirectML.dll")
        message(STATUS "Found DirectML.dll in runtimes/win-x64/native/")
      # 路径3: package目录下
      elseif(EXISTS "package/runtimes/win-x64/native/DirectML.dll")
        set(DML_DLL_PATH "package/runtimes/win-x64/native/DirectML.dll")
        message(STATUS "Found DirectML.dll in package/runtimes/win-x64/native/")
      else()
        # 尝试查找可能的子目录
        file(GLOB DML_PKG_DIRS LIST_DIRECTORIES true "microsoft.ai.directml.*")
        if(DML_PKG_DIRS)
          list(GET DML_PKG_DIRS 0 DML_PKG_DIR)
          if(EXISTS "${DML_PKG_DIR}/runtimes/win-x64/native/DirectML.dll")
            set(DML_DLL_PATH "${DML_PKG_DIR}/runtimes/win-x64/native/DirectML.dll")
            message(STATUS "Found DirectML.dll in ${DML_PKG_DIR}/runtimes/win-x64/native/")
          endif()
        endif()
      endif()
      
      if(DML_DLL_PATH)
        file(COPY ${DML_DLL_PATH} DESTINATION onnxruntime/lib)
      else()
        # 调试信息：列出当前目录结构
        message(STATUS "Current directory structure:")
        file(GLOB_RECURSE ALL_FILES "*")
        foreach(F ${ALL_FILES})
          message(STATUS "  ${F}")
        endforeach()
        message(FATAL_ERROR "Cannot find DirectML.dll after extraction. See directory structure above.")
      endif()
      
      # 清理DirectML临时文件
      if(DEFINED DML_PKG_DIR AND DML_PKG_DIR)
        file(REMOVE_RECURSE ${DML_PKG_DIR})
      endif()
      file(REMOVE_RECURSE runtimes package _rels bin build include)
      file(REMOVE directml.nupkg [Content_Types].xml Microsoft.AI.DirectML.nuspec LICENSE.txt LICENSE-CODE.txt README.md ThirdPartyNotices.txt .signature.p7s)
      
      message(STATUS "已打包 DirectML.dll，支持运行时切换到 DirectML")
    else()
      message(STATUS "Downloaded ONNX Runtime GPU version (CUDA + TensorRT)")
    endif()
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
endif()

if(EXISTS onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake)
  file(READ onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake FILE_CONTENT)

  set(OLD_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include/onnxruntime\"")
  set(NEW_STRING "INTERFACE_INCLUDE_DIRECTORIES \"\${_IMPORT_PREFIX}/include\"")

  string(REPLACE "${OLD_STRING}" "${NEW_STRING}" MODIFIED_CONTENT "${FILE_CONTENT}")
  file(WRITE onnxruntime/lib/cmake/onnxruntime/onnxruntimeTargets.cmake "${MODIFIED_CONTENT}")
endif()
