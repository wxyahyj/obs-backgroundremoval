cmake_minimum_required(VERSION 3.16)

# TensorRT 下载/配置脚本
# 用于 GitHub Actions CI 构建

if(NOT DEFINED TENSORRT_VERSION)
  set(TENSORRT_VERSION "10.9.0.36")
endif()

if(NOT DEFINED TENSORRT_PATH)
  set(TENSORRT_PATH "${CMAKE_CURRENT_LIST_DIR}/../TensorRT")
endif()

# 检查TensorRT是否已存在
if(EXISTS "${TENSORRT_PATH}/include/NvInfer.h")
  message(STATUS "TensorRT already exists at: ${TENSORRT_PATH}")
  return()
endif()

# Windows平台下载TensorRT
if(WIN32)
  message(STATUS "TensorRT not found, downloading...")
  
  # TensorRT Windows ZIP包URL (NVIDIA开发者账号需要)
  # 使用NVIDIA NGC公共URL或镜像
  set(TRT_ZIP_NAME "TensorRT-${TENSORRT_VERSION}.Windows10.x86_64.cuda-12.8.zip")
  
  # 尝试从环境变量获取下载链接
  if(DEFINED ENV{TENSORRT_DOWNLOAD_URL})
    set(TRT_DOWNLOAD_URL "$ENV{TENSORRT_DOWNLOAD_URL}")
    message(STATUS "Using TensorRT download URL from environment")
  else()
    # 默认URL (需要NVIDIA开发者账号)
    # 这里使用占位符，实际使用时需要设置环境变量
    set(TRT_DOWNLOAD_URL "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/10.9.0/zip/TensorRT-10.9.0.36.Windows10.x86_64.cuda-12.8.zip")
    message(WARNING "TensorRT download requires NVIDIA Developer account.")
    message(STATUS "Please set TENSORRT_DOWNLOAD_URL environment variable or manually download TensorRT")
  endif()
  
  # 创建临时目录
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../temp")
  
  # 下载TensorRT
  file(
    DOWNLOAD
      ${TRT_DOWNLOAD_URL}
      "${CMAKE_CURRENT_LIST_DIR}/../temp/${TRT_ZIP_NAME}"
    STATUS DOWNLOAD_STATUS
    SHOW_PROGRESS
  )
  
  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 STATUS_STRING)
  
  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download TensorRT: ${STATUS_STRING}")
  endif()
  
  # 解压
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_CURRENT_LIST_DIR}/../temp/${TRT_ZIP_NAME}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  )
  
  # 重命名到标准路径
  file(GLOB TRT_EXTRACTED_DIR "${CMAKE_CURRENT_LIST_DIR}/../TensorRT-*")
  if(TRT_EXTRACTED_DIR)
    file(RENAME ${TRT_EXTRACTED_DIR} ${TENSORRT_PATH})
  endif()
  
  # 清理临时文件
  file(REMOVE_RECURSE "${CMAKE_CURRENT_LIST_DIR}/../temp")
  
  message(STATUS "TensorRT downloaded and extracted to: ${TENSORRT_PATH}")
  
else()
  message(FATAL_ERROR "TensorRT download only supported on Windows in this script")
endif()
