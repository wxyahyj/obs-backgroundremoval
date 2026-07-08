#ifndef CONSTS_H
#define CONSTS_H

// constexpr确保编译期常量，避免多重定义链接错误
constexpr const char *USEGPU_CPU = "cpu";
constexpr const char *USEGPU_CUDA = "cuda";
constexpr const char *USEGPU_ROCM = "rocm";
constexpr const char *USEGPU_MIGRAPHX = "migraphx";
constexpr const char *USEGPU_TENSORRT = "tensorrt";
constexpr const char *USEGPU_COREML = "coreml";
constexpr const char *USEGPU_DML = "dml";

#endif /* CONSTS_H */
