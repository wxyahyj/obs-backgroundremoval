#pragma once

// 日志工具(M6)— 文件重定向 + 大小轮转。
// obs_log(主仓 stub)与 standalone 全部走此通道。

#include <cstdarg>
#include <string>

namespace ya {
namespace util {

// 初始化日志系统:logs/host.log;超过 max_bytes 轮转 .1/.2/.3。
// 重复调用仅刷新路径。
void log_init(const std::string& log_dir, size_t max_bytes = 5 * 1024 * 1024);

// 写一行(带时间戳 + 级别);文件 + stderr 双写。
void log_write(const char* level, const char* fmt, va_list args);

// 便捷版(非 variadic 调用方)
void log_printf(const char* level, const char* fmt, ...);

void log_flush();

} // namespace util
} // namespace ya
