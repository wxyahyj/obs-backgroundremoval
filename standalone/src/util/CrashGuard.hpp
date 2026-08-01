#pragma once

// 崩溃处理(M6)— 未处理异常 → minidump + 日志记录。

#include <string>

namespace ya {
namespace util {

// 安装 SEH 未处理异常过滤器;崩溃时写 minidump 到 dump_dir 并记录。
void install_crash_handler(const std::string& dump_dir);

} // namespace util
} // namespace ya
