// 日志实现 — 文件 + 轮转。

#include "Log.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ya {
namespace util {

namespace {

std::mutex g_mu;
std::string g_path;
size_t g_max_bytes = 5 * 1024 * 1024;

// 轮转:host.log → host.log.1 → host.log.2 → host.log.3(丢弃最旧)
void rotate_locked()
{
    namespace fs = std::filesystem;
    fs::remove(g_path + ".3");
    if (fs::exists(g_path + ".2"))
        fs::rename(g_path + ".2", g_path + ".3");
    if (fs::exists(g_path + ".1"))
        fs::rename(g_path + ".1", g_path + ".2");
    fs::rename(g_path, g_path + ".1");
}

std::string timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

} // namespace

void log_init(const std::string& log_dir, size_t max_bytes)
{
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_max_bytes = max_bytes;
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(log_dir, ec);
        g_path = (fs::path(log_dir) / "host.log").string();
        // 上次残留过大 → 启动即轮转一次
        if (fs::exists(g_path, ec) && fs::file_size(g_path, ec) > g_max_bytes) {
            try {
                rotate_locked();
            } catch (...) {
            }
        }
    }
    // 锁外写启动横幅(log_write 自持锁)
    log_write("INFO", "=== YoloAim started ===", "");
}

void log_write(const char* level, const char* fmt, va_list args)
{
    std::lock_guard<std::mutex> lock(g_mu);

    char msg[4096];
    std::vsnprintf(msg, sizeof(msg), fmt, args);

    const std::string line = "[" + timestamp() + "] [" + level + "] " + msg + "\n";

    // stderr 双写
    std::fputs(line.c_str(), stderr);

    if (g_path.empty())
        return;
    // 轮转检查
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(g_path, ec) && fs::file_size(g_path, ec) > g_max_bytes) {
        try {
            rotate_locked();
        } catch (...) {
        }
    }
    std::ofstream f(g_path, std::ios::app | std::ios::binary);
    if (f.is_open()) {
        f.write(line.data(), static_cast<std::streamsize>(line.size()));
        f.flush();
    }
}

void log_printf(const char* level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_write(level, fmt, args);
    va_end(args);
}

void log_flush()
{
    std::lock_guard<std::mutex> lock(g_mu);
    std::fflush(stderr);
}

} // namespace util
} // namespace ya
