// 崩溃处理实现 — MiniDumpWriteDump。

#include "CrashGuard.hpp"

#include "Log.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace ya {
namespace util {

namespace {

std::string g_dump_dir;

LONG WINAPI crash_filter(EXCEPTION_POINTERS* ep)
{
    log_printf("FATAL", "unhandled exception code=0x%08X at 0x%p",
               ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionCode : 0,
               ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionAddress : nullptr);
    log_flush();

#ifdef _WIN32
    if (!g_dump_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(g_dump_dir, ec);
        const std::string path = g_dump_dir + "/crash_" +
                                 std::to_string(static_cast<long long>(
                                     std::chrono::system_clock::now().time_since_epoch()
                                         .count())) +
                                 ".dmp";
        HANDLE file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION mei{};
            mei.ThreadId = GetCurrentThreadId();
            mei.ExceptionPointers = ep;
            mei.ClientPointers = TRUE;
            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                              static_cast<MINIDUMP_TYPE>(MiniDumpWithDataSegs |
                                                         MiniDumpWithHandleData),
                              ep ? &mei : nullptr, nullptr, nullptr);
            CloseHandle(file);
        }
        log_printf("FATAL", "minidump written: %s", path.c_str());
        log_flush();
    }
#endif
    return EXCEPTION_EXECUTE_HANDLER;
}

} // namespace

void install_crash_handler(const std::string& dump_dir)
{
    g_dump_dir = dump_dir;
#ifdef _WIN32
    SetUnhandledExceptionFilter(crash_filter);
    // 不弹系统"程序已停止工作"框,由本进程自行处理
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
}

} // namespace util
} // namespace ya
