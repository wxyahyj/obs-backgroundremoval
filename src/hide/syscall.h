/**
 * syscall.h - 直接/间接 syscall 封装（绕过 ntdll inline hook）
 *
 * 方案见 docs/worklog/罗技驱动调用与隐藏绕过反作弊hook.md：
 *   - 直接 syscall: 不进入 ntdll 函数体, 自己执行 syscall
 *   - 间接 syscall (P0 推荐): jmp 到 ntdll 内完好的 syscall;ret,
 *     使 syscall 指令的 RIP 落在 ntdll .text 范围, 规避反作弊的 RIP 范围检查
 *
 * SSN 随 Windows 版本变化, 运行时解析（Hell's Gate + 磁盘干净副本回退）。
 * 调用方（logi_driver.cpp）通过函数指针调用以下 stub, 失败时回退 GetProcAddress。
 */

#pragma once

#ifdef __cplusplus
#include <windows.h>
#include <stdint.h>
namespace Syscall {

/* 初始化: 提取 SSN + 解析 ntdll 内 syscall gadget + 检测/恢复 ntdll stub。
 * 返回 false 表示任一关键函数解析失败, 调用方应回退常规 GetProcAddress 路径。 */
bool init(void);

/* ntdll stub 完整性检测: 任一关键函数 prologue/尾部被 patch 即判定被 hook。 */
bool detectNtdllHooks(void);

/* 恢复 ntdll stub: 映射磁盘干净 ntdll, 逐函数覆盖被 patch 的 12 字节。
 * 返回恢复数量。 */
int restoreNtdllStubs(void);

} // namespace Syscall
#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* MASM stub 入口 (syscall_stubs.asm)。函数指针按 logi_driver.cpp 的 typedef 转换。 */

LONG SysNtCreateFile(
    PHANDLE FileHandle, ULONG DesiredAccess,
    void *ObjectAttributes, void *IoStatusBlock,
    PLARGE_INTEGER AllocationSize, ULONG FileAttributes,
    ULONG ShareAccess, ULONG CreateDisposition,
    ULONG CreateOptions, void *EaBuffer, ULONG EaLength);

LONG SysNtDeviceIoControlFile(
    HANDLE FileHandle, HANDLE Event, void *ApcRoutine, void *ApcContext,
    void *IoStatusBlock, ULONG IoControlCode,
    void *InputBuffer, ULONG InputBufferLength,
    void *OutputBuffer, ULONG OutputBufferLength);

LONG SysNtClose(HANDLE Handle);

LONG SysNtOpenDirectoryObject(
    PHANDLE DirectoryHandle, ULONG DesiredAccess, void *ObjectAttributes);

LONG SysNtQueryDirectoryObject(
    HANDLE DirectoryHandle, void *Buffer, ULONG BufferLength,
    BOOLEAN ReturnSingleEntry, BOOLEAN RestartScan,
    PULONG Context, PULONG ReturnLength);

#ifdef __cplusplus
}
#endif
