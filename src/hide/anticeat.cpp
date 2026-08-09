/**
 * anticeat.cpp - 反作弊环境检测实现
 *
 * 用 CreateToolhelp32Snapshot 枚举进程, 进程名 (szExeFile, 已含 .exe)
 * 小写化后做 FNV-1a 哈希, 与顶级反作弊进程名单比对。
 *
 * 名单覆盖:
 *   - EAC:        EasyAntiCheat.exe
 *   - BattlEye:   BEService.exe
 *   - Vanguard:   vgc.exe
 *   - ACE:        ACE-BASE.exe / ACE-Guard.exe
 *   - GameGuard:  npggsvc.exe / GameMon.des
 *   - XTrap:      XTrapManager.exe / XTrapSvc.exe
 *   - TenProtect: TP3Helper.exe / TenioDL.exe
 *   - 平台自带:   5EClient.exe / FaceItClient.exe
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include "anticeat.h"

namespace {

/* FNV-1a 64 (小写化) */
static uint64_t fnv1aLower(const char *s)
{
    uint64_t h = 0xcbf29ce484222325ull;
    while (*s) {
        char c = *s++;
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
        h ^= (uint8_t)c;
        h *= 0x100000001b3ull;
    }
    return h;
}

/* 顶级反作弊进程名单 (哈希, 运行时常量折叠) */
static const uint64_t kTopAcHashes[] = {
    fnv1aLower("EasyAntiCheat.exe"),
    fnv1aLower("BEService.exe"),
    fnv1aLower("vgc.exe"),
    fnv1aLower("ACE-BASE.exe"),
    fnv1aLower("ACE-Guard.exe"),
    fnv1aLower("npggsvc.exe"),
    fnv1aLower("GameMon.des"),
    fnv1aLower("XTrapManager.exe"),
    fnv1aLower("XTrapSvc.exe"),
    fnv1aLower("TP3Helper.exe"),
    fnv1aLower("TenioDL.exe"),
    fnv1aLower("5EClient.exe"),
    fnv1aLower("FaceItClient.exe"),
};

static bool isTopAcName(const WCHAR *name)
{
    /* 宽字符转小写 ASCII 后哈希 */
    uint64_t h = 0xcbf29ce484222325ull;
    for (const WCHAR *p = name; *p; p++) {
        WCHAR c = *p;
        if (c >= L'A' && c <= L'Z')
            c = (WCHAR)(c - L'A' + L'a');
        if (c > 0x7F)
            return false;
        h ^= (uint8_t)c;
        h *= 0x100000001b3ull;
    }
    for (size_t i = 0; i < sizeof(kTopAcHashes) / sizeof(kTopAcHashes[0]); i++) {
        if (h == kTopAcHashes[i])
            return true;
    }
    return false;
}

} // namespace

namespace AntiCheat {

bool detectTop(void)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE)
        return false;

    bool found = false;
    PROCESSENTRY32W pe;
    pe.dwSize = sizeof(pe);

    if (Process32FirstW(snap, &pe)) {
        do {
            if (isTopAcName(pe.szExeFile)) {
                found = true;
                break;
            }
        } while (Process32NextW(snap, &pe));
    }

    CloseHandle(snap);
    return found;
}

} // namespace AntiCheat
