/**
 * syscall_init.cpp - SSN 提取 + ntdll syscall gadget 解析 + hook 检测/恢复
 *
 * 流程 (Hell's Gate + 磁盘干净副本回退):
 *   1. PEB->Ldr 遍历模块表找 ntdll.dll 基址 (模块名/函数名均 FNV-1a 哈希匹配,
 *      无明文 IAT 依赖、无明文函数名字符串)
 *   2. live ntdll 导出表解析函数地址, 校验 12 字节 prologue 完好则直接取 SSN
 *   3. live stub 被 hook 时回退: 映射磁盘 System32\ntdll.dll 原字节 (无 SEC_IMAGE),
 *      解析导出表 (RVA->文件偏移), 从干净副本提取 SSN
 *   4. syscall gadget: 优先目标函数自身尾部 0F 05 C3; 被 patch 则扫描任意完好
 *      Nt stub 尾部作通用 gadget (间接 syscall, syscall RIP 落 ntdll)
 *   5. 检测并恢复被 hook 的 ntdll stub (12 字节完整校验, 磁盘副本覆盖)
 *
 * 任一关键函数解析失败 -> init() 返回 false -> 调用方回退 GetProcAddress 路径。
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <stdint.h>
#include "syscall.h"
#include "anticeat.h"

/* ==================== MASM 全局 (syscall_stubs.asm) ==================== */

extern "C" {
extern DWORD g_NtCreateFile_SSN;
extern uint64_t g_NtCreateFile_SyscallAddr;
extern DWORD g_NtDeviceIoControlFile_SSN;
extern uint64_t g_NtDeviceIoControlFile_SyscallAddr;
extern DWORD g_NtClose_SSN;
extern uint64_t g_NtClose_SyscallAddr;
extern DWORD g_NtOpenDirectoryObject_SSN;
extern uint64_t g_NtOpenDirectoryObject_SyscallAddr;
extern DWORD g_NtQueryDirectoryObject_SSN;
extern uint64_t g_NtQueryDirectoryObject_SyscallAddr;
}

namespace {

/* ==================== 目标函数表 ==================== */

struct Target {
    const char *name;
    DWORD *ssn;
    uint64_t *syscallAddr;
};

static const Target kTargets[] = {
    {"NtCreateFile", &g_NtCreateFile_SSN, &g_NtCreateFile_SyscallAddr},
    {"NtDeviceIoControlFile", &g_NtDeviceIoControlFile_SSN,
     &g_NtDeviceIoControlFile_SyscallAddr},
    {"NtClose", &g_NtClose_SSN, &g_NtClose_SyscallAddr},
    {"NtOpenDirectoryObject", &g_NtOpenDirectoryObject_SSN,
     &g_NtOpenDirectoryObject_SyscallAddr},
    {"NtQueryDirectoryObject", &g_NtQueryDirectoryObject_SSN,
     &g_NtQueryDirectoryObject_SyscallAddr},
};

/* hook 检测清单: 本模块用到的 + 反作弊常见 hook 目标 */
static const char *const kCheckFns[] = {
    "NtCreateFile", "NtDeviceIoControlFile", "NtClose",
    "NtOpenKey", "NtQueryInformationProcess", "NtSetInformationThread",
    "NtQuerySystemInformation", "NtOpenDirectoryObject",
    "NtQueryDirectoryObject",
};

/* ==================== FNV-1a (大小写不敏感) ==================== */

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

/* ==================== PEB 模块遍历 (无 GetModuleHandleW 依赖) ==================== */

/* ntdll 内部结构布局偏移 (x64), 全部用直接偏移访问, 避免结构体 packing 疑云 */
typedef struct _HIDE_UNICODE_STRING {
    USHORT Length;
    USHORT MaximumLength;
    PWSTR Buffer;
} HIDE_UNICODE_STRING;

#define LDR_INMEMORY_ORDER_LIST_OFFSET 0x20 /* PEB_LDR_DATA.InMemoryOrderModuleList */
#define LDR_DLLBASE_OFFSET             0x20 /* DllBase 相对 InMemoryOrderLinks */
#define LDR_BASENAME_OFFSET            0x48 /* BaseDllName 相对 InMemoryOrderLinks */

static uintptr_t getNtdllBase()
{
#if defined(_WIN64)
    uintptr_t ldr = *(uintptr_t *)(__readgsqword(0x60) + 0x18); /* PEB->Ldr */
#else
    uintptr_t ldr = *(uintptr_t *)(__readfsdword(0x30) + 0x0C); /* PEB->Ldr */
#endif
    if (!ldr)
        return 0;

    /* 与遍历逻辑一致: 模块名只 hash 到 '.' 前 (比较 "ntdll") */
    const uint64_t hashNtdll = fnv1aLower("ntdll");

    LIST_ENTRY *head = (LIST_ENTRY *)(ldr + LDR_INMEMORY_ORDER_LIST_OFFSET);
    for (LIST_ENTRY *e = head->Flink; e != head; e = e->Flink) {
        uintptr_t dllBase = *(uintptr_t *)((uint8_t *)e + LDR_DLLBASE_OFFSET);
        if (!dllBase)
            continue;
        HIDE_UNICODE_STRING *name =
            (HIDE_UNICODE_STRING *)((uint8_t *)e + LDR_BASENAME_OFFSET);
        if (!name->Buffer || name->Length == 0)
            continue;
        size_t chars = name->Length / sizeof(WCHAR);
        /* 模块名遍历到 '.' 即止 (比较 "ntdll") */
        uint64_t h = 0xcbf29ce484222325ull;
        size_t i = 0;
        for (; i < chars; i++) {
            WCHAR c = name->Buffer[i];
            if (c == L'.')
                break;
            if (c >= L'A' && c <= L'Z')
                c = (WCHAR)(c - L'A' + L'a');
            if (c > 0x7F)
                break;
            h ^= (uint8_t)c;
            h *= 0x100000001b3ull;
        }
        if (h == hashNtdll)
            return dllBase;
    }
    return 0;
}

/* ==================== live ntdll 导出表解析 ==================== */

static void *resolveExportLive(uintptr_t base, uint64_t nameHash)
{
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return NULL;
    const IMAGE_NT_HEADERS *nt =
        (const IMAGE_NT_HEADERS *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE)
        return NULL;
    const IMAGE_DATA_DIRECTORY *ed =
        &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!ed->VirtualAddress || !ed->Size)
        return NULL;

    const IMAGE_EXPORT_DIRECTORY *exp =
        (const IMAGE_EXPORT_DIRECTORY *)(base + ed->VirtualAddress);
    const DWORD *names = (const DWORD *)(base + exp->AddressOfNames);
    const WORD *ords = (const WORD *)(base + exp->AddressOfNameOrdinals);
    const DWORD *funcs = (const DWORD *)(base + exp->AddressOfFunctions);

    for (DWORD i = 0; i < exp->NumberOfNames; i++) {
        const char *nm = (const char *)(base + names[i]);
        if (fnv1aLower(nm) == nameHash) {
            DWORD rva = funcs[ords[i]];
            if (rva < ed->VirtualAddress || rva >= ed->VirtualAddress + ed->Size)
                return (void *)(base + rva);
            /* 转发导出 (forwarder), 忽略 */
            return NULL;
        }
    }
    return NULL;
}

/* ==================== 磁盘 ntdll 映射 (干净副本) ==================== */

struct DiskNtdll {
    HANDLE file;
    HANDLE mapping;
    BYTE *view;
    size_t size;
    bool ok;

    DiskNtdll() : file(INVALID_HANDLE_VALUE), mapping(NULL), view(NULL),
                  size(0), ok(false) {}

    bool map()
    {
        WCHAR sysDir[MAX_PATH];
        UINT n = GetSystemDirectoryW(sysDir, MAX_PATH);
        if (n == 0 || n >= MAX_PATH - 10)
            return false;
        wcscat_s(sysDir, L"\\ntdll.dll");

        file = CreateFileW(sysDir, GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file == INVALID_HANDLE_VALUE)
            return false;

        mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!mapping) {
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
            return false;
        }
        /* 不带 SEC_IMAGE: 得到磁盘原始字节, 不受 live 进程 hook 影响 */
        view = (BYTE *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            CloseHandle(mapping);
            CloseHandle(file);
            mapping = NULL;
            file = INVALID_HANDLE_VALUE;
            return false;
        }
        IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)view;
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
            unmap();
            return false;
        }
        IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(view + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) {
            unmap();
            return false;
        }
        size = nt->OptionalHeader.SizeOfImage;
        ok = true;
        return true;
    }

    void unmap()
    {
        if (view)
            UnmapViewOfFile(view);
        if (mapping)
            CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
        view = NULL;
        mapping = NULL;
        file = INVALID_HANDLE_VALUE;
        ok = false;
    }

    ~DiskNtdll() { unmap(); }
};

/* RVA -> 文件偏移 */
static bool rvaToFileOffset(const DiskNtdll &d, DWORD rva, DWORD *offset)
{
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)d.view;
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(d.view + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; i++, sec++) {
        if (rva >= sec->VirtualAddress &&
            rva < sec->VirtualAddress + max(sec->Misc.VirtualSize, sec->SizeOfRawData)) {
            *offset = sec->PointerToRawData + (rva - sec->VirtualAddress);
            return true;
        }
    }
    return false;
}

/* 磁盘副本导出表解析: 返回指向干净 stub 的指针 (view 内) */
static BYTE *resolveExportDisk(const DiskNtdll &d, uint64_t nameHash)
{
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)d.view;
    IMAGE_NT_HEADERS *nt = (IMAGE_NT_HEADERS *)(d.view + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY *ed =
        &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!ed->VirtualAddress || !ed->Size)
        return NULL;

    DWORD expOff;
    if (!rvaToFileOffset(d, ed->VirtualAddress, &expOff))
        return NULL;
    const IMAGE_EXPORT_DIRECTORY *exp =
        (const IMAGE_EXPORT_DIRECTORY *)(d.view + expOff);

    DWORD namesOff, ordsOff, funcsOff;
    if (!rvaToFileOffset(d, exp->AddressOfNames, &namesOff) ||
        !rvaToFileOffset(d, exp->AddressOfNameOrdinals, &ordsOff) ||
        !rvaToFileOffset(d, exp->AddressOfFunctions, &funcsOff))
        return NULL;

    const DWORD *names = (const DWORD *)(d.view + namesOff);
    const WORD *ords = (const WORD *)(d.view + ordsOff);
    const DWORD *funcs = (const DWORD *)(d.view + funcsOff);

    for (DWORD i = 0; i < exp->NumberOfNames; i++) {
        DWORD nameOff;
        if (!rvaToFileOffset(d, names[i], &nameOff))
            continue;
        if (fnv1aLower((const char *)(d.view + nameOff)) == nameHash) {
            DWORD fnOff;
            if (!rvaToFileOffset(d, funcs[ords[i]], &fnOff))
                return NULL;
            return d.view + fnOff;
        }
    }
    return NULL;
}

/* ==================== stub 校验 ==================== */

/* ntdll Nt stub 标准 11 字节:
 *   4C 8B D1 | B8 xx xx 00 00 | 0F 05 | C3
 *   r10=rcx  | eax=SSN       | syscall| ret */
/* prologue 校验 (SSN 提取用): 只要求前 4 字节 + SSN 合理。
 * 某些 hook 保留 prologue 与 SSN (转发需要), 只 patch 尾部 syscall,
 * 此时 SSN 仍可安全提取。 */
static bool validPrologue(const BYTE *p)
{
    return p[0] == 0x4C && p[1] == 0x8B && p[2] == 0xD1 && p[3] == 0xB8;
}

/* 完整 12 字节校验 (gadget 定位/hook 检测用): 尾部必须仍是 syscall;ret */
static bool validStub(const BYTE *p)
{
    return validPrologue(p) &&
           p[8] == 0x0F && p[9] == 0x05 && p[10] == 0xC3;
}

static DWORD stubSsn(const BYTE *p)
{
    return *(const DWORD *)(p + 4);
}

/* ==================== 通用 syscall gadget 扫描 ==================== */

/* 在 live ntdll 导出表中找任意 prologue 完好的 Nt stub, 取其尾部 syscall;ret。
 * SSN 已在 eax, syscall 与具体函数无关, 任何完好的尾部都可用。 */
static uint64_t findAnyGadget(uintptr_t base)
{
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)base;
    const IMAGE_NT_HEADERS *nt =
        (const IMAGE_NT_HEADERS *)(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY *ed =
        &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!ed->VirtualAddress || !ed->Size)
        return 0;

    const IMAGE_EXPORT_DIRECTORY *exp =
        (const IMAGE_EXPORT_DIRECTORY *)(base + ed->VirtualAddress);
    const DWORD *names = (const DWORD *)(base + exp->AddressOfNames);
    const WORD *ords = (const WORD *)(base + exp->AddressOfNameOrdinals);
    const DWORD *funcs = (const DWORD *)(base + exp->AddressOfFunctions);

    for (DWORD i = 0; i < exp->NumberOfNames; i++) {
        const char *nm = (const char *)(base + names[i]);
        /* 只取 Nt* 前缀, 减小误判面 */
        if (!(nm[0] == 'N' && nm[1] == 't'))
            continue;
        DWORD rva = funcs[ords[i]];
        const BYTE *p = (const BYTE *)(base + rva);
        if (validStub(p))
            return (uint64_t)(p + 8);
    }
    return 0;
}

/* ==================== hook 检测 / 恢复 ==================== */

static bool detectHooksInternal(uintptr_t base)
{
    for (size_t i = 0; i < sizeof(kCheckFns) / sizeof(kCheckFns[0]); i++) {
        void *fn = resolveExportLive(base, fnv1aLower(kCheckFns[i]));
        if (!fn)
            continue; /* 某些函数可能被转发或不存在, 不参与判定 */
        if (!validStub((const BYTE *)fn))
            return true;
    }
    return false;
}

static int restoreStubsInternal(uintptr_t base, const DiskNtdll &d)
{
    int restored = 0;
    for (size_t i = 0; i < sizeof(kCheckFns) / sizeof(kCheckFns[0]); i++) {
        void *live = resolveExportLive(base, fnv1aLower(kCheckFns[i]));
        BYTE *clean = resolveExportDisk(d, fnv1aLower(kCheckFns[i]));
        if (!live || !clean)
            continue;
        if (memcmp(live, clean, 12) == 0)
            continue;
        DWORD oldProtect = 0;
        if (VirtualProtect(live, 12, PAGE_EXECUTE_READWRITE, &oldProtect)) {
            memcpy(live, clean, 12);
            VirtualProtect(live, 12, oldProtect, &oldProtect);
            FlushInstructionCache(GetCurrentProcess(), live, 12);
            restored++;
        }
    }
    return restored;
}

} // namespace

/* ==================== 公共接口 ==================== */

namespace Syscall {

bool init(void)
{
    uintptr_t base = getNtdllBase();
    if (!base)
        return false;
    DiskNtdll disk;
    disk.map(); /* 可选; 失败则只用 live 路径 */

    uint64_t backupGadget = 0;

    for (size_t i = 0; i < sizeof(kTargets) / sizeof(kTargets[0]); i++) {
        const Target &t = kTargets[i];
        uint64_t hash = fnv1aLower(t.name);
        *t.ssn = 0;
        *t.syscallAddr = 0;

        /* 1. live stub (Hell's Gate): SSN 提取只看 prologue,
         *    尾部被 hook (0F 05 C3 被 patch) 不影响 SSN 提取 */
        void *live = resolveExportLive(base, hash);
        DWORD ssn = 0;
        if (live && validPrologue((const BYTE *)live)) {
            ssn = stubSsn((const BYTE *)live);
            /* 尾部完好则用自身 syscall;ret 作 gadget, 否则稍后通用回退 */
            if (validStub((const BYTE *)live))
                *t.syscallAddr = (uint64_t)((const BYTE *)live + 8);
        } else if (disk.ok) {
            /* 2. 磁盘干净副本回退 (仅取 SSN; gadget 需 live 地址, 走通用回退) */
            BYTE *clean = resolveExportDisk(disk, hash);
            if (clean && validPrologue(clean))
                ssn = stubSsn(clean);
        }
        *t.ssn = ssn;

        /* 3. gadget 通用回退: 自身尾部被 hook 时, 找任意完好 Nt stub 的
         *    syscall;ret (SSN 已在 eax, syscall 与具体函数无关) */
        if (*t.syscallAddr == 0) {
            if (backupGadget == 0)
                backupGadget = findAnyGadget(base);
            if (ssn != 0 && backupGadget != 0)
                *t.syscallAddr = backupGadget;
        }
    }

    /* SSN 完整性校验: 非零且 < 0x200 (Win10/11 已知最大 SSN 约 460) */
    for (size_t i = 0; i < sizeof(kTargets) / sizeof(kTargets[0]); i++) {
        if (*kTargets[i].ssn == 0 || *kTargets[i].ssn >= 0x200)
            return false;
    }

    /* hook 检测 + 恢复 (纵深防御; 失败不影响已提取的 SSN)。
     * 顶级反作弊环境下绝不改写 ntdll 内存 — 代码完整性校验是反作弊主力检测,
     * restore 会把 ntdll 页面改动暴露给校验。此时仅检测, 不恢复。 */
    if (detectHooksInternal(base)) {
        if (!AntiCheat::detectTop() && disk.ok)
            restoreStubsInternal(base, disk);
    }

    return true;
}

bool detectNtdllHooks(void)
{
    uintptr_t base = getNtdllBase();
    if (!base)
        return false;
    return detectHooksInternal(base);
}

int restoreNtdllStubs(void)
{
    uintptr_t base = getNtdllBase();
    if (!base)
        return 0;
    DiskNtdll disk;
    if (!disk.map())
        return 0;
    return restoreStubsInternal(base, disk);
}

} // namespace Syscall
