/**
 * logi_driver.cpp - 多驱动鼠标模拟（GHUB + LGS + Razer）
 *
 * 从独立DLL适配为插件内部静态链接模块。
 * 移除DllMain和DLL导出宏，新增互斥锁保护全局状态。
 *
 * NT原生API结构体在x64下布局正确（使用ULONG_PTR等宽度类型）。
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <setupapi.h>
#include "logi_driver.h"

#pragma comment(lib, "setupapi.lib")

/* ==================== NT Native API ==================== */

typedef struct _IO_STATUS_BLOCK_NT {
    union { LONG Status; PVOID Pointer; };
    ULONG_PTR Information;
} IO_STATUS_BLOCK_NT;

typedef struct _UNICODE_STRING_NT {
    USHORT Length;
    USHORT MaximumLength;
    PWSTR  Buffer;
} UNICODE_STRING_NT;

typedef struct _OBJECT_ATTRIBUTES_NT {
    ULONG              Length;
    HANDLE             RootDirectory;
    UNICODE_STRING_NT* ObjectName;
    ULONG              Attributes;
    PVOID              SecurityDescriptor;
    PVOID              SecurityQualityOfService;
} OBJECT_ATTRIBUTES_NT;

typedef LONG (WINAPI *fn_NtCreateFile)(
    PHANDLE, ULONG, OBJECT_ATTRIBUTES_NT*, IO_STATUS_BLOCK_NT*,
    PLARGE_INTEGER, ULONG, ULONG, ULONG, ULONG, PVOID, ULONG);

typedef LONG (WINAPI *fn_NtDeviceIoControlFile)(
    HANDLE, HANDLE, PVOID, PVOID, IO_STATUS_BLOCK_NT*,
    ULONG, PVOID, ULONG, PVOID, ULONG);

typedef LONG (WINAPI *fn_NtClose)(HANDLE);

typedef void (WINAPI *fn_RtlInitUnicodeString)(
    UNICODE_STRING_NT*, PCWSTR);

/* ==================== NT Directory Object API (Razer枚举) ==================== */

typedef struct _OBJECT_DIRECTORY_INFORMATION {
    UNICODE_STRING_NT Name;
    UNICODE_STRING_NT TypeName;
} OBJECT_DIRECTORY_INFORMATION;

typedef LONG (WINAPI *fn_NtOpenDirectoryObject)(
    PHANDLE, ULONG, OBJECT_ATTRIBUTES_NT*);
typedef LONG (WINAPI *fn_NtQueryDirectoryObject)(
    HANDLE, PVOID, ULONG, BOOLEAN, BOOLEAN, PULONG, PULONG);

#define DIRECTORY_QUERY  0x0001
#define STATUS_BUFFER_TOO_SMALL ((LONG)0xC0000023L)
#define STATUS_MORE_ENTRIES    ((LONG)0x00000105L)
#define OBJ_CASE_INSENSITIVE  0x00000040

/* ==================== 常量 ==================== */

#define LGHUB_MOUSE_IOCTL   0x2a2010

/* IOCTL鼠标报告结构体 */
#pragma pack(push, 1)
typedef struct _MOUSE_IO_5 {
    char  button;
    char  x;
    char  y;
    char  wheel;
    char  unk;
} MOUSE_IO_5;

typedef struct _MOUSE_IO_7 {
    char  button;
    char  reserved;
    short x;
    short y;
    char  wheel;
} MOUSE_IO_7;
#pragma pack(pop)

/* Razer 32字节IOCTL缓冲区 */
typedef struct _RAZER_REPORT {
    DWORD reserved1;
    DWORD type;
    DWORD flags;
    DWORD button;
    DWORD reserved3;
    DWORD dx;
    DWORD dy;
    DWORD reserved4;
} RAZER_REPORT;

static const GUID GUID_DEVINTERFACE_HID_LOCAL =
    {0x4d1e55b2, 0xf16f, 0x11cf, {0x88, 0xcb, 0x00, 0x11, 0x11, 0x00, 0x00, 0x30}};

/* 驱动类型标识 */
#define DRIVER_TYPE_NONE   0
#define DRIVER_TYPE_LGS    1
#define DRIVER_TYPE_GHUB   2
#define DRIVER_TYPE_RAZER  3

/* 按键位掩码（Logitech HID报告） */
#define BTN_NONE    0
#define BTN_LEFT    1
#define BTN_RIGHT   2
#define BTN_MIDDLE  4

/* Razer按键事件标志 */
#define RZ_BTN_LEFT_DOWN    0x0001
#define RZ_BTN_LEFT_UP      0x0002
#define RZ_BTN_RIGHT_DOWN   0x0004
#define RZ_BTN_RIGHT_UP     0x0008
#define RZ_BTN_MIDDLE_DOWN  0x0010
#define RZ_BTN_MIDDLE_UP    0x0020

#define RAZER_IOCTL  0x88883020

/* ==================== 全局状态 ==================== */

static HANDLE               g_device = INVALID_HANDLE_VALUE;
static fn_NtCreateFile          g_NtCreateFile = NULL;
static fn_NtDeviceIoControlFile g_NtDeviceIoControlFile = NULL;
static fn_NtClose               g_NtClose = NULL;
static fn_RtlInitUnicodeString  g_RtlInitUnicodeString = NULL;
static fn_NtOpenDirectoryObject g_NtOpenDirectoryObject = NULL;
static fn_NtQueryDirectoryObject g_NtQueryDirectoryObject = NULL;
static BOOL                     g_ntLoaded = FALSE;
static int                      g_driver_type = DRIVER_TYPE_NONE;
static int                      g_forced_type = DRIVER_TYPE_NONE;

/* 互斥锁保护全局状态，防止多线程竞态 */
static CRITICAL_SECTION g_mutex;
static BOOL g_mutex_initialized = FALSE;

/* ==================== 内部辅助函数 ==================== */

static BOOL load_nt_functions(void)
{
    HMODULE ntdll;

    if (g_ntLoaded) return TRUE;

    ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) ntdll = LoadLibraryW(L"ntdll.dll");
    if (!ntdll) return FALSE;

    g_NtCreateFile = (fn_NtCreateFile)
        GetProcAddress(ntdll, "NtCreateFile");
    g_NtDeviceIoControlFile = (fn_NtDeviceIoControlFile)
        GetProcAddress(ntdll, "NtDeviceIoControlFile");
    g_NtClose = (fn_NtClose)
        GetProcAddress(ntdll, "NtClose");
    g_RtlInitUnicodeString = (fn_RtlInitUnicodeString)
        GetProcAddress(ntdll, "RtlInitUnicodeString");
    g_NtOpenDirectoryObject = (fn_NtOpenDirectoryObject)
        GetProcAddress(ntdll, "NtOpenDirectoryObject");
    g_NtQueryDirectoryObject = (fn_NtQueryDirectoryObject)
        GetProcAddress(ntdll, "NtQueryDirectoryObject");

    g_ntLoaded = (g_NtCreateFile && g_NtDeviceIoControlFile &&
                  g_NtClose && g_RtlInitUnicodeString);
    return g_ntLoaded;
}

/* 通过NT路径打开设备（Logitech GHUB/LGS） */
static BOOL try_open_path(const WCHAR* path)
{
    UNICODE_STRING_NT name;
    OBJECT_ATTRIBUTES_NT attr;
    IO_STATUS_BLOCK_NT iosb;
    LONG status;

    g_RtlInitUnicodeString(&name, path);

    ZeroMemory(&attr, sizeof(attr));
    attr.Length = sizeof(attr);
    attr.ObjectName = &name;

    ZeroMemory(&iosb, sizeof(iosb));

    status = g_NtCreateFile(
        &g_device,
        0x40100000,                  /* GENERIC_WRITE | SYNCHRONIZE */
        &attr,
        &iosb,
        0,
        0x80,                        /* FILE_ATTRIBUTE_NORMAL */
        0,
        3,                           /* FILE_OPEN_IF */
        0x00000060,                  /* NON_DIRECTORY_FILE | SYNCHRONOUS_IO_NONALERT */
        0, 0);

    return (status == 0);
}

/* ==================== Logitech设备发现 ==================== */

static BOOL try_open_logitech_device(int index)
{
    static const WCHAR* ghub_paths[] = {
        L"\\??\\ROOT#SYSTEM#0000#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0003#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0004#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0005#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0006#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0007#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0008#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
        L"\\??\\ROOT#SYSTEM#0009#{1abc05c0-c378-41b9-9cef-df1aba82b015}",
    };
    static const WCHAR* lgs_paths[] = {
        L"\\??\\ROOT#SYSTEM#0000#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0001#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0002#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0003#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0004#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0005#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0006#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0007#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0008#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
        L"\\??\\ROOT#SYSTEM#0009#{df31f106-d870-453d-8fa1-ec8ab43fa1d2}",
    };

    if (index < 0 || index > 9) return FALSE;

    if (g_forced_type == DRIVER_TYPE_NONE || g_forced_type == DRIVER_TYPE_GHUB) {
        if (try_open_path(ghub_paths[index])) {
            g_driver_type = DRIVER_TYPE_GHUB;
            return TRUE;
        }
    }
    if (g_forced_type == DRIVER_TYPE_NONE || g_forced_type == DRIVER_TYPE_LGS) {
        if (try_open_path(lgs_paths[index])) {
            g_driver_type = DRIVER_TYPE_LGS;
            return TRUE;
        }
    }
    return FALSE;
}

/* ==================== Razer设备发现 ==================== */

static BOOL wstr_contains_i(const WCHAR* haystack, int hayLen, const WCHAR* needle, int needleLen)
{
    if (needleLen > hayLen) return FALSE;
    for (int i = 0; i <= hayLen - needleLen; i++) {
        BOOL match = TRUE;
        for (int j = 0; j < needleLen; j++) {
            WCHAR a = haystack[i + j];
            WCHAR b = needle[j];
            if (a >= L'a' && a <= L'z') a -= 32;
            if (b >= L'a' && b <= L'z') b -= 32;
            if (a != b) { match = FALSE; break; }
        }
        if (match) return TRUE;
    }
    return FALSE;
}

static BOOL try_open_razer_candidate(const WCHAR* devPath, DWORD desiredAccess)
{
    HANDLE h = CreateFileW(devPath, desiredAccess,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE)
        return FALSE;

    RAZER_REPORT testReport;
    ZeroMemory(&testReport, sizeof(testReport));
    testReport.type = 2;
    DWORD bytesReturned = 0;

    if (DeviceIoControl(h, RAZER_IOCTL,
            &testReport, sizeof(testReport),
            NULL, 0, &bytesReturned, NULL))
    {
        g_device = h;
        g_driver_type = DRIVER_TYPE_RAZER;
        return TRUE;
    }

    CloseHandle(h);
    return FALSE;
}

static BOOL try_open_razer_device(void)
{
    if (!g_NtOpenDirectoryObject || !g_NtQueryDirectoryObject)
        return FALSE;

    UNICODE_STRING_NT dirName;
    g_RtlInitUnicodeString(&dirName, L"\\GLOBAL??");

    OBJECT_ATTRIBUTES_NT oa;
    ZeroMemory(&oa, sizeof(oa));
    oa.Length = sizeof(oa);
    oa.ObjectName = &dirName;
    oa.Attributes = OBJ_CASE_INSENSITIVE;

    HANDLE hDir = NULL;
    LONG st = g_NtOpenDirectoryObject(&hDir, DIRECTORY_QUERY, &oa);
    if (st < 0 || !hDir) return FALSE;

    BOOL found = FALSE;

    ULONG returnLength = 0;
    ULONG context = 0;
    st = g_NtQueryDirectoryObject(hDir, NULL, 0, FALSE, TRUE, &context, &returnLength);
    if (st != (LONG)STATUS_BUFFER_TOO_SMALL || returnLength == 0) {
        returnLength = 65536;
    }

    ULONG bufSize = returnLength < 65536 ? 65536 : returnLength;
    BYTE* buf = (BYTE*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, bufSize);
    if (!buf) {
        g_NtClose(hDir);
        return FALSE;
    }

    context = 0;
    for (;;) {
        BOOL restartScan = (context == 0) ? TRUE : FALSE;
        st = g_NtQueryDirectoryObject(
            hDir, buf, bufSize, FALSE, restartScan, &context, &returnLength);
        if (st != 0 && st != STATUS_MORE_ENTRIES)
            break;

        OBJECT_DIRECTORY_INFORMATION* info = (OBJECT_DIRECTORY_INFORMATION*)buf;
        while (info->Name.Buffer != NULL && info->Name.Length > 0) {
            int nameChars = info->Name.Length / sizeof(WCHAR);

            if (wstr_contains_i(info->Name.Buffer, nameChars, L"RZCONTROL", 9)) {
                WCHAR devPath[512];
                devPath[0] = L'\\';
                devPath[1] = L'\\';
                devPath[2] = L'?';
                devPath[3] = L'\\';
                int copyLen = nameChars;
                if (copyLen > 500) copyLen = 500;
                for (int i = 0; i < copyLen; i++)
                    devPath[4 + i] = info->Name.Buffer[i];
                devPath[4 + copyLen] = L'\0';

                if (try_open_razer_candidate(devPath, 0)) {
                    found = TRUE;
                    break;
                }
            }
            info++;
        }

        if (found || st != STATUS_MORE_ENTRIES)
            break;
    }

    HeapFree(GetProcessHeap(), 0, buf);
    g_NtClose(hDir);
    return found;
}

static BOOL try_open_razer_hid_interface(void)
{
    HDEVINFO devInfo = SetupDiGetClassDevsW(
        &GUID_DEVINTERFACE_HID_LOCAL, NULL, NULL,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devInfo == INVALID_HANDLE_VALUE)
        return FALSE;

    SP_DEVICE_INTERFACE_DATA ifData;
    ZeroMemory(&ifData, sizeof(ifData));
    ifData.cbSize = sizeof(ifData);

    BOOL found = FALSE;

    for (DWORD index = 0;
         SetupDiEnumDeviceInterfaces(devInfo, NULL, &GUID_DEVINTERFACE_HID_LOCAL, index, &ifData);
         index++)
    {
        DWORD requiredSize = 0;
        (void)SetupDiGetDeviceInterfaceDetailW(devInfo, &ifData, NULL, 0, &requiredSize, NULL);
        if (requiredSize < sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W))
            continue;

        BYTE* buf = (BYTE*)HeapAlloc(GetProcessHeap(), 0, requiredSize);
        if (!buf)
            continue;

        SP_DEVICE_INTERFACE_DETAIL_DATA_W* detail =
            (SP_DEVICE_INTERFACE_DETAIL_DATA_W*)buf;
        detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);

        if (SetupDiGetDeviceInterfaceDetailW(devInfo, &ifData, detail, requiredSize, NULL, NULL)) {
            int pathLen = (int)lstrlenW(detail->DevicePath);

            if (wstr_contains_i(detail->DevicePath, pathLen, L"VID_1532", 8) &&
                (wstr_contains_i(detail->DevicePath, pathLen, L"MI_01", 5) ||
                 wstr_contains_i(detail->DevicePath, pathLen, L"COL01", 5) ||
                 wstr_contains_i(detail->DevicePath, pathLen, L"COL02", 5)))
            {
                if (try_open_razer_candidate(detail->DevicePath, 0) ||
                    try_open_razer_candidate(detail->DevicePath, GENERIC_READ | GENERIC_WRITE))
                {
                    found = TRUE;
                    HeapFree(GetProcessHeap(), 0, buf);
                    break;
                }
            }
        }

        HeapFree(GetProcessHeap(), 0, buf);
    }

    SetupDiDestroyDeviceInfoList(devInfo);
    return found;
}

/* ==================== IOCTL分发 ==================== */

static int send_ioctl(void* buf, ULONG size)
{
    IO_STATUS_BLOCK_NT iosb;
    LONG status;

    if (g_device == INVALID_HANDLE_VALUE) return 0;

    /* Razer使用DeviceIoControl */
    if (g_driver_type == DRIVER_TYPE_RAZER) {
        DWORD bytesReturned = 0;
        if (DeviceIoControl(g_device, RAZER_IOCTL,
                buf, size, NULL, 0, &bytesReturned, NULL))
        {
            return 1;
        }
        /* 失败时自动重连 */
        CloseHandle(g_device);
        g_device = INVALID_HANDLE_VALUE;
        g_driver_type = DRIVER_TYPE_NONE;
        if (try_open_razer_device()) {
            bytesReturned = 0;
            if (DeviceIoControl(g_device, RAZER_IOCTL,
                    buf, size, NULL, 0, &bytesReturned, NULL))
            {
                return 1;
            }
        }
        return 0;
    }

    /* Logitech使用NtDeviceIoControlFile */
    ZeroMemory(&iosb, sizeof(iosb));
    status = g_NtDeviceIoControlFile(
        g_device, NULL, NULL, NULL, &iosb,
        LGHUB_MOUSE_IOCTL, buf, size, NULL, 0);

    /* 失败时自动重连 */
    if (status != 0) {
        if (g_device != INVALID_HANDLE_VALUE) {
            g_NtClose(g_device);
            g_device = INVALID_HANDLE_VALUE;
        }
        g_driver_type = DRIVER_TYPE_NONE;

        for (int i = 0; i < 10; i++) {
            if (try_open_logitech_device(i)) {
                ZeroMemory(&iosb, sizeof(iosb));
                status = g_NtDeviceIoControlFile(
                    g_device, NULL, NULL, NULL, &iosb,
                    LGHUB_MOUSE_IOCTL, buf, size, NULL, 0);
                if (status == 0) return 1;
                g_NtClose(g_device);
                g_device = INVALID_HANDLE_VALUE;
                g_driver_type = DRIVER_TYPE_NONE;
                break;
            }
        }
        return 0;
    }

    return 1;
}

static int send_mouse_report(char btn, int dx, int dy, char wheel)
{
    if (g_driver_type == DRIVER_TYPE_RAZER) {
        RAZER_REPORT rpt;
        ZeroMemory(&rpt, sizeof(rpt));
        rpt.type = 2;
        rpt.flags = 0;
        rpt.button = (DWORD)(unsigned char)btn;
        rpt.dx = (DWORD)dx;
        rpt.dy = (DWORD)dy;
        (void)wheel;
        return send_ioctl(&rpt, sizeof(rpt));
    } else if (g_driver_type == DRIVER_TYPE_LGS) {
        MOUSE_IO_5 io;
        ZeroMemory(&io, sizeof(io));
        io.button = btn;
        io.x      = (char)(dx > 127 ? 127 : (dx < -127 ? -127 : dx));
        io.y      = (char)(dy > 127 ? 127 : (dy < -127 ? -127 : dy));
        io.wheel  = wheel;
        io.unk    = 0;
        return send_ioctl(&io, sizeof(io));
    } else {
        /* GHUB: 7字节格式，16位增量 */
        MOUSE_IO_7 io;
        ZeroMemory(&io, sizeof(io));
        io.button   = btn;
        io.reserved = 0;
        io.x        = (short)dx;
        io.y        = (short)dy;
        io.wheel    = wheel;
        return send_ioctl(&io, sizeof(io));
    }
}

static char button_to_mask(int button)
{
    switch (button) {
        case 1: return BTN_LEFT;
        case 2: return BTN_RIGHT;
        case 3: return BTN_MIDDLE;
        case 4: return BTN_MIDDLE;
        default: return BTN_LEFT;
    }
}

static DWORD razer_button_down_flag(int button)
{
    switch (button) {
        case 1: return RZ_BTN_LEFT_DOWN;
        case 2: return RZ_BTN_RIGHT_DOWN;
        case 3: return RZ_BTN_MIDDLE_DOWN;
        case 4: return RZ_BTN_MIDDLE_DOWN;
        default: return RZ_BTN_LEFT_DOWN;
    }
}

static DWORD razer_button_up_flag(int button)
{
    switch (button) {
        case 1: return RZ_BTN_LEFT_UP;
        case 2: return RZ_BTN_RIGHT_UP;
        case 3: return RZ_BTN_MIDDLE_UP;
        case 4: return RZ_BTN_MIDDLE_UP;
        default: return RZ_BTN_LEFT_UP;
    }
}

/* ==================== 模块生命周期 ==================== */

void logi_driver_init(void)
{
    if (!g_mutex_initialized) {
        InitializeCriticalSection(&g_mutex);
        g_mutex_initialized = TRUE;
    }
}

void logi_driver_cleanup(void)
{
    if (g_mutex_initialized) {
        EnterCriticalSection(&g_mutex);
        if (g_device != INVALID_HANDLE_VALUE) {
            if (g_NtClose)
                g_NtClose(g_device);
            else
                CloseHandle(g_device);
            g_device = INVALID_HANDLE_VALUE;
        }
        g_driver_type = DRIVER_TYPE_NONE;
        g_forced_type = DRIVER_TYPE_NONE;
        LeaveCriticalSection(&g_mutex);

        DeleteCriticalSection(&g_mutex);
        g_mutex_initialized = FALSE;
    }
}

/* ==================== 导出函数 ==================== */

int device_open(void)
{
    if (!load_nt_functions())
        return 0;

    /* 已打开则先关闭 */
    if (g_device != INVALID_HANDLE_VALUE) {
        g_NtClose(g_device);
        g_device = INVALID_HANDLE_VALUE;
    }
    g_driver_type = DRIVER_TYPE_NONE;

    /* 尝试Logitech（GHUB + LGS），设备索引0..9 */
    if (g_forced_type == DRIVER_TYPE_NONE ||
        g_forced_type == DRIVER_TYPE_GHUB ||
        g_forced_type == DRIVER_TYPE_LGS)
    {
        for (int i = 0; i < 10; i++) {
            if (try_open_logitech_device(i))
                return 1;
        }
    }

    /* 尝试Razer */
    if (g_forced_type == DRIVER_TYPE_NONE || g_forced_type == DRIVER_TYPE_RAZER) {
        if (try_open_razer_device() || try_open_razer_hid_interface())
            return 1;
    }

    return 0;
}

void device_close(void)
{
    if (g_device != INVALID_HANDLE_VALUE) {
        if (g_NtClose)
            g_NtClose(g_device);
        else
            CloseHandle(g_device);
        g_device = INVALID_HANDLE_VALUE;
    }
    g_driver_type = DRIVER_TYPE_NONE;
}

static int clamp_max(void)
{
    return (g_driver_type == DRIVER_TYPE_LGS) ? 127 : 32767;
}

int moveR(int x, int y)
{
    int maxv = clamp_max();
    /* 大位移分块发送 */
    while (x != 0 || y != 0) {
        int cx = x > maxv ? maxv : (x < -maxv ? -maxv : x);
        int cy = y > maxv ? maxv : (y < -maxv ? -maxv : y);
        if (!send_mouse_report(BTN_NONE, cx, cy, 0))
            return 0;
        x -= cx;
        y -= cy;
    }
    return 1;
}

int mouse_down(int button)
{
    if (g_driver_type == DRIVER_TYPE_RAZER) {
        return send_mouse_report((char)razer_button_down_flag(button), 0, 0, 0);
    }
    return send_mouse_report(button_to_mask(button), 0, 0, 0);
}

int mouse_up(int button)
{
    if (g_driver_type == DRIVER_TYPE_RAZER) {
        return send_mouse_report((char)razer_button_up_flag(button), 0, 0, 0);
    }
    (void)button;
    return send_mouse_report(BTN_NONE, 0, 0, 0);
}

int device_open2(int type)
{
    /* type: 0=自动, 1=GHUB, 2=LGS, 3=Razer */
    if (type == 1)
        g_forced_type = DRIVER_TYPE_GHUB;
    else if (type == 2)
        g_forced_type = DRIVER_TYPE_LGS;
    else if (type == 3)
        g_forced_type = DRIVER_TYPE_RAZER;
    else
        g_forced_type = DRIVER_TYPE_NONE;

    return device_open();
}

int get_driver_type(void)
{
    return g_driver_type;
}
