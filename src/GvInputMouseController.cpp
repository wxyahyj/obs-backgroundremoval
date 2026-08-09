#ifdef _WIN32

#include "GvInputMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <string>
#include <vector>
#include <algorithm>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "hid.lib")

#pragma pack(push, 1)
struct GvInputHidReport {
    WORD  magic;
    BYTE  length;
    BYTE  flags;
    BYTE  dx;
    BYTE  dy;
    BYTE  wheel;
	BYTE  padding[58];
};

// Tencent MyApp HID protocol: absolute 0-63 coordinates
// Byte 0 = ReportID=0x50, Byte 1 = X(0-63), Byte 2 = Y(0-63), Bytes 3-64 = zero
#pragma pack(pop)

static HANDLE findAndCreateHidDevice(const wchar_t* matchStr, bool printFirst, const char* logPrefix)
{
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

    HDEVINFO devInfo = SetupDiGetClassDevs(&hidGuid, NULL, NULL,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devInfo == INVALID_HANDLE_VALUE) return INVALID_HANDLE_VALUE;

    DWORD idx = 0;
    SP_DEVICE_INTERFACE_DATA ifaceData = { sizeof(SP_DEVICE_INTERFACE_DATA) };

    while (SetupDiEnumDeviceInterfaces(devInfo, NULL, &hidGuid, idx, &ifaceData)) {
        DWORD required = 0;
        SetupDiGetDeviceInterfaceDetail(devInfo, &ifaceData, NULL, 0, &required, NULL);
        if (required == 0) { idx++; continue; }

        PSP_DEVICE_INTERFACE_DETAIL_DATA detail =
            (PSP_DEVICE_INTERFACE_DETAIL_DATA)malloc(required);
        if (!detail) { idx++; continue; }
        detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        if (SetupDiGetDeviceInterfaceDetail(devInfo, &ifaceData, detail, required, NULL, NULL)) {
            std::wstring devPath(detail->DevicePath);

            if (printFirst && (idx < 12 || devPath.find(L"vhid") != std::wstring::npos ||
                    devPath.find(L"todesk") != std::wstring::npos)) {
                char pathA[128] = {0};
                WideCharToMultiByte(CP_UTF8, 0, detail->DevicePath, -1, pathA, sizeof(pathA)-1, NULL, NULL);
                obs_log(LOG_INFO, "[%s] HID[%lu]: %s", logPrefix, idx, pathA);
            }

            if (devPath.find(matchStr) != std::wstring::npos) {
                obs_log(LOG_INFO, "[%s] Found target device, trying CreateFile...", logPrefix);
                HANDLE h = CreateFile(detail->DevicePath,
                    GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (h != INVALID_HANDLE_VALUE) {
                    free(detail);
                    SetupDiDestroyDeviceInfoList(devInfo);
                    return h;
                } else {
                    DWORD err = GetLastError();
                    obs_log(LOG_WARNING, "[%s] CreateFile(GENERIC_WRITE) failed, error: %lu", logPrefix, err);
                    // Try with both read+write as fallback
                    h = CreateFile(detail->DevicePath,
                        GENERIC_WRITE | GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE,
                        NULL, OPEN_EXISTING, 0, NULL);
                    if (h != INVALID_HANDLE_VALUE) {
                        free(detail);
                        SetupDiDestroyDeviceInfoList(devInfo);
                        return h;
                    }
                    obs_log(LOG_WARNING, "[%s] CreateFile(GENERIC_WRITE|READ) also failed, error: %lu", logPrefix, GetLastError());
                }
            }
        }
        free(detail);
        idx++;
    }
    obs_log(LOG_WARNING, "[%s] target not found among %lu HID devices", logPrefix, idx);
    SetupDiDestroyDeviceInfoList(devInfo);
    return INVALID_HANDLE_VALUE;
}
// ================ GvInput ================

GvInputMouseController::GvInputMouseController()
    : AbstractMouseController()
    , hDevice(INVALID_HANDLE_VALUE)
    , deviceConnected(false)
{
    if (openDevice()) {
        deviceConnected = true;
        obs_log(LOG_INFO, "[GvInput] HID GVInput&Col04 opened");
    } else {
        obs_log(LOG_WARNING, "[GvInput] Failed to open GVInput&Col04");
    }
}

GvInputMouseController::~GvInputMouseController()
{
    closeDevice();
}

bool GvInputMouseController::openDevice()
{
    hDevice = findAndCreateHidDevice(L"gvinput&col04", true, "GvInput");
    return hDevice != INVALID_HANDLE_VALUE;
}

void GvInputMouseController::closeDevice()
{
    if (hDevice != INVALID_HANDLE_VALUE) {
        CloseHandle(hDevice);
        hDevice = INVALID_HANDLE_VALUE;
    }
    deviceConnected = false;
}

void GvInputMouseController::moveMouse(int dx, int dy)
{
    if (!deviceConnected || hDevice == INVALID_HANDLE_VALUE) {
        if (!openDevice()) return;
    }

    GvInputHidReport report = {};
    report.magic  = 0x0540;
    report.length = 4;
    report.flags  = 0;
    report.dx = (BYTE)(char)(dx > 127 ? 127 : (dx < -128 ? -128 : dx));
    report.dy = (BYTE)(char)(dy > 127 ? 127 : (dy < -128 ? -128 : dy));
    report.wheel = 0;

    DWORD written;
    if (!WriteFile(hDevice, &report, sizeof(report), &written, NULL)) {
        obs_log(LOG_WARNING, "[GvInput] WriteFile failed: %lu", GetLastError());
        closeDevice();
    }
}

void GvInputMouseController::performClickDown()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
}

void GvInputMouseController::performClickUp()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

bool GvInputMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

// ================ TencInput ================

TencInputMouseController::TencInputMouseController()
    : AbstractMouseController()
    , hDevice(INVALID_HANDLE_VALUE)
    , deviceConnected(false)
    , reportSize(65)
    , reportMagic(0x0540)
{
    if (openDevice()) {
        deviceConnected = true;
        obs_log(LOG_INFO, "[TencInput] HID TencentMyAppsHidBus device opened");
    } else {
        obs_log(LOG_WARNING, "[TencInput] Failed to open Tencent device");
    }
}

TencInputMouseController::~TencInputMouseController()
{
    closeDevice();
}

bool TencInputMouseController::openDevice()
{
    // Enumerate all tencent collections, select the one with largest OutLen
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

    std::wstring bestPath;
    int bestOutLen = 0;

    // First pass: probe all collections
    HDEVINFO devInfo = SetupDiGetClassDevs(&hidGuid, NULL, NULL,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devInfo == INVALID_HANDLE_VALUE) return false;

    DWORD idx = 0;
    SP_DEVICE_INTERFACE_DATA ifaceData = { sizeof(SP_DEVICE_INTERFACE_DATA) };
    while (SetupDiEnumDeviceInterfaces(devInfo, NULL, &hidGuid, idx, &ifaceData)) {
        DWORD required = 0;
        SetupDiGetDeviceInterfaceDetail(devInfo, &ifaceData, NULL, 0, &required, NULL);
        if (required == 0) { idx++; continue; }

        PSP_DEVICE_INTERFACE_DETAIL_DATA detail =
            (PSP_DEVICE_INTERFACE_DETAIL_DATA)malloc(required);
        if (!detail) { idx++; continue; }
        detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        if (SetupDiGetDeviceInterfaceDetail(devInfo, &ifaceData, detail, required, NULL, NULL)) {
            std::wstring devPath(detail->DevicePath);
            if (devPath.find(L"tencentmyappshidbus") != std::wstring::npos) {
                HANDLE hProbe = CreateFile(detail->DevicePath, 0,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
                if (hProbe != INVALID_HANDLE_VALUE) {
                    PHIDP_PREPARSED_DATA ppd = NULL;
                    if (HidD_GetPreparsedData(hProbe, &ppd)) {
                        HIDP_CAPS caps = {};
                        if (HidP_GetCaps(ppd, &caps) == HIDP_STATUS_SUCCESS) {
                            obs_log(LOG_INFO, "[TencInput] Col%ls: Page=0x%04X Usage=0x%04X In=%d Out=%d Feat=%d",
                                devPath.substr(devPath.find(L"col"), 5).c_str(),
                                caps.UsagePage, caps.Usage,
                                caps.InputReportByteLength, caps.OutputReportByteLength,
                                caps.FeatureReportByteLength);
                            int outLen = caps.OutputReportByteLength;
                            if (outLen > bestOutLen) {
                                bestOutLen = outLen;
                                bestPath = detail->DevicePath;
                            }
                        }
                        HidD_FreePreparsedData(ppd);
                    }
                    CloseHandle(hProbe);
                }
            }
        }
        free(detail);
        idx++;
    }
    SetupDiDestroyDeviceInfoList(devInfo);

    // Open the best collection (largest OutLen)
    if (!bestPath.empty() && bestOutLen >= 65) {
        hDevice = CreateFile(bestPath.c_str(), GENERIC_WRITE | GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
        if (hDevice == INVALID_HANDLE_VALUE) {
            hDevice = CreateFile(bestPath.c_str(), GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
        }
        if (hDevice != INVALID_HANDLE_VALUE) {
            reportSize = bestOutLen;
            reportMagic = 0;
            deviceConnected = true;
            obs_log(LOG_INFO, "[TencInput] Opened: OutLen=%d", reportSize);
            return true;
        }
        obs_log(LOG_WARNING, "[TencInput] CreateFile failed for best path, err=%lu", GetLastError());
    }

    // Fallback: try findAndCreateHidDevice (tries GENERIC_WRITE first, then GENERIC_WRITE|GENERIC_READ)
    hDevice = findAndCreateHidDevice(L"tencentmyappshidbus", false, "TencInput");
    if (hDevice != INVALID_HANDLE_VALUE) {
        reportSize = 65;
        reportMagic = 0;
        deviceConnected = true;
        obs_log(LOG_INFO, "[TencInput] Fallback opened, OutLen=%d", reportSize);
        return true;
    }

    obs_log(LOG_WARNING, "[TencInput] No suitable device found");
    return false;
}

void TencInputMouseController::closeDevice()
{
    if (hDevice != INVALID_HANDLE_VALUE) {
        CloseHandle(hDevice);
        hDevice = INVALID_HANDLE_VALUE;
    }
    deviceConnected = false;
}

void TencInputMouseController::moveMouse(int dx, int dy)
{
    if (!deviceConnected || hDevice == INVALID_HANDLE_VALUE) {
        if (!openDevice()) return;
    }

    // Tencent HID sub-report protocol:
    // Byte 0: ReportID = 0x50
    // Bytes 1-2: sub_report_length (LE16) = 2 for [dx,dy] pair
    // Bytes 3-4: sub_report data = dx, dy as signed 8-bit
    // Bytes 5-64: zero padding

    BYTE report[65] = {};
    report[0] = 0x50;          // ReportID
    report[1] = 2;             // sub_report_length LE16 = 2
    report[2] = 0;
    report[3] = (BYTE)(char)(dx > 127 ? 127 : (dx < -128 ? -128 : dx));
    report[4] = (BYTE)(char)(dy > 127 ? 127 : (dy < -128 ? -128 : dy));

    DWORD written;
    if (!WriteFile(hDevice, report, reportSize, &written, NULL)) {
        obs_log(LOG_WARNING, "[TencInput] WriteFile failed: %lu", GetLastError());
        closeDevice();
    }
}

void TencInputMouseController::performClickDown()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
}

void TencInputMouseController::performClickUp()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

bool TencInputMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif