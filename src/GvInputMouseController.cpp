#ifdef _WIN32

#include "GvInputMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <string>

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
                    GENERIC_WRITE | GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL, OPEN_EXISTING, 0, NULL);
                if (h != INVALID_HANDLE_VALUE) {
                    free(detail);
                    SetupDiDestroyDeviceInfoList(devInfo);
                    return h;
                } else {
                    obs_log(LOG_WARNING, "[%s] CreateFile failed, error: %lu", logPrefix, GetLastError());
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
    // Probe all tencent collections for HID caps, then open first writable
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

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
                // Probe this collection without opening for write
                HANDLE hProbe = CreateFile(detail->DevicePath, 0,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
                if (hProbe != INVALID_HANDLE_VALUE) {
                    PHIDP_PREPARSED_DATA ppd = NULL;
                    if (HidD_GetPreparsedData(hProbe, &ppd)) {
                        HIDP_CAPS caps = {};
                        if (HidP_GetCaps(ppd, &caps) == HIDP_STATUS_SUCCESS) {
                            obs_log(LOG_INFO, "[TencInput] Col%ls: UsagePage=0x%04X Usage=0x%04X In=%d Out=%d Feat=%d",
                                devPath.substr(devPath.find(L"col"), 5).c_str(),
                                caps.UsagePage, caps.Usage,
                                caps.InputReportByteLength, caps.OutputReportByteLength,
                                caps.FeatureReportByteLength);
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

    // Now open the first tencent device with write access (same as before)
    hDevice = findAndCreateHidDevice(L"tencentmyappshidbus", false, "TencInput");
    if (hDevice != INVALID_HANDLE_VALUE) {
        PHIDP_PREPARSED_DATA ppd = NULL;
        if (HidD_GetPreparsedData(hDevice, &ppd)) {
            HIDP_CAPS caps = {};
            if (HidP_GetCaps(ppd, &caps) == HIDP_STATUS_SUCCESS) {
                reportSize = caps.OutputReportByteLength;
                reportMagic = caps.UsagePage == 1 && caps.Usage == 2 ? 0 : 0x0540;
                obs_log(LOG_INFO, "[TencInput] Opened: UsagePage=0x%04X Usage=0x%04X OutLen=%d",
                    caps.UsagePage, caps.Usage, reportSize);
            }
            HidD_FreePreparsedData(ppd);
        }
        return true;
    }
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

    GvInputHidReport report = {};
    report.magic  = reportMagic;
    report.length = 4;
    report.flags  = 0;
    report.dx     = (BYTE)(char)(dx > 127 ? 127 : (dx < -128 ? -128 : dx));
    report.dy     = (BYTE)(char)(dy > 127 ? 127 : (dy < -128 ? -128 : dy));
    report.wheel  = 0;

    DWORD written;
    if (!WriteFile(hDevice, &report, sizeof(report), &written, NULL)) {
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