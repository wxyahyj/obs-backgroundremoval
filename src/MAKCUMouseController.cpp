#ifdef _WIN32

#include "MAKCUMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>
#include <setupapi.h>
#include <initguid.h>
#include <devguid.h>
#include <regstr.h>
#include <cstdio>
#include <cstring>

#pragma comment(lib, "setupapi.lib")

MAKCUMouseController::MAKCUMouseController()
    : AbstractMouseController()
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName("COM3")
    , baudRate(115200)
{
}

MAKCUMouseController::MAKCUMouseController(const std::string& port, int baud)
    : AbstractMouseController()
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName(port)
    , baudRate(baud)
{
    connectSerial();
}

MAKCUMouseController::~MAKCUMouseController()
{
    disconnectSerial();
}

bool MAKCUMouseController::connectSerial()
{
    if (serialConnected) {
        return true;
    }

    std::string fullPortName = "\\\\.\\" + portName;
    
    hSerial = CreateFileA(
        fullPortName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hSerial == INVALID_HANDLE_VALUE) {
        obs_log(LOG_WARNING, "[MAKCU] Failed to open serial port %s", portName.c_str());
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

    if (!GetCommState(hSerial, &dcbSerialParams)) {
        obs_log(LOG_ERROR, "[MAKCU] Failed to get serial port state");
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    dcbSerialParams.BaudRate = baudRate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    dcbSerialParams.fDtrControl = DTR_CONTROL_ENABLE;

    if (!SetCommState(hSerial, &dcbSerialParams)) {
        obs_log(LOG_ERROR, "[MAKCU] Failed to set serial port state");
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;

    if (!SetCommTimeouts(hSerial, &timeouts)) {
        obs_log(LOG_ERROR, "[MAKCU] Failed to set serial port timeouts");
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    serialConnected = true;
    obs_log(LOG_INFO, "[MAKCU] Connected to serial port %s at %d baud", portName.c_str(), baudRate);
    
    return true;
}

void MAKCUMouseController::disconnectSerial()
{
    if (hSerial != INVALID_HANDLE_VALUE) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
    }
    serialConnected = false;
    obs_log(LOG_INFO, "[MAKCU] Disconnected from serial port");
}

bool MAKCUMouseController::sendSerialCommand(const std::string& command)
{
    if (!serialConnected) {
        if (!connectSerial()) {
            return false;
        }
    }

    std::string cmd = command + "\n";
    DWORD bytesWritten;
    
    if (!WriteFile(hSerial, cmd.c_str(), static_cast<DWORD>(cmd.length()), &bytesWritten, NULL)) {
        obs_log(LOG_ERROR, "[MAKCU] Failed to write to serial port");
        serialConnected = false;
        return false;
    }

    return true;
}

void MAKCUMouseController::move(int dx, int dy)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.move(%d,%d)", dx, dy);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::moveTo(int x, int y)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.moveTo(%d,%d)", x, y);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::clickDown()
{
    sendSerialCommand("km.left(1)");
}

void MAKCUMouseController::clickUp()
{
    sendSerialCommand("km.left(0)");
}

void MAKCUMouseController::moveMouse(int dx, int dy)
{
    if (serialConnected) {
        move(dx, dy);
    } else {
        connectSerial();
    }
}

void MAKCUMouseController::performClickDown()
{
    clickDown();
}

void MAKCUMouseController::performClickUp()
{
    clickUp();
}

bool MAKCUMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

bool MAKCUMouseController::testCommunication()
{
    if (!serialConnected) {
        if (!connectSerial()) {
            return false;
        }
    }
    
    return sendSerialCommand("km.version()");
}

#endif
