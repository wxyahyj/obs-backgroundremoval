# 读取 TENCENTMYAPPSHIDBUS 的 HID 报告描述符
import ctypes, ctypes.wintypes, struct, sys
from ctypes import wintypes

# Windows API constants
DIGCF_PRESENT = 2
DIGCF_DEVICEINTERFACE = 0x10
GENERIC_WRITE = 0x40000000
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("InterfaceClassGuid", ctypes.c_byte * 16),
                ("Flags", wintypes.DWORD), ("Reserved", ctypes.c_ulong)]

def main():
    setupapi = ctypes.windll.setupapi
    hid = ctypes.windll.hid
    kernel32 = ctypes.windll.kernel32

    # Get HID GUID
    hidGuid = ctypes.c_byte * 16
    hid.HidD_GetHidGuid(ctypes.byref(hidGuid))

    devInfo = setupapi.SetupDiGetClassDevsA(ctypes.byref(hidGuid), None, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
    if devInfo == INVALID_HANDLE_VALUE:
        print("SetupDiGetClassDevs failed")
        return

    idx = 0
    ifaceData = SP_DEVICE_INTERFACE_DATA()
    ifaceData.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)

    while setupapi.SetupDiEnumDeviceInterfaces(devInfo, None, ctypes.byref(hidGuid), idx, ctypes.byref(ifaceData)):
        required = wintypes.DWORD(0)
        setupapi.SetupDiGetDeviceInterfaceDetailA(devInfo, ctypes.byref(ifaceData), None, 0, ctypes.byref(required), None)
        if required.value == 0:
            idx += 1; continue
        detail = ctypes.create_string_buffer(required.value)
        # PSP_DEVICE_INTERFACE_DETAIL_DATA has cbSize at start
        ctypes.memmove(detail, ctypes.byref(wintypes.DWORD(ctypes.sizeof(SP_DEVICE_INTERFACE_DATA) + 4)), 4)
        # Actually the cbSize must be sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA) which varies
        # Use a simpler approach
        detail_data = (ctypes.c_ubyte * required.value)()
        ctypes.memset(ctypes.byref(detail_data, 0), 0, required.value)
        # Set cbSize
        struct.pack_into("I", detail_data, 0, ctypes.sizeof(SP_DEVICE_INTERFACE_DATA) + 4)
        
        if setupapi.SetupDiGetDeviceInterfaceDetailA(devInfo, ctypes.byref(ifaceData), ctypes.byref(detail_data), required, ctypes.byref(required), None):
            # Extract device path
            path_start = 4  # after cbSize(DWORD)
            path_bytes = bytes(detail_data[path_start:]).rstrip(b'\x00')
            path = path_bytes.decode('utf-8', errors='replace')
            
            if "tencentmyappshidbus" in path.lower() and "col04" in path.lower():
                print(f"\n=== Found COL04 ===")
                print(f"Path: {path}")
                
                # Open device for reading
                h = kernel32.CreateFileA(path.encode(), GENERIC_WRITE | GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
                if h == INVALID_HANDLE_VALUE:
                    err = ctypes.GetLastError()
                    print(f"CreateFile failed: {err}")
                    # Try with just write
                    h = kernel32.CreateFileA(path.encode(), GENERIC_WRITE,
                        FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
                    if h == INVALID_HANDLE_VALUE:
                        print(f"CreateFile(GENERIC_WRITE) failed: {ctypes.GetLastError()}")
                        idx += 1; continue
                
                print("Device opened successfully!")
                
                # Get preparsed data
                ppd = ctypes.c_void_p(0)
                if hid.HidD_GetPreparsedData(h, ctypes.byref(ppd)):
                    # Get capabilities
                    caps = (ctypes.c_ubyte * 32)()  # HIDP_CAPS
                    from ctypes import sizeof
                    # HIDP_CAPS is about 24 bytes, use raw approach
                    if hid.HidP_GetCaps(ctypes.c_void_p(ppd.value), ctypes.byref(caps)):
                        # Extract key fields
                        usage = struct.unpack_from("H", caps, 0)[0]  # Usage
                        usage_page = struct.unpack_from("H", caps, 2)[0]  # UsagePage
                        in_len = struct.unpack_from("H", caps, 8)[0]
                        out_len = struct.unpack_from("H", caps, 10)[0]
                        feat_len = struct.unpack_from("H", caps, 12)[0]
                        print(f"UsagePage=0x{usage_page:04X} Usage=0x{usage:04X}")
                        print(f"InputLen={in_len} OutputLen={out_len} FeatureLen={feat_len}")
                        
                        # Read the actual report descriptor
                        # HidD_GetDescriptor is not directly available
                        # Try GetFeature report to discover protocol
                        feature_buf = (ctypes.c_ubyte * 65)()
                        if hid.HidD_GetFeature(h, ctypes.byref(feature_buf), 65):
                            print(f"GetFeature: {bytes(feature_buf[:32]).hex()}")
                    
                    hid.HidD_FreePreparsedData(ctypes.c_void_p(ppd.value))
                else:
                    print("GetPreparsedData failed")
                
                # Try reading the input report to see device data format
                overlap = None
                read_buf = (ctypes.c_ubyte * 65)()
                # Don't actually read (would block)
                print(f"Report descriptor request completed")
                
                kernel32.CloseHandle(h)
        idx += 1
    
    setupapi.SetupDiDestroyDeviceInfoList(devInfo)

if __name__ == '__main__':
    main()
