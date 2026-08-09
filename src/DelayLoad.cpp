#define NOMINMAX
#include <windows.h>
#include <delayimp.h>

#include <filesystem>
#include <string>

#include <obs-module.h>
#include "plugin-support.h"

extern "C" {

// FFmpeg DLL (UDP 直收通道软解). 同样优先插件同目录, 否则 OBS bin/64bit 自带的 avcodec-61 等。
static bool isDelayLoadDll(const std::string &dllName)
{
	return dllName == "onnxruntime.dll" ||
	       dllName == "avcodec-61.dll" || dllName == "avformat-61.dll" ||
	       dllName == "avutil-59.dll" || dllName == "swscale-8.dll";
}

FARPROC WINAPI DelayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli)
{
	if (dliNotify == dliNotePreLoadLibrary) {
		const std::string dllName(pdli->szDll);
		if (isDelayLoadDll(dllName)) {
			// Prefer same directory as this plugin DLL (obs-plugins/64bit),
			// not a nested PLUGIN_NAME subfolder (often missing).
			HMODULE self = nullptr;
			GetModuleHandleExW(
				GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				reinterpret_cast<LPCWSTR>(&DelayLoadHook), &self);
			wchar_t modPath[MAX_PATH] = {};
			if (self && GetModuleFileNameW(self, modPath, MAX_PATH) > 0) {
				std::filesystem::path dir = std::filesystem::path(modPath).parent_path();
				std::filesystem::path absPath = dir / dllName;
				if (std::filesystem::exists(absPath)) {
					obs_log(LOG_INFO, "Loading %s from %S", dllName.c_str(), absPath.c_str());
					return (FARPROC)LoadLibraryExW(absPath.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
				}
			}
			// Fallback: default loader search (finds OBS bin/64bit DLLs)
			return NULL;
		} else {
			return NULL;
		}
	}
	return NULL;
}

const PfnDliHook __pfnDliNotifyHook2 = DelayLoadHook;
}
