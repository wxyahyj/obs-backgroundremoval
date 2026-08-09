/**
 * anticeat.h - 反作弊环境检测 (文档 5.6 方案)
 *
 * 顶级反作弊 (EAC/BattlEye/Vanguard/ACE/GameGuard/XTrap/TenProtect/5E/FaceIt)
 * 会对直接 syscall、IOCTL 流量做运行时特征扫描。检测到顶级反作弊时:
 *   - MouseControllerFactory 降级 LogiDriver -> MAKCU/GvInput/WindowsAPI
 *   - Syscall::init 跳过 ntdll stub 恢复 (不改写 ntdll 内存, 防完整性校验)
 *
 * 进程名全部 FNV-1a 哈希比对, 不落明文 (与 hide 模块风格一致)。
 */

#pragma once

#ifdef __cplusplus
namespace AntiCheat {

/* 检测是否运行在顶级反作弊环境。进程枚举, 成本低, 可在工厂创建时调用。 */
bool detectTop(void);

} // namespace AntiCheat
#endif
