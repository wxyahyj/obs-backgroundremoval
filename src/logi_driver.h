/**
 * logi_driver.h - 多驱动鼠标模拟（罗技G HUB / LGS / 雷蛇 Synapse）
 *
 * 从独立DLL适配为插件内部静态链接模块。
 * 支持：
 *   - Logitech G HUB   (IOCTL 0x2a2010, 7字节报告, 16位增量)
 *   - Logitech LGS      (IOCTL 0x2a2010, 5字节报告, 8位增量)
 *   - Razer Synapse 3    (IOCTL 0x88883020, 32字节报告, RZCONTROL设备)
 *   - Razer Synapse 2    (rzcontrol.sys, 相同IOCTL)
 *
 * 要求：
 *   1. 至少安装并运行一种支持的驱动软件
 *   2. 以管理员身份运行
 *   3. 不需要物理外设
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化驱动模块（初始化互斥锁等），在首次使用前调用 */
void logi_driver_init(void);

/* 清理驱动模块（关闭设备、销毁互斥锁），在不再使用时调用 */
void logi_driver_cleanup(void);

/**
 * 打开驱动总线枚举设备（自动检测GHUB/LGS/Razer）。
 * @return 1成功, 0失败。
 */
int device_open(void);

/**
 * 指定驱动类型打开设备。
 * @param type  0=自动检测, 1=强制GHUB, 2=强制LGS, 3=强制Razer
 * @return 1成功, 0失败。
 */
int device_open2(int type);

/**
 * 获取当前已激活的驱动类型。
 * @return 0=无, 1=LGS, 2=GHUB, 3=Razer
 */
int get_driver_type(void);

/**
 * 关闭设备句柄。
 */
void device_close(void);

/**
 * 相对移动鼠标。
 * @param x  水平增量（正=右, 负=左）
 * @param y  垂直增量（正=下, 负=上）
 * @return 1成功, 0失败。
 */
int moveR(int x, int y);

/**
 * 按下鼠标按键。
 * @param button  1=左键, 2=右键, 3=中键
 * @return 1成功, 0失败。
 */
int mouse_down(int button);

/**
 * 释放鼠标按键。
 * @param button  1=左键, 2=右键, 3=中键
 * @return 1成功, 0失败。
 */
int mouse_up(int button);

#ifdef __cplusplus
}
#endif
