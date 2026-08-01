// RemoteSender 测试工具
// 编译: cl /EHsc /std:c++17 test_sender.cpp RemoteSender.cpp
// 使用: test_sender.exe 192.168.1.100 9999
// 发一张测试图到手机，验证连接

#include "RemoteSender.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: test_sender.exe 手机IP 端口\n");
        printf("示例: test_sender.exe 192.168.1.100 9999\n");
        return 1;
    }

    ya::RemoteSender sender;
    printf("连接 %s:%s ...\n", argv[1], argv[2]);
    if (!sender.connect(argv[1], atoi(argv[2]))) {
        printf("连接失败!\n");
        return 1;
    }
    printf("已连接！发送测试帧...\n");

    auto r = sender.capture_and_send(320, 320, 0, 0, 60);
    if (r.ok) {
        printf("发送成功！手机端预览框应显示测试图案。\n");
        printf("推理耗时: %.1fms\n", r.infer_ms);
    } else {
        printf("发送失败: %s\n", r.error.c_str());
    }

    sender.disconnect();
    return 0;
}
