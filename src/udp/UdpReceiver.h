#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

/**
 * UDP MPEG-TS (H.264) 接收 + 软解线程。
 *
 * 用途: 接收端 OBS 用 FFmpeg 自定义输出推的 udp:// MPEG-TS 流,
 * 在插件内直接解码成 BGRA cv::Mat, 经回调注入推理四缓冲队列,
 * 完全绕过 OBS 渲染管线, 降低画面到达检测器的延迟。
 *
 * FFmpeg DLL (avcodec-61/avformat-61/avutil-59/swscale-8) 由 OBS 自带,
 * 通过 DelayLoad 链接; 缺失时 start() 返回 false, 不影响插件其余功能。
 */
class UdpReceiver {
public:
	typedef std::function<void(const cv::Mat &bgraFrame)> FrameCallback;

	UdpReceiver() = default;
	~UdpReceiver();

	UdpReceiver(const UdpReceiver &) = delete;
	UdpReceiver &operator=(const UdpReceiver &) = delete;

	/** 启动接收线程。FFmpeg DLL 缺失/端口绑定失败返回 false, 详情见 lastError() */
	bool start(int port, FrameCallback callback);
	void stop();
	bool isRunning() const;

	// 统计
	double fps() const;
	int frameWidth() const;
	int frameHeight() const;
	uint64_t totalFrames() const;   // av_read_frame 读到的包数
	uint64_t decodedFrames() const; // 解码出的视频帧数
	const std::string &lastError() const;

private:
	void threadFunc(int port);

	std::thread thread_;
	std::atomic<bool> running_{false};
	FrameCallback callback_;

	mutable std::mutex statsMutex_;
	std::string lastError_;
	double fps_ = 0.0;
	int width_ = 0;
	int height_ = 0;
	uint64_t totalFrames_ = 0;
	uint64_t decodedFrames_ = 0;
};
