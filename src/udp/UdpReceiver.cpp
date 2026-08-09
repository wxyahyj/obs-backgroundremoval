#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>

#include "UdpReceiver.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <obs-module.h>
#include "plugin-support.h"

// ============================================================================
// FFmpeg DLL 可用性探测 (DelayLoad 保护: DLL 缺失时 UDP 功能禁用, 插件照常)
// ============================================================================
static bool ffmpegDllsAvailable(std::string &missing)
{
	static const char *dlls[] = {
		"avcodec-61.dll",
		"avformat-61.dll",
		"avutil-59.dll",
		"swscale-8.dll",
	};
	for (const char *dll : dlls) {
		if (GetModuleHandleA(dll)) {
			continue; // 已被 OBS (obs-ffmpeg 插件) 加载
		}
		HMODULE h = LoadLibraryA(dll);
		if (!h) {
			missing += dll;
			missing += " ";
			return false;
		}
		FreeLibrary(h);
	}
	return true;
}

UdpReceiver::~UdpReceiver()
{
	stop();
}

bool UdpReceiver::start(int port, FrameCallback callback)
{
	stop();

	std::string missing;
	if (!ffmpegDllsAvailable(missing)) {
		std::lock_guard<std::mutex> lock(statsMutex_);
		lastError_ = "FFmpeg DLL 不可用 (缺 " + missing + "), 需 OBS 31+";
		obs_log(LOG_ERROR, "[UDP Receive] %s", lastError_.c_str());
		return false;
	}

	callback_ = std::move(callback);
	running_.store(true, std::memory_order_release);
	thread_ = std::thread(&UdpReceiver::threadFunc, this, port);
	return true;
}

void UdpReceiver::stop()
{
	running_.store(false, std::memory_order_release);
	if (thread_.joinable()) {
		thread_.join();
	}
	callback_ = nullptr;
}

bool UdpReceiver::isRunning() const
{
	return running_.load(std::memory_order_acquire) && thread_.joinable();
}

double UdpReceiver::fps() const
{
	std::lock_guard<std::mutex> lock(statsMutex_);
	return fps_;
}

int UdpReceiver::frameWidth() const
{
	std::lock_guard<std::mutex> lock(statsMutex_);
	return width_;
}

int UdpReceiver::frameHeight() const
{
	std::lock_guard<std::mutex> lock(statsMutex_);
	return height_;
}

uint64_t UdpReceiver::totalFrames() const
{
	std::lock_guard<std::mutex> lock(statsMutex_);
	return totalFrames_;
}

uint64_t UdpReceiver::decodedFrames() const
{
	std::lock_guard<std::mutex> lock(statsMutex_);
	return decodedFrames_;
}

const std::string &UdpReceiver::lastError() const
{
	return lastError_;
}

// ============================================================================
// 接收 + 解码主循环
// ============================================================================
void UdpReceiver::threadFunc(int port)
{
#ifdef _WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
#endif

	const std::string url = "udp://0.0.0.0:" + std::to_string(port);
	obs_log(LOG_INFO, "[UDP Receive] listening on %s", url.c_str());

	auto frameStart = std::chrono::steady_clock::now();
	uint64_t frameCount = 0;

	// 连接质量: 无更多数据超时后尝试重连
	constexpr int kReconnectAfterIdleMs = 5000;
	std::chrono::steady_clock::time_point lastPacketTime = std::chrono::steady_clock::now();

	while (running_.load(std::memory_order_acquire)) {
		AVDictionary *options = nullptr;
		av_dict_set(&options, "reuse", "1", 0);
		av_dict_set(&options, "fifo_size", "400000", 0);   // 内部缓冲 (字节), 调小降延迟
		av_dict_set(&options, "buffer_size", "2000000", 0); // socket 缓冲
		av_dict_set(&options, "timeout", "2000000", 0);     // 无数据 2s 视为超时 (微秒)

		AVFormatContext *fmtCtx = nullptr;
		if (avformat_open_input(&fmtCtx, url.c_str(), nullptr, &options) != 0) {
			{
				std::lock_guard<std::mutex> lock(statsMutex_);
				lastError_ = "无法打开 udp:// 端口 " + std::to_string(port) + " (被占用?)";
			}
			obs_log(LOG_ERROR, "[UDP Receive] %s", lastError_.c_str());
			break;
		}
		av_dict_free(&options);

		// 低延迟: 关闭 demuxer 内部缓冲, 逐包立即输出
		fmtCtx->flags |= AVFMT_FLAG_NOBUFFER | AVFMT_FLAG_FLUSH_PACKETS;

		if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
			obs_log(LOG_ERROR, "[UDP Receive] avformat_find_stream_info 失败");
			avformat_close_input(&fmtCtx);
			break;
		}

		int videoIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
		if (videoIdx < 0) {
			obs_log(LOG_ERROR, "[UDP Receive] 流中无视频轨");
			avformat_close_input(&fmtCtx);
			break;
		}

		AVCodecParameters *params = fmtCtx->streams[videoIdx]->codecpar;
		const AVCodec *codec = avcodec_find_decoder(params->codec_id);
		if (!codec) {
			obs_log(LOG_ERROR, "[UDP Receive] 无解码器 (codec_id=%d)", params->codec_id);
			avformat_close_input(&fmtCtx);
			break;
		}

		AVCodecContext *decCtx = avcodec_alloc_context3(codec);
		if (!decCtx || avcodec_parameters_to_context(decCtx, params) < 0) {
			obs_log(LOG_ERROR, "[UDP Receive] 解码器上下文创建失败");
			if (decCtx) avcodec_free_context(&decCtx);
			avformat_close_input(&fmtCtx);
			break;
		}
		decCtx->thread_count = 2; // 软解双线程
		AVRational fr = fmtCtx->streams[videoIdx]->avg_frame_rate;
		obs_log(LOG_INFO, "[UDP Receive] decoder=%s, %dx%d, %.2ffps",
			codec->name, decCtx->width, decCtx->height,
			fr.num && fr.den ? (double)fr.num / fr.den : 0.0);
		if (avcodec_open2(decCtx, codec, nullptr) < 0) {
			obs_log(LOG_ERROR, "[UDP Receive] avcodec_open2 失败");
			avcodec_free_context(&decCtx);
			avformat_close_input(&fmtCtx);
			break;
		}

		AVFrame *frame = av_frame_alloc();
		AVPacket *packet = av_packet_alloc();
		SwsContext *swsCtx = nullptr;
		int swsW = 0, swsH = 0;
		enum AVPixelFormat swsFmt = AV_PIX_FMT_NONE;

		bool connected = true;
		while (running_.load(std::memory_order_acquire)) {
			int ret = av_read_frame(fmtCtx, packet);
			if (ret < 0) {
				av_packet_unref(packet);
				if (ret == AVERROR(EAGAIN)) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					continue;
				}
				// EOF/超时: 发送端可能停止或丢流, 等待后重连
				auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::steady_clock::now() - lastPacketTime).count();
				if (idle >= kReconnectAfterIdleMs) {
					obs_log(LOG_WARNING, "[UDP Receive] 流空闲 %lldms, 重连", (long long)idle);
					connected = false;
					break;
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				continue;
			}

			lastPacketTime = std::chrono::steady_clock::now();
			{
				std::lock_guard<std::mutex> lock(statsMutex_);
				totalFrames_++;
			}

			if (packet->stream_index == videoIdx) {
				if (avcodec_send_packet(decCtx, packet) == 0) {
					while (true) {
						ret = avcodec_receive_frame(decCtx, frame);
						if (ret < 0) {
							break;
						}
						// sws 转 BGRA (与推理管线 CV_8UC4 一致)
						if (!swsCtx || frame->width != swsW || frame->height != swsH ||
						    frame->format != swsFmt) {
							if (swsCtx) sws_freeContext(swsCtx);
							swsW = frame->width;
							swsH = frame->height;
							swsFmt = (enum AVPixelFormat)frame->format;
							swsCtx = sws_getContext(swsW, swsH, swsFmt,
									 swsW, swsH, AV_PIX_FMT_BGRA,
									 SWS_BILINEAR, nullptr, nullptr, nullptr);
							{
								std::lock_guard<std::mutex> lock(statsMutex_);
								width_ = swsW;
								height_ = swsH;
							}
							obs_log(LOG_INFO, "[UDP Receive] 解码尺寸 %dx%d fmt=%d",
								swsW, swsH, (int)swsFmt);
						}
						if (swsCtx) {
							cv::Mat bgra(swsH, swsW, CV_8UC4);
							uint8_t *dst[4] = {bgra.data};
							int dstStride[4] = {(int)bgra.step};
							sws_scale(swsCtx, frame->data, frame->linesize,
								  0, swsH, dst, dstStride);

							frameCount++;
							{
								std::lock_guard<std::mutex> lock(statsMutex_);
								decodedFrames_++;
							}
							if (callback_) {
								callback_(bgra);
							}
						}
						av_frame_unref(frame);
					}
				}
			}
			av_packet_unref(packet);

			// FPS 统计 (1s 窗口)
			auto now = std::chrono::steady_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - frameStart).count();
			if (elapsed >= 1000) {
				std::lock_guard<std::mutex> lock(statsMutex_);
				fps_ = (double)frameCount * 1000.0 / (double)elapsed;
				frameCount = 0;
				frameStart = now;
			}
		}

		if (swsCtx) sws_freeContext(swsCtx);
		av_frame_free(&frame);
		av_packet_free(&packet);
		avcodec_free_context(&decCtx);
		avformat_close_input(&fmtCtx);

		if (!connected) {
			// 重连前短暂等待, 避免发送端重启时疯狂重试
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}
	}

	running_.store(false, std::memory_order_release);
	obs_log(LOG_INFO, "[UDP Receive] thread exit");
}