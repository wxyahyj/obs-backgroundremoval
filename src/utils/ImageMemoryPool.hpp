#pragma once

#include <vector>
#include <mutex>
#include <cstddef>
#include <cstdlib>
#include <opencv2/opencv.hpp>

/**
 * 图像内存预分配池
 * 
 * 功能：
 * - 启动时预分配多种常用尺寸的内存块
 * - 运行时零分配，减少内存碎片
 * - 32字节内存对齐，优化SIMD访问
 * - 线程安全
 */
class ImageMemoryPool {
public:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;
        int width;
        int height;
        int type;
    };

private:
    std::vector<MemoryBlock> blocks_;
    std::mutex mutex_;
    bool initialized_ = false;
    
    // 禁用拷贝
    ImageMemoryPool(const ImageMemoryPool&) = delete;
    ImageMemoryPool& operator=(const ImageMemoryPool&) = delete;

public:
    ImageMemoryPool() = default;
    ~ImageMemoryPool() {
        Release();
    }
    
    /**
     * 初始化内存池
     * @param maxWidth 最大宽度
     * @param maxHeight 最大高度
     */
    void Initialize(int maxWidth, int maxHeight) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return;
        
        // 预分配常用分辨率的内存块（BGRA格式，4通道）
        std::vector<size_t> sizes = {
            640 * 640 * 4,      // 正方形推理
            640 * 480 * 4,      // VGA
            1280 * 720 * 4,     // 720p
            1920 * 1080 * 4,    // 1080p
            static_cast<size_t>(maxWidth * maxHeight * 4)  // 最大尺寸
        };
        
        for (size_t size : sizes) {
            // 32字节对齐，适合AVX
            void* ptr = _aligned_malloc(size, 32);
            if (ptr) {
                blocks_.push_back({ptr, size, false, 0, 0, 0});
            }
        }
        
        initialized_ = true;
    }
    
    /**
     * 获取图像内存块
     * @param width 图像宽度
     * @param height 图像高度
     * @param type OpenCV类型（如CV_8UC4）
     * @return cv::Mat 包装的内存块，如果无可用块则返回空Mat
     */
    cv::Mat GetImage(int width, int height, int type) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t required = static_cast<size_t>(width * height * CV_ELEM_SIZE(type));
        
        // 首次适配算法
        for (auto& block : blocks_) {
            if (!block.inUse && block.size >= required) {
                block.inUse = true;
                block.width = width;
                block.height = height;
                block.type = type;
                return cv::Mat(height, width, type, block.ptr);
            }
        }
        
        // 无可用块，降级到普通分配
        return cv::Mat(height, width, type);
    }
    
    /**
     * 释放图像内存块
     * @param mat 要释放的Mat
     */
    void ReleaseImage(cv::Mat& mat) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.inUse && block.ptr == mat.data) {
                block.inUse = false;
                block.width = 0;
                block.height = 0;
                block.type = 0;
                mat.release();
                return;
            }
        }
        
        // 不在池中，直接释放
        mat.release();
    }
    
    /**
     * 获取预分配缓冲区（用于推理输入）
     * @param requiredSize 所需大小
     * @return 缓冲区指针
     */
    void* GetBuffer(size_t requiredSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (!block.inUse && block.size >= requiredSize) {
                block.inUse = true;
                return block.ptr;
            }
        }
        
        return nullptr;
    }
    
    /**
     * 释放缓冲区
     * @param ptr 缓冲区指针
     */
    void ReleaseBuffer(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.inUse && block.ptr == ptr) {
                block.inUse = false;
                return;
            }
        }
    }
    
    /**
     * 获取统计信息
     */
    size_t GetTotalBlocks() const { return blocks_.size(); }
    
    size_t GetUsedBlocks() const {
        size_t count = 0;
        for (const auto& block : blocks_) {
            if (block.inUse) count++;
        }
        return count;
    }
    
    size_t GetTotalMemory() const {
        size_t total = 0;
        for (const auto& block : blocks_) {
            total += block.size;
        }
        return total;
    }
    
    /**
     * 释放所有内存
     */
    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.ptr) {
                _aligned_free(block.ptr);
            }
        }
        blocks_.clear();
        initialized_ = false;
    }
    
    /**
     * 获取单例实例
     */
    static ImageMemoryPool& Instance() {
        static ImageMemoryPool instance;
        return instance;
    }
};

// 全局访问宏
#define g_imagePool ImageMemoryPool::Instance()
