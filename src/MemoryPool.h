#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <opencv2/core.hpp>
#include "models/Detection.h"

template<typename T>
class MemoryPool {
public:
    explicit MemoryPool(size_t maxSize = 10) : maxSize_(maxSize) {}
    
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) noexcept = default;
    MemoryPool& operator=(MemoryPool&&) noexcept = default;

    T acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            T buffer = std::move(pool_.back());
            pool_.pop_back();
            return buffer;
        }
        return T();
    }
    
    void release(T&& buffer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.size() < maxSize_) {
            pool_.push_back(std::move(buffer));
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

private:
    std::vector<T> pool_;
    mutable std::mutex mutex_;
    size_t maxSize_;
};

class ImageMemoryPool {
public:
    explicit ImageMemoryPool(size_t maxSize = 10) : maxSize_(maxSize) {}
    
    ImageMemoryPool(const ImageMemoryPool&) = delete;
    ImageMemoryPool& operator=(const ImageMemoryPool&) = delete;
    ImageMemoryPool(ImageMemoryPool&&) noexcept = default;
    ImageMemoryPool& operator=(ImageMemoryPool&&) noexcept = default;

    struct ImageBufferKey {
        int rows;
        int cols;
        int type;
        
        bool operator==(const ImageBufferKey& other) const {
            return rows == other.rows && cols == other.cols && type == other.type;
        }
    };
    
    struct ImageBufferKeyHash {
        size_t operator()(const ImageBufferKey& key) const {
            size_t h1 = std::hash<int>()(key.rows);
            size_t h2 = std::hash<int>()(key.cols);
            size_t h3 = std::hash<int>()(key.type);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    
    cv::Mat acquire(int rows, int cols, int type) {
        std::lock_guard<std::mutex> lock(mutex_);
        ImageBufferKey key{rows, cols, type};
        
        auto it = pool_.find(key);
        if (it != pool_.end() && !it->second.empty()) {
            cv::Mat buffer = std::move(it->second.back());
            it->second.pop_back();
            return buffer;
        }
        
        return cv::Mat(rows, cols, type);
    }
    
    void release(cv::Mat&& buffer) {
        if (buffer.empty()) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        ImageBufferKey key{buffer.rows, buffer.cols, buffer.type()};
        
        auto it = pool_.find(key);
        if (it != pool_.end()) {
            if (it->second.size() < maxSize_) {
                it->second.push_back(std::move(buffer));
            }
        } else {
            std::vector<cv::Mat> buffers;
            buffers.reserve(5);
            buffers.push_back(std::move(buffer));
            pool_[key] = std::move(buffers);
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }
    
    size_t getTotalBufferCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& pair : pool_) {
            count += pair.second.size();
        }
        return count;
    }

private:
    std::unordered_map<ImageBufferKey, std::vector<cv::Mat>, ImageBufferKeyHash> pool_;
    mutable std::mutex mutex_;
    size_t maxSize_;
};

using DetectionMemoryPool = MemoryPool<std::vector<Detection>>;
