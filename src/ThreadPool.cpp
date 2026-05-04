#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t numThreads) {
    running_ = true;
    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back(&ThreadPool::worker, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

ThreadPool::ThreadPool(ThreadPool&& other) noexcept
    : workers_(std::move(other.workers_)),
      taskQueue_(std::move(other.taskQueue_)),
      running_(other.running_.load()) {
    other.running_ = false;
}

ThreadPool& ThreadPool::operator=(ThreadPool&& other) noexcept {
    if (this != &other) {
        shutdown();
        workers_ = std::move(other.workers_);
        taskQueue_ = std::move(other.taskQueue_);
        running_ = other.running_.load();
        other.running_ = false;
    }
    return *this;
}

void ThreadPool::shutdown() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    taskCondition_.notify_all();
    
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

void ThreadPool::worker() {
    while (running_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(taskQueueMutex_);
            taskCondition_.wait(lock, [this] { 
                return !running_ || !taskQueue_.empty(); 
            });
            
            if (!running_ && taskQueue_.empty()) {
                return;
            }
            
            if (!taskQueue_.empty()) {
                task = std::move(taskQueue_.front());
                taskQueue_.pop();
            }
        }
        
        if (task) {
            task();
        }
    }
}
