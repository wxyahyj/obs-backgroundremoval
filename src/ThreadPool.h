#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads = 4);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) noexcept;
    ThreadPool& operator=(ThreadPool&&) noexcept;

    template<typename F>
    void submit(F&& task) {
        {
            std::unique_lock<std::mutex> lock(taskQueueMutex_);
            taskQueue_.push(std::function<void()>(std::forward<F>(task)));
        }
        taskCondition_.notify_one();
    }

    void shutdown();
    bool isRunning() const { return running_; }
    size_t getThreadCount() const { return workers_.size(); }

private:
    void worker();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> taskQueue_;
    std::mutex taskQueueMutex_;
    std::condition_variable taskCondition_;
    std::atomic<bool> running_{false};
};
