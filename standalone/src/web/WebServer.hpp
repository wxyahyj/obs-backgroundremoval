#pragma once

#include "EngineLoop.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ya {

class WebServer {
public:
    WebServer(EngineLoop& engine, std::string web_root);
    ~WebServer();

    bool start(const std::string& host, int port);
    void stop();
    bool running() const { return running_.load(); }
    int port() const { return port_; }

private:
    void thread_main(std::string host, int port);

    EngineLoop& engine_;
    std::string web_root_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
    std::thread th_;
    int port_ = 0;
};

} // namespace ya
