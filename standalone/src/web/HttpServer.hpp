#pragma once

// 极简 winsock HTTP 服务器(M4 重写)— 纯框架层,无业务。
// 支持:GET/PUT/POST/OPTIONS,精确路由,静态文件服务,127.0.0.1 绑定。

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>

namespace ya {
namespace web {

struct HttpRequest {
    std::string method; // GET / PUT / POST / OPTIONS
    std::string path;   // 不含 query
    std::string query;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;

    static HttpResponse json(const std::string& body_json, int status = 200)
    {
        HttpResponse r;
        r.status = status;
        r.content_type = "application/json; charset=utf-8";
        r.body = body_json;
        return r;
    }
    static HttpResponse text(const std::string& body, int status = 200)
    {
        HttpResponse r;
        r.status = status;
        r.content_type = "text/plain; charset=utf-8";
        r.body = body;
        return r;
    }
};

class HttpServer {
public:
    using Handler = std::function<HttpResponse(const HttpRequest&)>;

    HttpServer() = default;
    ~HttpServer() { stop(); }

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // 绑定 127.0.0.1:port;static_root 非空时提供静态文件服务。
    // 返回 false 表示端口占用/初始化失败。
    bool start(int port, const std::string& static_root);
    void stop();
    bool running() const { return running_.load(); }
    int port() const { return port_; }

    // 精确路由
    void route(const std::string& method, const std::string& path, Handler h);

private:
    void thread_main();
    HttpResponse dispatch(const HttpRequest& req);
    HttpResponse serve_static(const std::string& path);

    std::map<std::string, std::map<std::string, Handler>> routes_; // method → path → handler
    std::string static_root_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    int port_ = 17890;
    std::thread th_;
};

} // namespace web
} // namespace ya
