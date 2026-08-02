// 极简 winsock HTTP 服务器实现。

#include "HttpServer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace ya {
namespace web {

namespace {

std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string header_value(const std::string& req, const char* name)
{
    const std::string key = to_lower(std::string(name)) + ":";
    const std::string lower = to_lower(req);
    const auto p = lower.find(key);
    if (p == std::string::npos)
        return {};
    auto e = req.find("\r\n", p);
    if (e == std::string::npos)
        e = req.size();
    std::string v = req.substr(p + key.size(), e - p - key.size());
    // 去首尾空白
    size_t b = v.find_first_not_of(" \t");
    size_t en = v.find_last_not_of(" \t");
    if (b == std::string::npos)
        return {};
    return v.substr(b, en - b + 1);
}

size_t content_length(const std::string& req)
{
    const std::string v = header_value(req, "Content-Length");
    if (v.empty())
        return 0;
    try {
        return static_cast<size_t>(std::stoull(v));
    } catch (...) {
        return 0;
    }
}

// 解析请求行 + 头 + body
bool parse_request(SOCKET client, HttpRequest& out)
{
    std::string buf;
    char tmp[8192];
    const size_t header_max = 64 * 1024;
    while (buf.find("\r\n\r\n") == std::string::npos) {
        const int n = recv(client, tmp, sizeof(tmp), 0);
        if (n <= 0)
            return false;
        buf.append(tmp, tmp + n);
        if (buf.size() > header_max)
            return false;
    }
    const size_t hdr_end = buf.find("\r\n\r\n") + 4;

    // 请求行: METHOD PATH?QUERY HTTP/1.1
    const size_t line_end = buf.find("\r\n");
    const std::string line = buf.substr(0, line_end);
    std::istringstream ls(line);
    ls >> out.method >> out.path;
    if (out.method.empty() || out.path.empty())
        return false;
    const size_t q = out.path.find('?');
    if (q != std::string::npos) {
        out.query = out.path.substr(q + 1);
        out.path = out.path.substr(0, q);
    }

    // body
    const size_t cl = content_length(buf);
    out.body.clear();
    if (cl > 0) {
        const size_t have = buf.size() >= hdr_end ? buf.size() - hdr_end : 0;
        if (have < cl) {
            out.body = buf.substr(hdr_end);
            while (out.body.size() < cl) {
                const int n = recv(client, tmp, sizeof(tmp), 0);
                if (n <= 0)
                    return false;
                out.body.append(tmp, tmp + n);
            }
        } else {
            out.body = buf.substr(hdr_end, cl);
        }
        out.body.resize(cl);
    }
    return true;
}

std::string mime_type(const std::string& path)
{
    const std::string ext = to_lower(std::filesystem::path(path).extension().string());
    if (ext == ".html")
        return "text/html; charset=utf-8";
    if (ext == ".js")
        return "application/javascript";
    if (ext == ".css")
        return "text/css; charset=utf-8";
    if (ext == ".json")
        return "application/json";
    if (ext == ".png")
        return "image/png";
    if (ext == ".svg")
        return "image/svg+xml";
    if (ext == ".ico")
        return "image/x-icon";
    return "application/octet-stream";
}

} // namespace

void HttpServer::route(const std::string& method, const std::string& path, Handler h)
{
    routes_[method][path] = std::move(h);
}

bool HttpServer::start(int port, const std::string& static_root)
{
    if (running_.load())
        return false;
    port_ = port;
    static_root_ = static_root;
    stop_.store(false);
    running_.store(true);
    th_ = std::thread(&HttpServer::thread_main, this);
    // 等待监听就绪(最多 2s)
    for (int i = 0; i < 200 && running_.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (th_.joinable() && i > 50) {
            // thread_main 内部启动失败会置 running_=false
        }
    }
    return running_.load();
}

void HttpServer::stop()
{
    if (!running_.load())
        return;
    stop_.store(true);
#ifdef _WIN32
    // 自连打断 accept
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s != INVALID_SOCKET) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(port_));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        closesocket(s);
    }
#endif
    if (th_.joinable())
        th_.join();
    running_.store(false);
}

void HttpServer::thread_main()
{
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        running_.store(false);
        return;
    }
    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd == INVALID_SOCKET) {
        running_.store(false);
        WSACleanup();
        return;
    }
    BOOL yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&yes), sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port_));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        std::fprintf(stderr, "[web] bind 127.0.0.1:%d failed\n", port_);
        closesocket(listen_fd);
        running_.store(false);
        WSACleanup();
        return;
    }
    if (listen(listen_fd, 16) == SOCKET_ERROR) {
        closesocket(listen_fd);
        running_.store(false);
        WSACleanup();
        return;
    }
    std::fprintf(stderr, "[web] listening on http://127.0.0.1:%d\n", port_);

    while (!stop_.load()) {
        SOCKET client = accept(listen_fd, nullptr, nullptr);
        if (client == INVALID_SOCKET) {
            if (stop_.load())
                break;
            continue;
        }
        if (stop_.load()) {
            closesocket(client);
            break;
        }

        HttpRequest req;
        if (!parse_request(client, req)) {
            closesocket(client);
            continue;
        }

        HttpResponse resp;
        if (req.method == "OPTIONS") {
            // CORS preflight
            const std::string raw =
                "HTTP/1.1 204 No Content\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: GET, POST, PUT, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type, X-Filename\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
            send(client, raw.data(), static_cast<int>(raw.size()), 0);
            closesocket(client);
            continue;
        }
        resp = dispatch(req);

        std::ostringstream o;
        o << "HTTP/1.1 " << resp.status << " " << (resp.status == 200 ? "OK" : "Error")
          << "\r\n"
          << "Access-Control-Allow-Origin: *\r\n"
          << "Cache-Control: no-store\r\n"
          << "Content-Type: " << resp.content_type << "\r\n"
          << "Content-Length: " << resp.body.size() << "\r\n"
          << "Connection: close\r\n\r\n";
        const std::string head = o.str();
        send(client, head.data(), static_cast<int>(head.size()), 0);
        if (!resp.body.empty())
            send(client, resp.body.data(), static_cast<int>(resp.body.size()), 0);
        closesocket(client);
    }
    closesocket(listen_fd);
    WSACleanup();
    running_.store(false);
#endif
}

HttpResponse HttpServer::dispatch(const HttpRequest& req)
{
    try {
        const auto mi = routes_.find(req.method);
        if (mi != routes_.end()) {
            const auto hi = mi->second.find(req.path);
            if (hi != mi->second.end())
                return hi->second(req);
        }
        // 静态文件
        if (req.method == "GET" && !static_root_.empty())
            return serve_static(req.path);
        return HttpResponse::text("not found", 404);
    } catch (const std::exception& e) {
        // 防御:handler 异常 → 500,绝不带崩进程
        std::fprintf(stderr, "[web] handler exception: %s\n", e.what());
        return HttpResponse::json("{\"ok\":false,\"error\":\"internal\"}", 500);
    } catch (...) {
        std::fprintf(stderr, "[web] handler unknown exception\n");
        return HttpResponse::json("{\"ok\":false,\"error\":\"internal\"}", 500);
    }
}

HttpResponse HttpServer::serve_static(const std::string& path)
{
    std::string rel = path;
    if (rel.empty() || rel == "/")
        rel = "/index.html";
    // 防目录穿越
    if (rel.find("..") != std::string::npos)
        return HttpResponse::text("forbidden", 403);
    if (!rel.empty() && rel[0] == '/')
        rel = rel.substr(1);

    const std::filesystem::path full = std::filesystem::path(static_root_) / rel;
    std::ifstream f(full, std::ios::binary);
    if (!f.is_open())
        return HttpResponse::text("not found", 404);
    std::ostringstream ss;
    ss << f.rdbuf();
    HttpResponse r;
    r.status = 200;
    r.content_type = mime_type(full.string());
    r.body = ss.str();
    return r;
}

} // namespace web
} // namespace ya
