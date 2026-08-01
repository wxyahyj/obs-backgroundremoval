#include "WebServer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
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
namespace {

std::string read_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string header_value(const std::string& req, const char* name) {
    // case-insensitive header lookup
    std::string lower = req;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::string key = std::string(name);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    key += ":";
    auto p = lower.find(key);
    if (p == std::string::npos) return {};
    p = req.find(':', p);
    if (p == std::string::npos) return {};
    ++p;
    while (p < req.size() && (req[p] == ' ' || req[p] == '\t')) ++p;
    auto e = req.find("\r\n", p);
    if (e == std::string::npos) e = req.size();
    return req.substr(p, e - p);
}

size_t content_length(const std::string& req) {
    auto v = header_value(req, "Content-Length");
    if (v.empty()) return 0;
    try {
        return static_cast<size_t>(std::stoull(v));
    } catch (...) {
        return 0;
    }
}

// Read remaining body after headers into raw bytes (for large uploads).
bool recv_full_body(SOCKET client, std::string& headers, std::vector<uint8_t>& body,
                    size_t max_body = 256ull * 1024ull * 1024ull) {
    auto hdr_end = headers.find("\r\n\r\n");
    if (hdr_end == std::string::npos) {
        // keep reading until headers complete
        char buf[8192];
        while (headers.find("\r\n\r\n") == std::string::npos) {
            const int n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) return false;
            headers.append(buf, buf + n);
            if (headers.size() > 1024 * 1024) return false;
        }
        hdr_end = headers.find("\r\n\r\n");
    }
    const size_t cl = content_length(headers);
    if (cl > max_body) return false;
    body.clear();
    // bytes already after headers
    const size_t body_start = hdr_end + 4;
    if (body_start < headers.size()) {
        body.insert(body.end(), headers.begin() + static_cast<std::ptrdiff_t>(body_start),
                    headers.end());
        headers.resize(body_start); // keep headers only
    }
    char buf[65536];
    while (body.size() < cl) {
        const size_t need = cl - body.size();
        const int n = recv(client, buf, static_cast<int>(std::min(need, sizeof(buf))), 0);
        if (n <= 0) return false;
        body.insert(body.end(), buf, buf + n);
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') {
            o.push_back('\\');
            o.push_back(c);
        } else if (c == '\n') {
            o += "\\n";
        } else if (c == '\r') {
            o += "\\r";
        } else if (c == '\t') {
            o += "\\t";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            // skip other control
        } else {
            o.push_back(c);
        }
    }
    return o;
}

std::string json_models(const EngineLoop& engine, const std::string& folder = {},
                        bool recursive = true) {
    auto list = engine.list_models(folder, recursive);
    auto cfg = engine.config();
    std::ostringstream o;
    o << "{\"ok\":true"
      << ",\"current\":\"" << json_escape(cfg.infer.model_path) << "\""
      << ",\"models_dir\":\"" << json_escape(cfg.models_dir) << "\""
      << ",\"folder\":\"" << json_escape(folder.empty() ? cfg.models_dir : folder) << "\""
      << ",\"recursive\":" << (recursive ? "true" : "false")
      << ",\"count\":" << list.size()
      << ",\"models\":[";
    for (size_t i = 0; i < list.size(); ++i) {
        const auto& m = list[i];
        if (i) o << ",";
        o << "{\"name\":\"" << json_escape(m.name) << "\","
          << "\"path\":\"" << json_escape(m.path) << "\","
          << "\"dir\":\"" << json_escape(m.dir) << "\","
          << "\"size\":" << m.size << "}";
    }
    o << "]}";
    return o.str();
}

// Extract ?key= from first request line (before path strip).
std::string query_param(const std::string& req, const char* key) {
    const auto sp1 = req.find(' ');
    const auto sp2 = sp1 == std::string::npos ? std::string::npos : req.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return {};
    std::string full = req.substr(sp1 + 1, sp2 - sp1 - 1);
    const auto q = full.find('?');
    if (q == std::string::npos) return {};
    std::string qs = full.substr(q + 1);
    const std::string needle = std::string(key) + "=";
    auto p = qs.find(needle);
    if (p == std::string::npos) return {};
    std::string v = qs.substr(p + needle.size());
    auto amp = v.find('&');
    if (amp != std::string::npos) v = v.substr(0, amp);
    // minimal url decode
    std::string d;
    d.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == '+') d.push_back(' ');
        else if (v[i] == '%' && i + 2 < v.size()) {
            auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hex(v[i + 1]), lo = hex(v[i + 2]);
            if (hi >= 0 && lo >= 0) {
                d.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
            } else d.push_back(v[i]);
        } else d.push_back(v[i]);
    }
    return d;
}

std::string json_status(const EngineLoop& engine) {
	    const EngineSnapshot s = engine.snapshot();
	    const std::string preview = engine.preview_path();
	    const EngineConfig cfg = engine.config();
	    // Prefer runtime fields filled by engine thread (actual device after CPU fallback).
	    const std::string cap_mode =
	        s.capture_mode[0] ? std::string(s.capture_mode) : cfg.capture_mode;
	    const std::string dev_req =
	        s.device_req[0] ? std::string(s.device_req) : cfg.infer.device;
	    const std::string dev_act =
	        s.device_act[0] ? std::string(s.device_act) : cfg.infer.device;
	    const std::string model =
	        s.model_path[0] ? std::string(s.model_path) : cfg.infer.model_path;
	    const int cap_w = s.capture_w > 0 ? s.capture_w : cfg.width;
	    const int cap_h = s.capture_h > 0 ? s.capture_h : cfg.height;
// Prefer runtime origin (center crop screen coords) over static region_x=0.
		    // For center mode, DXGI ROI starts at origin; region_x/y config may be 0.
		    const int ox = s.origin_x;
		    const int oy = s.origin_y;
		    const int rx = (cap_mode == "center") ? ox
		                                         : (s.capture_w > 0 ? s.region_x : cfg.region_x);
		    const int ry = (cap_mode == "center") ? oy
		                                         : (s.capture_h > 0 ? s.region_y : cfg.region_y);
		    const int rw = cfg.region_width > 0 ? cfg.region_width : cap_w;
		    const int rh = cfg.region_height > 0 ? cfg.region_height : cap_h;
		    // Honest device labels for UI (cuda+cpu_pre is still CUDA inference)
		    std::string dev_label = dev_act;
		    if (dev_act.find("cuda") != std::string::npos &&
		        dev_act.find("cpu_pre") != std::string::npos) {
			    dev_label = "cuda (EP GPU · preprocess CPU)";
		    } else if (dev_act == "cpu" && dev_req != "cpu") {
			    dev_label = "cpu (FALLBACK from " + dev_req + ")";
		    }

		    std::ostringstream o;
		    o << "{"
		      << "\"running\":" << (s.running ? "true" : "false") << ","
		      << "\"capture_ok\":" << (s.capture_ok ? "true" : "false") << ","
		      << "\"infer_ok\":" << (s.infer_ok ? "true" : "false") << ","
		      << "\"aim_ok\":" << (s.aim_ok ? "true" : "false") << ","
		      << "\"aim_hotkey_down\":" << (s.aim_hotkey_down ? "true" : "false") << ","
		      << "\"aim_moved\":" << (s.aim_moved ? "true" : "false") << ","
		      << "\"capture_fps\":" << s.capture_fps << ","
		      << "\"infer_fps\":" << s.infer_fps << ","
		      << "\"infer_ms\":" << s.infer_ms << ","
		      << "\"last_det_count\":" << s.last_det_count << ","
		      << "\"aim_err_x\":" << s.aim_err_x << ","
		      << "\"aim_err_y\":" << s.aim_err_y << ","
		      << "\"aim_out_x\":" << s.aim_out_x << ","
		      << "\"aim_out_y\":" << s.aim_out_y << ","
		      << "\"frame_count\":" << engine.frame_count() << ","
		      << "\"backend\":\"" << s.backend << "\","
		      << "\"capture_mode\":\"" << json_escape(cap_mode) << "\","
		      << "\"capture_size\":\"" << cap_w << "x" << cap_h << "\","
		      << "\"capture_w\":" << cap_w << ","
		      << "\"capture_h\":" << cap_h << ","
		      << "\"capture_region\":[" << rx << "," << ry << "," << rw << "," << rh << "],"
		      << "\"infer_roi\":[" << ox << "," << oy << "," << cap_w << "," << cap_h << "],"
		      << "\"infer_is_fullscreen\":"
		      << ((cap_w >= 1800 && cap_h >= 1000) ? "true" : "false") << ","
		      << "\"use_region\":" << (cfg.use_region ? "true" : "false") << ","
		      << "\"region_width\":" << rw << ","
		      << "\"region_height\":" << rh << ","
		      << "\"origin\":[" << ox << "," << oy << "],"
		      << "\"device\":\"" << json_escape(dev_label) << "\","
		      << "\"device_requested\":\"" << json_escape(dev_req) << "\","
		      << "\"device_actual\":\"" << json_escape(dev_act) << "\","
	      << "\"model_path\":\"" << json_escape(model) << "\","
	      << "\"model_version\":" << cfg.infer.model_version << ","
	      << "\"input_size\":" << cfg.infer.input_resolution << ","
	      << "\"confidence\":" << cfg.infer.confidence << ","
	      << "\"nms\":" << cfg.infer.nms << ","
	      << "\"infer_enabled\":" << (cfg.infer_enabled ? "true" : "false") << ","
	      << "\"aim_enabled\":" << (cfg.aim_enabled ? "true" : "false") << ","
	      << "\"preview_path\":\"" << json_escape(preview) << "\","
	      << "\"last_error\":\"" << json_escape(s.last_error) << "\"";
#ifdef YA_WITH_AIM
	    {
		    auto as = engine.aim_status();
		    o << ",\"active_slot\":" << as.active_slot
		      << ",\"controller_type\":" << as.controller_type
		      << ",\"controller\":\"" << FullAimBridge::controller_name(
		                                    static_cast<ControllerType>(as.controller_type))
		      << "\""
		      << ",\"algorithm\":" << as.algorithm
		      << ",\"algorithm_name\":\"" << FullAimBridge::algorithm_name(
		                                       static_cast<AlgorithmType>(as.algorithm))
		      << "\""
		      << ",\"fov_px\":" << as.fov_px
		      << ",\"controller_ok\":" << (as.controller_ok ? "true" : "false")
		      << ",\"aim_error\":\"" << json_escape(as.last_error) << "\"";
	    }
#endif
	    o << "}";
	    return o.str();
	}

std::string http_response(int code, const char* status, const std::string& body,
                          const char* content_type) {
    std::ostringstream o;
    o << "HTTP/1.1 " << code << " " << status << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Cache-Control: no-store\r\n"
      << "Access-Control-Allow-Origin: *\r\n"
      << "Connection: close\r\n\r\n"
      << body;
    return o.str();
}

std::string http_response_bin(int code, const char* status, const std::vector<uint8_t>& body,
                              const char* content_type) {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << code << " " << status << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Cache-Control: no-store\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n";
    std::string out = hdr.str();
    out.append(reinterpret_cast<const char*>(body.data()), body.size());
    return out;
}

std::string request_path(const std::string& req) {
    // "GET /path?x=1 HTTP/1.1"
    const auto sp1 = req.find(' ');
    if (sp1 == std::string::npos) return "/";
    const auto sp2 = req.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) return "/";
    std::string p = req.substr(sp1 + 1, sp2 - sp1 - 1);
    const auto q = p.find('?');
    if (q != std::string::npos) p = p.substr(0, q);
    return p;
}

std::string request_body(const std::string& req) {
    const auto p = req.find("\r\n\r\n");
    if (p == std::string::npos) return {};
    return req.substr(p + 4);
}

std::string request_method(const std::string& req) {
    const auto sp = req.find(' ');
    if (sp == std::string::npos) return {};
    return req.substr(0, sp);
}

} // namespace

WebServer::WebServer(EngineLoop& engine, std::string web_root)
    : engine_(engine), web_root_(std::move(web_root)) {}

WebServer::~WebServer() { stop(); }

bool WebServer::start(const std::string& host, int port) {
    if (running_.load()) return true;
    stop_ = false;
    port_ = port;
    th_ = std::thread([this, host, port] { thread_main(host, port); });
    // wait until listening or short timeout
    for (int i = 0; i < 100 && !running_.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return running_.load();
}

void WebServer::stop() {
    stop_ = true;
#ifdef _WIN32
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s != INVALID_SOCKET) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(port_ > 0 ? port_ : 17890));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        closesocket(s);
    }
#endif
    if (th_.joinable()) th_.join();
    running_ = false;
}

void WebServer::thread_main(std::string host, int port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return;

    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_fd == INVALID_SOCKET) {
        WSACleanup();
        return;
    }

    BOOL yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes),
               sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    (void)host;
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        closesocket(listen_fd);
        WSACleanup();
        return;
    }
    if (listen(listen_fd, 16) == SOCKET_ERROR) {
        closesocket(listen_fd);
        WSACleanup();
        return;
    }

    running_ = true;

    while (!stop_.load()) {
        SOCKET client = accept(listen_fd, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;
        if (stop_.load()) {
            closesocket(client);
            break;
        }

        std::string headers;
        {
            char buf[8192];
            const int n = recv(client, buf, sizeof(buf), 0);
            if (n <= 0) {
                closesocket(client);
                continue;
            }
            headers.assign(buf, buf + n);
        }
        std::vector<uint8_t> raw_body;
        // For methods with body, complete headers + Content-Length body.
        {
            const std::string method0 = request_method(headers);
            if (method0 == "POST" || method0 == "PUT") {
                if (!recv_full_body(client, headers, raw_body)) {
                    const auto r = http_response(400, "Bad Request",
                                                 "{\"ok\":false,\"error\":\"bad body\"}",
                                                 "application/json");
                    send(client, r.data(), static_cast<int>(r.size()), 0);
                    closesocket(client);
                    continue;
                }
            }
        }
        const std::string& req = headers;
        const std::string path = request_path(req);
        const std::string method = request_method(req);
        const bool is_get = method == "GET";
        const bool is_post = method == "POST";
        const bool is_put = method == "PUT";
        const bool is_options = method == "OPTIONS";
        const std::string body_str(raw_body.begin(), raw_body.end());

        std::string response;
        if (is_options) {
            // CORS preflight for browser PUT / upload
            std::ostringstream o;
            o << "HTTP/1.1 204 No Content\r\n"
              << "Access-Control-Allow-Origin: *\r\n"
              << "Access-Control-Allow-Methods: GET, POST, PUT, OPTIONS\r\n"
              << "Access-Control-Allow-Headers: Content-Type, X-Filename\r\n"
              << "Content-Length: 0\r\n"
              << "Connection: close\r\n\r\n";
            response = o.str();
        } else if (is_get && path == "/api/health") {
            response = http_response(200, "OK", "{\"ok\":true}", "application/json");
        } else if (is_get && path == "/api/status") {
            response = http_response(200, "OK", json_status(engine_), "application/json");
        } else if (is_get && path == "/api/models") {
            const std::string dir = query_param(req, "dir");
            const std::string rec = query_param(req, "recursive");
            const bool recursive = rec.empty() || rec == "1" || rec == "true" || rec == "yes";
            response = http_response(200, "OK", json_models(engine_, dir, recursive),
                                     "application/json");
        } else if (is_post && path == "/api/models/scan") {
            // Body: {"dir":"D:/AI/models","recursive":true,"select_first":true}
            std::string body_str(raw_body.begin(), raw_body.end());
            if (body_str.empty()) body_str = request_body(req);
            std::string dir, err;
            bool recursive = true;
            bool select_first = false;
            // lightweight parse
            auto extract_str = [&](const char* key) -> std::string {
                const std::string pat = std::string("\"") + key + "\"";
                auto p = body_str.find(pat);
                if (p == std::string::npos) return {};
                p = body_str.find(':', p);
                if (p == std::string::npos) return {};
                p = body_str.find('"', p);
                if (p == std::string::npos) return {};
                std::string out;
                for (size_t i = p + 1; i < body_str.size(); ++i) {
                    if (body_str[i] == '\\' && i + 1 < body_str.size()) {
                        out.push_back(body_str[i + 1]);
                        ++i;
                    } else if (body_str[i] == '"') break;
                    else out.push_back(body_str[i]);
                }
                return out;
            };
            auto extract_bool = [&](const char* key, bool defv) {
                const std::string pat = std::string("\"") + key + "\"";
                auto p = body_str.find(pat);
                if (p == std::string::npos) return defv;
                p = body_str.find(':', p);
                if (p == std::string::npos) return defv;
                ++p;
                while (p < body_str.size() && (body_str[p] == ' ' || body_str[p] == '\t')) ++p;
                if (body_str.compare(p, 4, "true") == 0) return true;
                if (body_str.compare(p, 5, "false") == 0) return false;
                return defv;
            };
            dir = extract_str("dir");
            if (dir.empty()) dir = extract_str("folder");
            if (dir.empty()) dir = extract_str("path");
            if (dir.empty()) dir = query_param(req, "dir");
            recursive = extract_bool("recursive", true);
            select_first = extract_bool("select_first", false);
            if (dir.empty()) {
                response = http_response(400, "Bad Request",
                    "{\"ok\":false,\"error\":\"dir required\"}", "application/json");
            } else if (!engine_.set_model_folder(dir, recursive, &err)) {
                response = http_response(
                    400, "Bad Request",
                    std::string("{\"ok\":false,\"error\":\"") + json_escape(err) + "\"}",
                    "application/json");
            } else {
                auto list = engine_.list_models(dir, recursive);
                if (select_first && !list.empty()) {
                    // apply model_path via existing config path
                    std::string put =
                        std::string("{\"model_path\":\"") + json_escape(list[0].path) +
                        "\",\"infer\":{\"model_path\":\"" + json_escape(list[0].path) + "\"}}";
                    std::string e2;
                    engine_.apply_config_json(put, &e2);
                    if (engine_.running())
                        engine_.request_reload_model();
                }
                response = http_response(200, "OK", json_models(engine_, dir, recursive),
                                         "application/json");
            }
        } else if (is_post && path == "/api/models/upload") {
            std::string fname = header_value(req, "X-Filename");
            if (fname.empty()) {
                // first request line: POST /api/models/upload?name=foo.onnx HTTP/1.1
                const auto sp1 = req.find(' ');
                const auto sp2 = sp1 == std::string::npos ? std::string::npos : req.find(' ', sp1 + 1);
                if (sp1 != std::string::npos && sp2 != std::string::npos) {
                    std::string full = req.substr(sp1 + 1, sp2 - sp1 - 1);
                    auto npos = full.find("name=");
                    if (npos != std::string::npos) {
                        fname = full.substr(npos + 5);
                        auto amp = fname.find('&');
                        if (amp != std::string::npos) fname = fname.substr(0, amp);
                    }
                }
            }
            if (fname.empty()) fname = "upload.onnx";
            // URL-decode minimal
            {
                std::string d;
                d.reserve(fname.size());
                for (size_t i = 0; i < fname.size(); ++i) {
                    if (fname[i] == '+') d.push_back(' ');
                    else if (fname[i] == '%' && i + 2 < fname.size()) {
                        auto hex = [](char c) -> int {
                            if (c >= '0' && c <= '9') return c - '0';
                            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                            return -1;
                        };
                        int hi = hex(fname[i + 1]), lo = hex(fname[i + 2]);
                        if (hi >= 0 && lo >= 0) {
                            d.push_back(static_cast<char>((hi << 4) | lo));
                            i += 2;
                        } else d.push_back(fname[i]);
                    } else d.push_back(fname[i]);
                }
                fname = std::move(d);
            }
            // strip path separators
            {
                auto slash = fname.find_last_of("/\\");
                if (slash != std::string::npos) fname = fname.substr(slash + 1);
            }
            std::string out_path, err;
            if (!engine_.save_model_file(fname, raw_body, &out_path, &err)) {
                response = http_response(
                    400, "Bad Request",
                    std::string("{\"ok\":false,\"error\":\"") + json_escape(err) + "\"}",
                    "application/json");
            } else {
                response = http_response(
                    200, "OK",
                    std::string("{\"ok\":true,\"path\":\"") + json_escape(out_path) +
                        "\",\"name\":\"" + json_escape(fname) + "\"}",
                    "application/json");
            }
        } else if (is_get && path == "/api/detections") {
            auto dets = engine_.last_detections();
            std::ostringstream o;
            o << "{\"count\":" << dets.size() << ",\"items\":[";
            for (size_t i = 0; i < dets.size(); ++i) {
                const auto& d = dets[i];
                if (i) o << ",";
o << "{"
	                  << "\"class_id\":" << d.class_id << ","
	                  << "\"confidence\":" << d.confidence << ","
	                  << "\"x\":" << d.x << ",\"y\":" << d.y << ","
	                  << "\"w\":" << d.w << ",\"h\":" << d.h << ","
	                  << "\"cx\":" << d.cx << ",\"cy\":" << d.cy << ","
	                  << "\"track_id\":" << d.track_id
	                  << "}";
	            }
	            o << "]}";
	            response = http_response(200, "OK", o.str(), "application/json");
	        } else if (is_get && (path == "/api/preview.bmp" || path == "/api/preview")) {
            auto bmp = engine_.preview_bmp();
            if (bmp.empty()) {
                response = http_response(404, "Not Found",
                                         "no preview yet — wait 1s after Start", "text/plain");
            } else {
                response = http_response_bin(200, "OK", bmp, "image/bmp");
            }
        } else if (is_post && path == "/api/engine/start") {
            const bool ok = engine_.start();
            std::ostringstream o;
            o << "{\"ok\":" << (ok ? "true" : "false")
              << ",\"running\":" << (engine_.running() ? "true" : "false")
              << ",\"action\":\"start\"}";
            response = http_response(ok ? 200 : 500, ok ? "OK" : "Error", o.str(),
                                     "application/json");
        } else if (is_post && path == "/api/engine/stop") {
            engine_.stop();
            response = http_response(
                200, "OK",
                "{\"ok\":true,\"running\":false,\"action\":\"stop\",\"infer_unloaded\":true}",
                "application/json");
} else if (is_get && path == "/api/config") {
            response = http_response(200, "OK", engine_.config_json(), "application/json");
} else if ((is_put || is_post) && path == "/api/config") {
			            const std::string& body = body_str.empty() ? request_body(req) : body_str;
			            std::string err;
			            // sendBeacon may POST empty body → treat as flush-only save
			            if (body.empty() || body == "{}" || body == "null") {
				            if (!engine_.save_user_config(&err)) {
					            response = http_response(
					                500, "Error",
					                std::string("{\"ok\":false,\"error\":\"") +
					                    json_escape(err.empty() ? "save failed" : err) + "\"}",
					                "application/json");
				            } else {
					            response = http_response(200, "OK",
					                                     "{\"ok\":true,\"saved\":true}",
					                                     "application/json");
				            }
			            } else if (!engine_.apply_config_json(body, &err)) {
			                response = http_response(400, "Bad Request",
			                                         std::string("{\"ok\":false,\"error\":\"") +
			                                             (err.empty() ? "invalid" : err) + "\"}",
			                                         "application/json");
			            } else {
			                response = http_response(200, "OK",
			                                         std::string("{\"ok\":true,\"config\":") +
			                                             engine_.config_json() + "}",
			                                         "application/json");
			            }
		} else if ((is_put || is_post) && path == "/api/config/save") {
			std::string err;
			if (!engine_.save_user_config(&err)) {
				response = http_response(500, "Error",
				                         std::string("{\"ok\":false,\"error\":\"") +
				                             json_escape(err.empty() ? "save failed" : err) +
				                             "\"}",
				                         "application/json");
			} else {
				response =
				    http_response(200, "OK", "{\"ok\":true,\"saved\":true}", "application/json");
			}
		} else if (is_post && path == "/api/engine/reload_model") {
			            std::string err;
			            if (!engine_.request_reload_model(&err)) {
				            response = http_response(
				                400, "Bad Request",
				                std::string("{\"ok\":false,\"error\":\"") + json_escape(err) + "\"}",
				                "application/json");
			            } else {
				            response =
				                http_response(200, "OK", "{\"ok\":true,\"action\":\"reload_model\"}",
				                              "application/json");
			            }
		        } else if (is_post && path == "/api/engine/reload_capture") {
			            std::string err;
			            if (!engine_.request_reload_capture(&err)) {
				            response = http_response(
				                400, "Bad Request",
				                std::string("{\"ok\":false,\"error\":\"") + json_escape(err) + "\"}",
				                "application/json");
			            } else {
				            response = http_response(
				                200, "OK", "{\"ok\":true,\"action\":\"reload_capture\"}",
				                "application/json");
			            }
#ifdef YA_WITH_AIM
			        } else if (is_post && path == "/api/controller/test") {
		            const std::string& body = body_str.empty() ? request_body(req) : body_str;
		            auto find_str = [&](const char *key, const std::string &fb) {
			            const std::string pat = std::string("\"") + key + "\"";
			            auto p = body.find(pat);
		            if (p == std::string::npos)
			            return fb;
		            p = body.find('"', body.find(':', p));
		            if (p == std::string::npos)
			            return fb;
		            auto e = body.find('"', p + 1);
		            if (e == std::string::npos)
			            return fb;
		            return body.substr(p + 1, e - p - 1);
	            };
	            auto find_num = [&](const char *key, int fb) {
		            const std::string pat = std::string("\"") + key + "\"";
		            auto p = body.find(pat);
		            if (p == std::string::npos)
			            return fb;
		            p = body.find(':', p);
		            if (p == std::string::npos)
			            return fb;
		            try {
			            return std::stoi(body.substr(p + 1));
		            } catch (...) {
			            return fb;
		            }
	            };
	            std::string type = find_str("controller", find_str("type", "WindowsAPI"));
	            std::string port_s = find_str("makcuPort", "COM5");
	            int baud = find_num("makcuBaudRate", 40000);
	            int logi = find_num("logiDriverType", 0);
	            std::string err;
	            bool ok = engine_.test_controller(type, port_s, baud, logi, &err);
	            response = http_response(
	                ok ? 200 : 500, ok ? "OK" : "Error",
	                std::string("{\"ok\":") + (ok ? "true" : "false") + ",\"message\":\"" +
	                    json_escape(err) + "\"}",
	                "application/json");
#endif
	        } else if (is_get) {
            std::string file_path = web_root_ + "/index.html";
            if (path == "/css/app.css") file_path = web_root_ + "/css/app.css";
            else if (path == "/js/app.js") file_path = web_root_ + "/js/app.js";
            else if (path != "/" && path != "/index.html") {
                // only allow simple relative under web root
                if (path.find("..") == std::string::npos && path.find('\\') == std::string::npos)
                    file_path = web_root_ + path;
            }
            std::string body = read_file(file_path);
            if (body.empty() && (path == "/" || path == "/index.html")) {
                body = "<!doctype html><html><body><h1>Yolo Aim Launcher</h1>"
                       "<p>Missing webui. Try /api/status and /api/preview.bmp</p></body></html>";
            }
            const char* ct = "text/html; charset=utf-8";
            if (file_path.size() >= 4 && file_path.substr(file_path.size() - 4) == ".css")
                ct = "text/css; charset=utf-8";
            if (file_path.size() >= 3 && file_path.substr(file_path.size() - 3) == ".js")
                ct = "application/javascript; charset=utf-8";
            response = body.empty() ? http_response(404, "Not Found", "not found", "text/plain")
                                    : http_response(200, "OK", body, ct);
        } else {
            response = http_response(405, "Method Not Allowed", "method", "text/plain");
        }

        // Send full response (handle partial writes)
        {
            const char *data = response.data();
            int remaining = static_cast<int>(response.size());
            while (remaining > 0) {
                int sent = send(client, data, remaining, 0);
                if (sent <= 0)
                    break;
                data += sent;
                remaining -= sent;
            }
        }
        closesocket(client);
    }

    closesocket(listen_fd);
    WSACleanup();
    running_ = false;
#else
    (void)host;
    (void)port;
#endif
}

} // namespace ya
