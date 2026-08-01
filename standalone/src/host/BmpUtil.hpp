#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ya {

// Pack BGR8 tightly-packed frame into a 24-bit BMP (bottom-up rows).
inline bool bgr_to_bmp_bytes(const uint8_t* bgr, int width, int height,
                             std::vector<uint8_t>& out) {
    if (!bgr || width <= 0 || height <= 0) return false;
    const int row_stride = ((width * 3 + 3) / 4) * 4;
    const int image_size = row_stride * height;
    const int file_size = 14 + 40 + image_size;
    out.assign(static_cast<size_t>(file_size), 0);

    auto* p = out.data();
    // BITMAPFILEHEADER
    p[0] = 'B';
    p[1] = 'M';
    std::memcpy(p + 2, &file_size, 4);
    const uint32_t off = 14 + 40;
    std::memcpy(p + 10, &off, 4);
    // BITMAPINFOHEADER
    const uint32_t biSize = 40;
    const int32_t biWidth = width;
    const int32_t biHeight = height; // positive = bottom-up
    const uint16_t biPlanes = 1;
    const uint16_t biBitCount = 24;
    std::memcpy(p + 14, &biSize, 4);
    std::memcpy(p + 18, &biWidth, 4);
    std::memcpy(p + 22, &biHeight, 4);
    std::memcpy(p + 26, &biPlanes, 2);
    std::memcpy(p + 28, &biBitCount, 2);
    std::memcpy(p + 34, &image_size, 4);

    uint8_t* bits = p + 54;
    for (int y = 0; y < height; ++y) {
        const uint8_t* src = bgr + static_cast<size_t>(height - 1 - y) * width * 3;
        uint8_t* dst = bits + static_cast<size_t>(y) * row_stride;
        std::memcpy(dst, src, static_cast<size_t>(width) * 3);
    }
    return true;
}

inline bool write_bmp_file(const std::string& path, const uint8_t* bgr, int width,
                           int height) {
    std::vector<uint8_t> bytes;
    if (!bgr_to_bmp_bytes(bgr, width, height, bytes)) return false;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(ofs);
}

} // namespace ya
