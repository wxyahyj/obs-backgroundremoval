#ifndef DML_PREPROCESSOR_H
#define DML_PREPROCESSOR_H

#ifdef _WIN32
#define NOMINMAX
#endif

#include <cstdint>
#include <vector>

struct DmlPreprocessParams {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float scale;
    int padX;
    int padY;
};

// Preprocessed frame ready for ONNX Runtime DML EP inference.
// Owns a float buffer (CHW layout, normalized) that the inference thread
// wraps directly with Ort::Value::CreateTensor.
struct DmlPreprocessedFrame {
    std::vector<float> data;   // CHW, NCHW with N=1
    int width = 0;
    int height = 0;
    int channels = 3;
    int srcWidth = 0;   // crop region pixel width before letterbox
    int srcHeight = 0;  // crop region pixel height before letterbox
    int cropX = 0;      // crop region top-left X in the full frame
    int cropY = 0;      // crop region top-left Y in the full frame
    int fullWidth = 0;  // full frame width (for restoring normalized coords)
    int fullHeight = 0; // full frame height (for restoring normalized coords)
    int64_t timestamp = 0;

    bool valid() const { return !data.empty() && width > 0 && height > 0; }
    void reset() { data.clear(); width = height = channels = 0; srcWidth = srcHeight = 0;
                   cropX = cropY = fullWidth = fullHeight = 0; timestamp = 0; }
};

class DmlPreprocessor {
public:
    DmlPreprocessor();
    ~DmlPreprocessor();

    bool initialize();
    void release();

    // CPU-side letterbox + normalize from raw BGRA pixels.
    // Callable from the render thread (uses no D3D11 immediate context).
    bool preprocessFromBgra(
        const uint8_t* bgraData,
        int srcWidth,
        int srcHeight,
        int srcStrideBytes,
        int dstWidth,
        int dstHeight,
        DmlPreprocessedFrame& outFrame,
        DmlPreprocessParams* outParams = nullptr
    );

    bool isInitialized() const { return initialized_; }

    static DmlPreprocessParams calculateParams(
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight
    );

private:
    bool initialized_;
};

#endif
