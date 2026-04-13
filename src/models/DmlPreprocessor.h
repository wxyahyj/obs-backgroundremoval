#ifndef DML_PREPROCESSOR_H
#define DML_PREPROCESSOR_H

#ifdef _WIN32
#define NOMINMAX
#endif

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>

using Microsoft::WRL::ComPtr;

struct DmlPreprocessParams {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float scale;
    int padX;
    int padY;
};

class DmlPreprocessor {
public:
    DmlPreprocessor();
    ~DmlPreprocessor();
    
    // 初始化预处理器（不再需要传入D3D11设备，会从纹理动态获取）
    bool initialize();
    void release();
    
    bool preprocessFromTexture(
        ID3D11Texture2D* srcTexture,
        float* dstBuffer,
        int dstWidth,
        int dstHeight,
        DmlPreprocessParams* outParams = nullptr
    );
    
    bool isInitialized() const { return initialized_; }
    
    static DmlPreprocessParams calculateParams(
        int srcWidth, int srcHeight,
        int dstWidth, int dstHeight
    );
    
private:
    bool createD3D12Device();
    bool copyAndPreprocess(
        ID3D11Texture2D* srcTexture,
        float* dstBuffer,
        const DmlPreprocessParams& params
    );
    
    bool initialized_;
    
    // D3D12设备用于未来的GPU加速预处理（可选）
    ComPtr<ID3D12Device> d3d12Device_;
    ComPtr<ID3D12CommandQueue> commandQueue_;
    ComPtr<ID3D12CommandAllocator> commandAllocator_;
    ComPtr<ID3D12GraphicsCommandList> commandList_;
    ComPtr<ID3D12Fence> fence_;
    UINT64 fenceValue_;
    HANDLE fenceEvent_;
    
    // 复用staging texture（每次从纹理获取OBS设备）
    ComPtr<ID3D11Texture2D> cachedStagingTexture_;
    D3D11_TEXTURE2D_DESC cachedStagingDesc_;
    
    std::vector<unsigned char> cpuBuffer_;
    std::vector<float> preprocessedBuffer_;
};

#endif
