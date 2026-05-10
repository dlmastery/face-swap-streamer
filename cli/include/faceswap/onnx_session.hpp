#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward-declare ORT types so this header doesn't pull in the heavy
// onnxruntime_cxx_api.h. ORT defines these as structs, so match.
namespace Ort { struct Session; struct Env; struct SessionOptions; }

namespace faceswap {

/// Thin RAII wrapper around ONNX Runtime's Session for a single model file.
/// Owns its own Env so each session is independent (cheaper than sharing
/// for small numbers of sessions; we have ~6).
///
/// Thread-safety: ORT sessions are thread-safe for inference (Run is
/// internally serialised) but expensive — one session per model is fine.
class OnnxSession {
public:
    enum class Provider { CPU, CUDA, TensorRT };

    OnnxSession();
    ~OnnxSession();

    OnnxSession(const OnnxSession&) = delete;
    OnnxSession& operator=(const OnnxSession&) = delete;
    OnnxSession(OnnxSession&&) noexcept;
    OnnxSession& operator=(OnnxSession&&) noexcept;

    /// Loads a .onnx file and configures the requested execution provider.
    /// Falls back to CPU if CUDA/TRT init fails — unlike the Python build
    /// where we make this fatal, the CLI tolerates it but logs loudly.
    void load(const std::string& path, Provider provider, int cuda_device = 0);

    /// Run inference. Caller supplies input tensor data + shape; output
    /// tensors are returned as flat float vectors paired with their shape.
    /// All inputs/outputs are float32 NCHW for the models we care about.
    struct Output {
        std::vector<float> data;
        std::vector<int64_t> shape;
        std::string name;
    };
    std::vector<Output> run(
        const std::vector<std::string>& input_names,
        const std::vector<const float*>& input_data,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::string>& output_names);

    /// Convenience helpers — populated after load(), match the model's signature.
    const std::vector<std::string>& input_names() const;
    const std::vector<std::string>& output_names() const;
    Provider active_provider() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

}  // namespace faceswap
