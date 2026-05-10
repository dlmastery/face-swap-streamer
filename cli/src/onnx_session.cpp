#include "faceswap/onnx_session.hpp"

#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>

#include <numeric>
#include <stdexcept>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <codecvt>
  #include <locale>
#endif

namespace faceswap {

namespace {

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), out.data(), n);
    return out;
}
#endif

std::int64_t shape_count(const std::vector<std::int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>{});
}

}  // namespace

struct OnnxSession::Impl {
    Ort::Env             env{ORT_LOGGING_LEVEL_WARNING, "faceswap"};
    Ort::SessionOptions  opts;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string>      input_names;
    std::vector<std::string>      output_names;
    std::vector<Ort::AllocatedStringPtr> input_name_holders;
    std::vector<Ort::AllocatedStringPtr> output_name_holders;
    Provider                      active = Provider::CPU;
};

OnnxSession::OnnxSession() : p(std::make_unique<Impl>()) {}
OnnxSession::~OnnxSession() = default;
OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

void OnnxSession::load(const std::string& path, Provider provider, int cuda_device) {
    p->opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    p->opts.SetIntraOpNumThreads(1);
    p->opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    bool ep_attached = false;
    if (provider == Provider::TensorRT) {
        try {
            OrtTensorRTProviderOptions trt{};
            trt.device_id = cuda_device;
            trt.trt_fp16_enable = 1;
            p->opts.AppendExecutionProvider_TensorRT(trt);
            p->active = Provider::TensorRT;
            ep_attached = true;
        } catch (const Ort::Exception& e) {
            fmt::print(stderr, "[ort] TRT EP unavailable, falling back to CUDA: {}\n", e.what());
        }
    }
    if (!ep_attached && (provider == Provider::CUDA || provider == Provider::TensorRT)) {
        try {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = cuda_device;
            cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda.do_copy_in_default_stream = 1;
            p->opts.AppendExecutionProvider_CUDA(cuda);
            p->active = Provider::CUDA;
            ep_attached = true;
        } catch (const Ort::Exception& e) {
            fmt::print(stderr, "[ort] CUDA EP unavailable, falling back to CPU: {}\n", e.what());
        }
    }
    if (!ep_attached) p->active = Provider::CPU;

#ifdef _WIN32
    p->session = std::make_unique<Ort::Session>(p->env, widen(path).c_str(), p->opts);
#else
    p->session = std::make_unique<Ort::Session>(p->env, path.c_str(), p->opts);
#endif

    Ort::AllocatorWithDefaultOptions alloc;
    const std::size_t n_in  = p->session->GetInputCount();
    const std::size_t n_out = p->session->GetOutputCount();
    p->input_names.clear();   p->input_name_holders.clear();
    p->output_names.clear();  p->output_name_holders.clear();
    p->input_names.reserve(n_in);   p->input_name_holders.reserve(n_in);
    p->output_names.reserve(n_out); p->output_name_holders.reserve(n_out);
    for (std::size_t i = 0; i < n_in; ++i) {
        p->input_name_holders.emplace_back(p->session->GetInputNameAllocated(i, alloc));
        p->input_names.emplace_back(p->input_name_holders.back().get());
    }
    for (std::size_t i = 0; i < n_out; ++i) {
        p->output_name_holders.emplace_back(p->session->GetOutputNameAllocated(i, alloc));
        p->output_names.emplace_back(p->output_name_holders.back().get());
    }
}

std::vector<OnnxSession::Output> OnnxSession::run(
    const std::vector<std::string>& input_names,
    const std::vector<const float*>& input_data,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::string>& output_names) {

    if (!p->session) throw std::runtime_error("OnnxSession::run before load");
    if (input_data.size() != input_names.size() || input_shapes.size() != input_names.size())
        throw std::runtime_error("OnnxSession::run input arity mismatch");

    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_vals;
    input_vals.reserve(input_names.size());
    for (std::size_t i = 0; i < input_names.size(); ++i) {
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(
            cpu_mem,
            const_cast<float*>(input_data[i]),
            (size_t)shape_count(input_shapes[i]),
            input_shapes[i].data(), input_shapes[i].size()));
    }

    std::vector<const char*> in_c, out_c;
    in_c.reserve(input_names.size());
    out_c.reserve(output_names.size());
    for (auto& s : input_names)  in_c.push_back(s.c_str());
    for (auto& s : output_names) out_c.push_back(s.c_str());

    auto outputs = p->session->Run(
        Ort::RunOptions{nullptr},
        in_c.data(),  input_vals.data(),  input_vals.size(),
        out_c.data(), out_c.size());

    std::vector<Output> result;
    result.reserve(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        Output o;
        o.name = output_names[i];
        auto info  = outputs[i].GetTensorTypeAndShapeInfo();
        o.shape    = info.GetShape();
        const std::size_t n = (std::size_t)info.GetElementCount();
        o.data.assign(outputs[i].GetTensorData<float>(), outputs[i].GetTensorData<float>() + n);
        result.push_back(std::move(o));
    }
    return result;
}

const std::vector<std::string>& OnnxSession::input_names()  const { return p->input_names; }
const std::vector<std::string>& OnnxSession::output_names() const { return p->output_names; }
OnnxSession::Provider          OnnxSession::active_provider() const { return p->active; }

}  // namespace faceswap
