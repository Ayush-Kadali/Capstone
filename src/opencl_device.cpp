#include "placementlens/device.hpp"
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#if defined(PL_HAVE_OPENCL) && PL_HAVE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace pl {
namespace {
void check(cl_int status, const std::string& operation) {
    if (status != CL_SUCCESS)
        throw std::runtime_error("E_OPENCL: " + operation + " failed (" + std::to_string(status) + ")");
}
void require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error("E_OPENCL: " + reason);
}
cl_device_id find_device() {
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "enumerate platforms");
    require(count != 0, "no OpenCL platform available");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "read platforms");
    for (auto platform : platforms) {
        cl_device_id device = nullptr;
        const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (status == CL_DEVICE_NOT_FOUND) continue;
        check(status, "enumerate GPU devices");
        cl_bool available = CL_FALSE, compiler = CL_FALSE;
        check(clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr), "device availability");
        check(clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler), &compiler, nullptr), "compiler availability");
        if (available && compiler) return device;
    }
    throw std::runtime_error("E_OPENCL: no available GPU with an OpenCL compiler");
}
std::string device_string(cl_device_id device, cl_device_info parameter) {
    std::size_t size = 0;
    check(clGetDeviceInfo(device, parameter, 0, nullptr, &size), "device description size");
    std::string value(size, '\0');
    check(clGetDeviceInfo(device, parameter, size, value.data(), nullptr), "device description");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}
const char* kernel_source = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
__kernel void element_add(__global const float* a, __global const float* b,
                          __global float* out) {
    size_t i = get_global_id(0); out[i] = a[i] + b[i];
}
__kernel void element_relu(__global const float* a, __global float* out) {
    size_t i = get_global_id(0); out[i] = isfinite(a[i]) ? fmax(a[i], 0.0f) : a[i];
}
__kernel void matrix_product(__global const float* a, __global const float* b,
                             __global float* out, uint inner, uint columns, uint activate) {
    size_t i = get_global_id(0), row = i / columns, col = i % columns;
    float acc = 0.0f;
    for (uint k = 0; k < inner; ++k) acc += a[row * inner + k] * b[k * columns + col];
    out[i] = activate && isfinite(acc) ? fmax(acc, 0.0f) : acc;
}
__kernel void row_sum(__global const float* a, __global float* out, uint columns) {
    size_t row = get_global_id(0); float acc = 0.0f;
    for (uint column = 0; column < columns; ++column) acc += a[row * columns + column];
    out[row] = acc;
}
)CLC";

class OpenCLDevice final : public DeviceBackend {
    struct Buffer { cl_mem memory; Shape shape; bool initialized = false; };
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    std::map<std::string, cl_kernel> kernels_;
    std::map<std::string, Buffer> buffers_;
    std::string description_;
    double elapsed_ = -1;
    std::size_t diagnostic_bytes_ = 0;
    void cleanup() noexcept {
        if (queue_) clFinish(queue_);
        for (auto& entry : buffers_) clReleaseMemObject(entry.second.memory);
        for (auto& entry : kernels_) clReleaseKernel(entry.second);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }
    Buffer& buffer(const std::string& handle) {
        auto found = buffers_.find(handle);
        require(found != buffers_.end(), "unknown allocation " + handle);
        return found->second;
    }
    void finish_event(cl_event event) {
        const cl_int status = clWaitForEvents(1, &event);
        cl_ulong start = 0, end = 0;
        const auto a = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
        const auto b = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
        clReleaseEvent(event);
        check(status, "wait for event");
        elapsed_ = a == CL_SUCCESS && b == CL_SUCCESS && end >= start ? static_cast<double>(end - start) / 1000.0 : -1.0;
    }
public:
    OpenCLDevice() {
        try {
            const auto device = find_device();
            description_ = "OpenCL GPU: " + device_string(device, CL_DEVICE_NAME) + " (" + device_string(device, CL_DEVICE_VERSION) + ")";
            cl_int status = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
            check(status, "create context");
            queue_ = clCreateCommandQueue(context_, device, CL_QUEUE_PROFILING_ENABLE, &status);
            check(status, "create profiling queue");
            program_ = clCreateProgramWithSource(context_, 1, &kernel_source, nullptr, &status);
            check(status, "create kernel program");
            status = clBuildProgram(program_, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
            if (status != CL_SUCCESS) {
                std::size_t size = 0;
                clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
                std::string log(size, '\0');
                if (size) clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
                throw std::runtime_error("E_OPENCL: build kernels failed: " + log);
            }
            for (const auto* key : {"element_add", "element_relu", "matrix_product", "row_sum"}) {
                auto kernel = clCreateKernel(program_, key, &status);
                check(status, "create kernel");
                kernels_.emplace(key, kernel);
            }
        } catch (...) { cleanup(); throw; }
    }
    ~OpenCLDevice() override { cleanup(); }
    std::string description() const override { return description_; }
    double last_elapsed_us() const override { return elapsed_; }
    std::size_t diagnostic_readback_bytes() const override { return diagnostic_bytes_; }
    void allocate(const std::string& handle, Shape shape) override {
        elapsed_ = -1;
        require(handle.rfind("D:", 0) == 0 && handle.size() > 2, "device handle must start with D:");
        require(!buffers_.count(handle), "duplicate allocation " + handle);
        cl_int status = CL_SUCCESS;
        auto memory = clCreateBuffer(context_, CL_MEM_READ_WRITE, shape.bytes(), nullptr, &status);
        check(status, "allocate buffer");
        try { buffers_.emplace(handle, Buffer{memory, shape, false}); }
        catch (...) { clReleaseMemObject(memory); throw; }
    }
    void release(const std::string& handle) override {
        elapsed_ = -1;
        auto& value = buffer(handle);
        check(clReleaseMemObject(value.memory), "release buffer");
        buffers_.erase(handle);
    }
    void write(const std::string& handle, const Tensor& host) override {
        elapsed_ = -1;
        auto& target = buffer(handle);
        require(!target.initialized, "cannot overwrite initialized allocation");
        require(target.shape == host.shape && host.values.size() == host.shape.elements(), "write shape/storage mismatch");
        for (float value : host.values) require(std::isfinite(value), "nonfinite host input");
        cl_event event = nullptr;
        check(clEnqueueWriteBuffer(queue_, target.memory, CL_TRUE, 0, host.shape.bytes(), host.values.data(), 0, nullptr, &event), "upload buffer");
        finish_event(event);
        target.initialized = true;
    }
    void read(const std::string& handle, Tensor& host) override {
        elapsed_ = -1;
        auto& source = buffer(handle);
        require(source.initialized, "read of uninitialized allocation");
        require(source.shape == host.shape && host.values.size() == host.shape.elements(), "read shape/storage mismatch");
        cl_event event = nullptr;
        check(clEnqueueReadBuffer(queue_, source.memory, CL_TRUE, 0, host.shape.bytes(), host.values.data(), 0, nullptr, &event), "download buffer");
        finish_event(event);
        for (float value : host.values) require(std::isfinite(value), "device produced nonfinite output");
    }
    void synchronize() override {
        elapsed_ = -1;
        check(clFinish(queue_), "synchronize queue");
    }
    void compute(const Command& command) override {
        elapsed_ = -1;
        require(command.version == command_version && command.kind == Kind::Compute && command.backend == Backend::Device, "invalid compute descriptor");
        const bool unary = command.op == Op::Relu || command.op == Op::RowSum;
        require(command.inputs.size() == (unary ? 1U : 2U), "wrong kernel operand count");
        auto& output = buffer(command.output);
        auto& first = buffer(command.inputs[0]);
        require(!output.initialized && first.initialized, "invalid kernel initialization state");
        require(output.shape == command.shape, "descriptor output shape mismatch");
        for (const auto& input : command.inputs) {
            require(input != command.output, "in-place device compute is unsupported");
            require(buffer(input).initialized, "uninitialized kernel operand");
        }
        std::string key;
        if (command.op == Op::Relu) {
            require(first.shape == output.shape, "ReLU shape mismatch"); key = "element_relu";
        } else if (command.op == Op::RowSum) {
            require(output.shape == Shape{first.shape.rows, 1}, "RowSum shape mismatch"); key = "row_sum";
        } else if (command.op == Op::Add) {
            require(first.shape == buffer(command.inputs[1]).shape && first.shape == output.shape, "Add shape mismatch"); key = "element_add";
        } else {
            require(command.op == Op::MatMul || command.op == Op::MatMulRelu, "unknown kernel opcode");
            const auto second = buffer(command.inputs[1]).shape;
            require(first.shape.cols == second.rows && output.shape == Shape{first.shape.rows, second.cols}, "MatMul shape mismatch"); key = "matrix_product";
        }
        cl_kernel kernel = kernels_.at(key);
        cl_uint index = 0;
        for (const auto& input : command.inputs) {
            auto memory = buffer(input).memory;
            check(clSetKernelArg(kernel, index++, sizeof(memory), &memory), "set input argument");
        }
        check(clSetKernelArg(kernel, index++, sizeof(output.memory), &output.memory), "set output argument");
        if (key == "matrix_product") {
            const cl_uint inner = static_cast<cl_uint>(first.shape.cols), columns = static_cast<cl_uint>(output.shape.cols);
            const cl_uint activate = command.op == Op::MatMulRelu ? 1U : 0U;
            check(clSetKernelArg(kernel, index++, sizeof(inner), &inner), "set inner dimension");
            check(clSetKernelArg(kernel, index++, sizeof(columns), &columns), "set column dimension");
            check(clSetKernelArg(kernel, index++, sizeof(activate), &activate), "set activation");
        } else if (command.op == Op::RowSum) {
            const cl_uint columns = static_cast<cl_uint>(first.shape.cols);
            check(clSetKernelArg(kernel, index++, sizeof(columns), &columns), "set reduction dimension");
        }
        const auto global = output.shape.elements();
        cl_event event = nullptr;
        check(clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, &global, nullptr, 0, nullptr, &event), "dispatch kernel");
        finish_event(event);
        // Strict numerical diagnostics apply even to dead graph values. This
        // extra readback is validation traffic, not a planner COPY command.
        // Keep elapsed_ as the kernel event duration; host wall time includes
        // this diagnostic transfer and scan.
        std::vector<float> diagnostic(output.shape.elements());
        check(clEnqueueReadBuffer(queue_, output.memory, CL_TRUE, 0, output.shape.bytes(), diagnostic.data(), 0, nullptr, nullptr), "validate kernel output");
        diagnostic_bytes_ += output.shape.bytes();
        for (float value : diagnostic) require(std::isfinite(value), "device computation produced nonfinite output");
        output.initialized = true;
    }
};
}
bool opencl_available(std::string& reason) {
    try { auto device = find_device(); reason = device_string(device, CL_DEVICE_NAME); return true; }
    catch (const std::exception& error) { reason = error.what(); return false; }
}
std::unique_ptr<DeviceBackend> make_opencl_device() { return std::make_unique<OpenCLDevice>(); }
}
#else
namespace pl {
bool opencl_available(std::string& reason) {
    reason = "OpenCL backend was not compiled; install OpenCL headers/runtime and configure PL_HAVE_OPENCL through CMake";
    return false;
}
std::unique_ptr<DeviceBackend> make_opencl_device() {
    throw std::runtime_error("E_OPENCL: backend was not compiled");
}
}
#endif
