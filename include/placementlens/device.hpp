#pragma once
#include "placementlens/core.hpp"
#include <memory>

namespace pl {
// A backend owns its device allocations. Operations are deliberately blocking in
// this release; the command interpreter still enforces explicit SYNC boundaries.
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;
    virtual std::string description() const = 0;
    virtual void allocate(const std::string& handle, Shape shape) = 0;
    virtual void release(const std::string& handle) = 0;
    virtual void write(const std::string& handle, const Tensor& host) = 0;
    virtual void read(const std::string& handle, Tensor& host) = 0;
    virtual void compute(const Command& command) = 0;
    virtual void synchronize() = 0;
    // Event execution duration, excluding host dispatch and wait overhead.
    // Returns -1 when the previous operation has no profiling event.
    virtual double last_elapsed_us() const = 0;
    // Cumulative successful diagnostic reads for this backend instance, even
    // when the following numerical check fails. Excludes explicit read().
    virtual std::size_t diagnostic_readback_bytes() const { return 0; }
};
// Availability probes do not compile kernels. Factory/build errors must be
// reported as failures, not silently replaced by software or test skips.
bool opencl_available(std::string& reason);
std::unique_ptr<DeviceBackend> make_opencl_device();
}
