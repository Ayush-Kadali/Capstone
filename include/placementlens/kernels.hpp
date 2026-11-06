#pragma once
#include "placementlens/core.hpp"

namespace pl {
struct KernelStats {
    std::size_t tile_count = 0;
    std::size_t tile_load_bytes = 0;
    std::size_t peak_local_bytes = 0;
};
// Fill preallocated output without resizing or replacing its storage. Inputs
// and output must not alias. Statistics accumulate across calls. Tiling applies
// only to MatMul/MatMulRelu when tile_size is nonzero; other kernels are scalar.
void compute_software(Op op, const std::vector<const Tensor*>& inputs,
                      Tensor& output, const Config& config, KernelStats& stats,
                      const std::atomic<bool>* cancel = nullptr);
}
