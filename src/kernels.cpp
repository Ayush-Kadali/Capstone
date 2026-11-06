#include "placementlens/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pl {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cancellation(const std::atomic<bool>* cancel) {
    require(!cancel || !cancel->load(std::memory_order_relaxed), "E_CANCELLED: kernel cancelled");
}
void finite(float value) {
    require(std::isfinite(value), "E_NUMERIC: computation produced nonfinite output");
}
void validate(Op op, const std::vector<const Tensor*>& inputs, const Tensor& output) {
    const bool unary = op == Op::Relu || op == Op::RowSum;
    require(op == Op::MatMul || op == Op::MatMulRelu || op == Op::Add || unary,
            "E_OPCODE: unsupported software kernel");
    require(inputs.size() == (unary ? 1U : 2U), "E_ARITY: wrong kernel operand count");
    require(output.values.size() == output.shape.elements(), "E_VALUE: invalid output storage");
    for (const auto* input : inputs) {
        require(input != nullptr && input != &output, "E_BUFFER: missing or aliased kernel operand");
        require(input->values.size() == input->shape.elements(), "E_VALUE: invalid input storage");
        for (float value : input->values) require(std::isfinite(value), "E_VALUE: nonfinite kernel input");
    }
    const auto a = inputs[0]->shape;
    if (op == Op::Relu) require(a == output.shape, "E_SHAPE: ReLU shape mismatch");
    else if (op == Op::RowSum) require(output.shape == Shape{a.rows, 1}, "E_SHAPE: RowSum requires rows x 1 output");
    else {
        const auto b = inputs[1]->shape;
        if (op == Op::Add) require(a == b && a == output.shape, "E_SHAPE: Add shape mismatch");
        else require(a.cols == b.rows && output.shape == Shape{a.rows, b.cols}, "E_SHAPE: MatMul shape mismatch");
    }
}
void scalar_product(const Tensor& a, const Tensor& b, Tensor& output, bool activate,
                    const std::atomic<bool>* cancel) {
    for (std::size_t row = 0; row < a.shape.rows; ++row) {
        for (std::size_t col = 0; col < b.shape.cols; ++col) {
            cancellation(cancel);
            float acc = 0;
            for (std::size_t k = 0; k < a.shape.cols; ++k) {
                if (k % 256 == 0) cancellation(cancel);
                acc += a.values[row * a.shape.cols + k] * b.values[k * b.shape.cols + col];
            }
            // Check before activation so ReLU cannot hide negative overflow.
            finite(acc);
            output.values[row * output.shape.cols + col] = activate ? std::max(0.0F, acc) : acc;
        }
    }
}
void tiled_product(const Tensor& a, const Tensor& b, Tensor& output, bool activate,
                   const Config& config, KernelStats& stats, const std::atomic<bool>* cancel) {
    const auto tile = config.tile_size;
    require(tile >= 1 && tile <= 64, "E_TILE: tile size must be in 1..64");
    const auto local_bytes = 3 * tile * tile * sizeof(float);
    require(local_bytes <= config.local_bytes, "E_LOCAL_MEMORY: three tile buffers exceed local byte budget");
    // Fixed square scratch buffers are actual storage, not a modeled-only bound.
    std::vector<float> local_a(tile * tile), local_b(tile * tile), local_c(tile * tile);
    stats.peak_local_bytes = std::max(stats.peak_local_bytes, local_bytes);
    for (std::size_t row = 0; row < a.shape.rows; row += tile) {
        const auto rows = std::min(tile, a.shape.rows - row);
        for (std::size_t col = 0; col < b.shape.cols; col += tile) {
            const auto columns = std::min(tile, b.shape.cols - col);
            std::fill(local_c.begin(), local_c.end(), 0.0F);
            for (std::size_t base = 0; base < a.shape.cols; base += tile) {
                cancellation(cancel);
                const auto inner = std::min(tile, a.shape.cols - base);
                for (std::size_t i = 0; i < rows; ++i)
                    for (std::size_t k = 0; k < inner; ++k)
                        local_a[i * tile + k] = a.values[(row + i) * a.shape.cols + base + k];
                for (std::size_t k = 0; k < inner; ++k)
                    for (std::size_t j = 0; j < columns; ++j)
                        local_b[k * tile + j] = b.values[(base + k) * b.shape.cols + col + j];
                ++stats.tile_count;
                stats.tile_load_bytes += (rows * inner + inner * columns) * sizeof(float);
                for (std::size_t i = 0; i < rows; ++i) {
                    cancellation(cancel);
                    for (std::size_t j = 0; j < columns; ++j)
                        for (std::size_t k = 0; k < inner; ++k)
                            local_c[i * tile + j] += local_a[i * tile + k] * local_b[k * tile + j];
                }
            }
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    const auto value = local_c[i * tile + j];
                    finite(value);
                    output.values[(row + i) * output.shape.cols + col + j] = activate ? std::max(0.0F, value) : value;
                }
            }
        }
    }
}
}
void compute_software(Op op, const std::vector<const Tensor*>& inputs, Tensor& output,
                      const Config& config, KernelStats& stats, const std::atomic<bool>* cancel) {
    cancellation(cancel);
    validate(op, inputs, output);
    require(config.tile_size <= 64, "E_TILE: tile size must be zero or in 1..64");
    const auto& a = *inputs[0];
    if (op == Op::MatMul || op == Op::MatMulRelu) {
        if (config.tile_size == 0) scalar_product(a, *inputs[1], output, op == Op::MatMulRelu, cancel);
        else tiled_product(a, *inputs[1], output, op == Op::MatMulRelu, config, stats, cancel);
    } else if (op == Op::RowSum) {
        for (std::size_t row = 0; row < a.shape.rows; ++row) {
            float acc = 0;
            for (std::size_t col = 0; col < a.shape.cols; ++col) {
                if (col % 256 == 0) cancellation(cancel);
                acc += a.values[row * a.shape.cols + col];
            }
            finite(acc);
            output.values[row] = acc;
        }
    } else {
        for (std::size_t i = 0; i < output.values.size(); ++i) {
            if (i % 256 == 0) cancellation(cancel);
            const auto value = op == Op::Relu ? std::max(0.0F, a.values[i]) : a.values[i] + inputs[1]->values[i];
            finite(value);
            output.values[i] = value;
        }
    }
    cancellation(cancel);
}
}
