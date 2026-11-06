#include "placementlens/device.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
int checks = 0;
void expect(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& action, const std::string& message) {
    try { action(); }
    catch (const std::exception&) { ++checks; return; }
    throw std::runtime_error("expected rejection: " + message);
}
pl::Command compute(pl::Op op, std::vector<std::string> inputs,
                    const std::string& output, pl::Shape shape) {
    pl::Command command;
    command.kind = pl::Kind::Compute;
    command.backend = pl::Backend::Device;
    command.op = op;
    command.inputs = std::move(inputs);
    command.output = output;
    command.shape = shape;
    return command;
}
void equal(const pl::Tensor& value, const std::vector<float>& expected) {
    expect(value.values.size() == expected.size(), "output element count");
    for (std::size_t i = 0; i < expected.size(); ++i)
        expect(std::isfinite(value.values[i]) && std::abs(value.values[i] - expected[i]) < 0.0001f, "kernel result mismatch");
}
}
int main() {
    std::string reason;
    if (!pl::opencl_available(reason)) {
        std::cout << "SKIP: " << reason << '\n';
        return 77;
    }
    try {
        auto device = pl::make_opencl_device();
        std::cout << device->description() << '\n';
        expect(device->diagnostic_readback_bytes() == 0, "diagnostic counter starts empty");
        rejects([&] { device->allocate("H:bad", {1, 1}); }, "host handle");
        rejects([&] { device->allocate("D:bad", {0, 1}); }, "zero shape");
        rejects([&] { device->release("D:missing"); }, "unknown release");
        device->allocate("D:a", {2, 3});
        device->allocate("D:b", {3, 2});
        device->allocate("D:product", {2, 2});
        device->allocate("D:relu", {2, 2});
        device->allocate("D:sum", {2, 2});
        device->allocate("D:fused", {2, 2});
        expect(device->last_elapsed_us() == -1, "allocation must not invent event timing");
        rejects([&] { device->allocate("D:a", {2, 3}); }, "duplicate allocation");
        pl::Tensor product{{2, 2}, std::vector<float>(4)};
        rejects([&] { device->read("D:product", product); }, "uninitialized read");
        const auto matmul = compute(pl::Op::MatMul, {"D:a", "D:b"}, "D:product", {2, 2});
        rejects([&] { device->compute(matmul); }, "uninitialized compute");
        rejects([&] { device->write("D:a", {{2, 3}, {1, 2}}); }, "short host storage");
        device->write("D:a", {{2, 3}, {-1, 2, 3, 4, -5, 6}});
        expect(device->last_elapsed_us() >= 0, "upload event profiling");
        device->write("D:b", {{3, 2}, {1, -2, 3, -4, 5, -6}});
        rejects([&] { device->write("D:a", {{2, 3}, {1, 2, 3, 4, 5, 6}}); }, "overwrite input");
        auto invalid = matmul;
        invalid.version = pl::command_version + 1;
        rejects([&] { device->compute(invalid); }, "version");
        invalid = matmul;
        invalid.shape = {1, 4};
        rejects([&] { device->compute(invalid); }, "descriptor shape");
        invalid = matmul;
        invalid.inputs.pop_back();
        rejects([&] { device->compute(invalid); }, "arity");
        device->compute(matmul);
        expect(device->last_elapsed_us() >= 0, "MatMul kernel profiling");
        device->read("D:product", product);
        equal(product, {20, -24, 19, -24});
        expect(device->diagnostic_readback_bytes() == 16, "one kernel diagnostic, explicit read excluded");
        expect(device->last_elapsed_us() >= 0, "download event profiling");
        rejects([&] { device->compute(matmul); }, "overwrite computed output");
        device->compute(compute(pl::Op::Relu, {"D:product"}, "D:relu", {2, 2}));
        expect(device->last_elapsed_us() >= 0, "ReLU kernel profiling");
        device->read("D:relu", product);
        equal(product, {20, 0, 19, 0});
        device->compute(compute(pl::Op::Add, {"D:product", "D:relu"}, "D:sum", {2, 2}));
        expect(device->last_elapsed_us() >= 0, "Add kernel profiling");
        device->read("D:sum", product);
        equal(product, {40, -24, 38, -24});
        device->compute(compute(pl::Op::MatMulRelu, {"D:a", "D:b"}, "D:fused", {2, 2}));
        expect(device->last_elapsed_us() >= 0, "fused kernel profiling");
        device->read("D:fused", product);
        equal(product, {20, 0, 19, 0});
        pl::Tensor wrong{{1, 4}, std::vector<float>(4)};
        rejects([&] { device->read("D:fused", wrong); }, "read shape mismatch");
        device->synchronize();
        expect(device->last_elapsed_us() == -1, "synchronize has no fabricated event");
        device->release("D:product");
        rejects([&] { device->read("D:product", product); }, "use after release");
        device->allocate("D:product", {2, 2});
        rejects([&] { device->read("D:product", product); }, "reallocation resets state");
        device->compute(matmul);
        device->read("D:product", product);
        equal(product, {20, -24, 19, -24});

        device->allocate("D:large", {1, 1});
        device->allocate("D:overflow", {1, 1});
        device->write("D:large", {{1, 1}, {std::numeric_limits<float>::max()}});
        expect(device->diagnostic_readback_bytes() == 80, "five diagnostics, no allocation/copy/invalid descriptor traffic counted");
        // Failure must happen at compute, even if the graph never reads this value.
        rejects([&] { device->compute(compute(pl::Op::Add, {"D:large", "D:large"}, "D:overflow", {1, 1})); }, "dead output overflow");
        expect(device->diagnostic_readback_bytes() == 84, "failed numeric validation still counts transferred bytes");
        device->allocate("D:negative", {1, 1});
        device->allocate("D:fused_overflow", {1, 1});
        device->write("D:negative", {{1, 1}, {-2}});
        rejects([&] { device->compute(compute(pl::Op::MatMulRelu, {"D:large", "D:negative"}, "D:fused_overflow", {1, 1})); }, "negative overflow before activation");
        expect(device->diagnostic_readback_bytes() == 88, "fused numeric failure counts diagnostic bytes");
        device->allocate("D:rows", {2, 1});
        device->compute(compute(pl::Op::RowSum, {"D:product"}, "D:rows", {2, 1}));
        expect(device->last_elapsed_us() >= 0, "RowSum hardware event");
        pl::Tensor rows{{2, 1}, std::vector<float>(2)};
        device->read("D:rows", rows);
        equal(rows, {-4, -5});
        expect(device->diagnostic_readback_bytes() == 96, "RowSum diagnostic bytes");
        // Real adapter through the common planner/executor, including preparation,
        // explicit FREE and same-instance cleanup across repeated graph runs.
        std::istringstream graph_input("tensor X f32 2 3 1 2 3 4 5 6\nrowsum sum X Y\noutput Y\n");
        const auto graph = pl::parse(graph_input);
        for (bool reuse : {false, true}) for (bool mirrored : {false, true}) {
            pl::Config config; config.reuse_buffers = reuse;
            if (mirrored) config.mirrored_inputs = {"X"};
            for (const auto& plan : pl::enumerate(graph, config).plans) {
                const auto run = pl::execute(graph, plan, config, device.get());
                expect(run.correct, "common executor hardware correctness");
                equal(run.output, {6, 15});
                expect(run.software_buffer_reuses == 0, "external adapter does not claim vector reuse");
                expect(run.preparation_h2d_bytes == (mirrored ? 24U : 0U), "real preparation accounting");
            }
        }
        std::cout << "PASS: " << checks << " real OpenCL kernel/lifecycle checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
