#include "placementlens/kernels.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>

namespace {
int checks = 0;
void check(bool condition, const char* label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}
void rejects(const std::function<void()>& action, const char* code) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find(code) != std::string::npos; }
    check(rejected, code);
}
pl::Tensor tensor(std::size_t rows, std::size_t cols) {
    return {{rows, cols}, std::vector<float>(rows * cols)};
}
std::vector<float> expected_product(const pl::Tensor& a, const pl::Tensor& b, bool activate) {
    std::vector<float> expected;
    for (std::size_t row = 0; row < a.shape.rows; ++row) {
        for (std::size_t col = 0; col < b.shape.cols; ++col) {
            double sum = 0;
            for (std::size_t k = 0; k < a.shape.cols; ++k)
                sum += static_cast<double>(a.values[row * a.shape.cols + k]) * b.values[k * b.shape.cols + col];
            expected.push_back(static_cast<float>(activate && sum < 0 ? 0 : sum));
        }
    }
    return expected;
}
std::size_t blocks(std::size_t extent, std::size_t tile) { return (extent + tile - 1) / tile; }
void rectangular_tiles() {
    pl::Config config;
    for (std::size_t m = 1; m <= 5; ++m) {
        for (std::size_t k = 1; k <= 7; ++k) {
            for (std::size_t n = 1; n <= 5; ++n) {
                auto a = tensor(m,k), b = tensor(k,n);
                for (std::size_t i = 0; i < a.values.size(); ++i) a.values[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
                for (std::size_t i = 0; i < b.values.size(); ++i) b.values[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
                for (const auto tile : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{8}}) {
                    config.tile_size = tile;
                    for (const auto op : {pl::Op::MatMul, pl::Op::MatMulRelu}) {
                        auto out = tensor(m,n);
                        auto* const storage = out.values.data();
                        const auto capacity = out.values.capacity();
                        pl::KernelStats stats;
                        pl::compute_software(op, {&a,&b}, out, config, stats);
                        check(out.values == expected_product(a,b,op == pl::Op::MatMulRelu), "rectangular/edge tile agrees with independent double calculation");
                        check(out.values.data() == storage && out.values.capacity() == capacity, "kernel preserves pooled output allocation");
                        if (tile == 0) {
                            check(stats.tile_count == 0 && stats.tile_load_bytes == 0 && stats.peak_local_bytes == 0, "scalar product uses no tile scratch");
                        } else {
                            check(stats.tile_count == blocks(m,tile)*blocks(n,tile)*blocks(k,tile), "exact number of multiply tiles");
                            check(stats.tile_load_bytes == (m*k*blocks(n,tile)+k*n*blocks(m,tile))*sizeof(float), "edge tile traffic counts only actual A/B loads");
                            check(stats.peak_local_bytes == 3*tile*tile*sizeof(float), "fixed three-buffer scratch capacity");
                        }
                    }
                }
            }
        }
    }
}
void hand_cases_and_reduction() {
    pl::Tensor a{{2,3},{1,-2,3,0,4,-1}}, b{{3,2},{2,1,1,-3,-2,2}};
    auto output = tensor(2,2);
    pl::Config config; config.tile_size=2;
    pl::KernelStats stats;
    pl::compute_software(pl::Op::MatMul,{&a,&b},output,config,stats);
    check(output.values == std::vector<float>({-6,13,6,-14}), "hand-computed MatMul");
    check(stats.tile_count == 2 && stats.tile_load_bytes == 48 && stats.peak_local_bytes == 48, "hand-counted edge tile bytes and scratch");
    pl::compute_software(pl::Op::MatMulRelu,{&a,&b},output,config,stats);
    check(output.values == std::vector<float>({0,13,6,0}), "hand-computed fused activation");
    check(stats.tile_count == 4 && stats.tile_load_bytes == 96 && stats.peak_local_bytes == 48, "statistics sum traffic/count but maximize scratch");
    auto rows = tensor(2,1);
    pl::compute_software(pl::Op::RowSum,{&a},rows,config,stats);
    check(rows.values == std::vector<float>({2,3}), "hand-computed RowSum");
    check(stats.tile_count == 4 && stats.tile_load_bytes == 96 && stats.peak_local_bytes == 48, "RowSum does not claim tile scratch");
    for (std::size_t columns : {std::size_t{1},std::size_t{3},std::size_t{257},std::size_t{1024}}) {
        auto values = tensor(3,columns), result = tensor(3,1);
        for (std::size_t i=0;i<values.values.size();++i) values.values[i] = static_cast<float>(static_cast<int>(i%9)-4);
        pl::compute_software(pl::Op::RowSum,{&values},result,config,stats);
        for (std::size_t row=0;row<3;++row) {
            double expected=0;
            for (std::size_t col=0;col<columns;++col) expected+=values.values[row*columns+col];
            check(result.values[row] == static_cast<float>(expected), "reduction agrees with independent double row accumulation");
        }
    }
    pl::Tensor cancellation_order{{1,3},{100000000,1,-100000000}}, ones{{3,1},{1,1,1}};
    auto scalar = tensor(1,1), tiled = tensor(1,1);
    config.tile_size=0;
    pl::compute_software(pl::Op::MatMul,{&cancellation_order,&ones},scalar,config,stats);
    config.tile_size=2;
    pl::compute_software(pl::Op::MatMul,{&cancellation_order,&ones},tiled,config,stats);
    check(scalar.values[0] == 0 && tiled.values == scalar.values, "tiles preserve f32 k order across boundaries");
    pl::compute_software(pl::Op::RowSum,{&cancellation_order},scalar,config,stats);
    check(scalar.values[0] == 0, "RowSum uses declared f32 accumulation");
    pl::Tensor x{{1,3},{-1,2,-3}}, y{{1,3},{4,5,6}};
    auto elementwise = tensor(1,3);
    pl::compute_software(pl::Op::Add,{&x,&y},elementwise,config,stats);
    check(elementwise.values == std::vector<float>({3,7,3}), "Add helper");
    pl::compute_software(pl::Op::Relu,{&x},elementwise,config,stats);
    check(elementwise.values == std::vector<float>({0,2,0}), "ReLU helper");
}
void bounds_failures_and_cancellation() {
    pl::Tensor a{{1,1},{2}}, b{{1,1},{3}};
    auto out=tensor(1,1); pl::Config config; pl::KernelStats stats;
    config.tile_size=65;
    rejects([&]{pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);},"E_TILE");
    config.tile_size=std::numeric_limits<std::size_t>::max();
    rejects([&]{pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);},"E_TILE");
    config.tile_size=2; config.local_bytes=47;
    rejects([&]{pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);},"E_LOCAL_MEMORY");
    config.local_bytes=48;
    pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);
    check(out.values[0] == 6 && stats.peak_local_bytes==48, "exact scratch budget accepted even for edge-only tile");
    config.tile_size=64; config.local_bytes=49152;
    pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);
    check(stats.peak_local_bytes==49152, "maximum tile size is bounded and supported");
    config.tile_size=0; config.local_bytes=0;
    pl::compute_software(pl::Op::MatMul,{&a,&b},out,config,stats);
    check(out.values[0]==6, "scalar path needs no local scratch budget");
    rejects([&]{pl::compute_software(pl::Op::Add,{&a},out,config,stats);},"E_ARITY");
    rejects([&]{pl::compute_software(pl::Op::Relu,{nullptr},out,config,stats);},"E_BUFFER");
    rejects([&]{pl::compute_software(pl::Op::Relu,{&out},out,config,stats);},"E_BUFFER");
    auto malformed=out; malformed.values.clear();
    rejects([&]{pl::compute_software(pl::Op::Relu,{&malformed},out,config,stats);},"E_VALUE");
    rejects([&]{pl::compute_software(pl::Op::Relu,{&a},malformed,config,stats);},"E_VALUE");
    auto wrong=tensor(1,2);
    rejects([&]{pl::compute_software(pl::Op::RowSum,{&a},wrong,config,stats);},"E_SHAPE");
    rejects([&]{pl::compute_software(static_cast<pl::Op>(99),{&a},out,config,stats);},"E_OPCODE");
    pl::Tensor huge{{1,2},{3e38F,3e38F}};
    rejects([&]{pl::compute_software(pl::Op::RowSum,{&huge},out,config,stats);},"E_NUMERIC");
    pl::Tensor overflow{{1,1},{3e38F}}, negative{{1,1},{-2}};
    for (const auto tile : {std::size_t{0},std::size_t{1},std::size_t{2}}) {
        config.tile_size=tile; config.local_bytes=32768;
        rejects([&]{pl::compute_software(pl::Op::MatMulRelu,{&overflow,&negative},out,config,stats);},"E_NUMERIC");
    }
    auto infinite=a; infinite.values[0]=std::numeric_limits<float>::infinity();
    rejects([&]{pl::compute_software(pl::Op::Relu,{&infinite},out,config,stats);},"E_VALUE");
    std::atomic<bool> cancel{true};
    out.values[0]=123;
    for (const auto op : {pl::Op::MatMul,pl::Op::MatMulRelu,pl::Op::Add,pl::Op::RowSum,pl::Op::Relu}) {
        const bool unary=op==pl::Op::Relu || op==pl::Op::RowSum;
        rejects([&]{pl::compute_software(op,unary?std::vector<const pl::Tensor*>{&a}:std::vector<const pl::Tensor*>{&a,&b},out,config,stats,&cancel);},"E_CANCELLED");
        check(out.values[0]==123,"pre-cancel does not modify output");
    }
}
}
int main() {
    try {
        rectangular_tiles(); hand_cases_and_reduction(); bounds_failures_and_cancellation();
        std::cout << "PASS " << checks << " kernel assertions\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
