#include "placementlens/cnn.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pl {
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void cancelled(const std::atomic<bool>* token) {
    require(!token || !token->load(std::memory_order_relaxed), "E_CANCELLED: CNN execution cancelled");
}
std::size_t product(std::initializer_list<std::size_t> xs) {
    std::size_t value = 1;
    for (auto x : xs) {
        require(x > 0 && x <= max_elements / value, "E_SHAPE: CNN tensor exceeds bounded element count");
        value *= x;
    }
    return value;
}
std::size_t extent(std::size_t input, std::size_t kernel, std::size_t dilation,
                   std::size_t before, std::size_t after, std::size_t stride) {
    require(kernel > 0 && dilation > 0 && stride > 0, "E_SHAPE: kernel/dilation/stride must be positive");
    require(kernel - 1 <= (max_elements - 1) / dilation, "E_SHAPE: effective kernel exceeds bound");
    const auto effective = (kernel - 1) * dilation + 1;
    require(before < effective && after < effective && input + before + after >= effective,
            "E_SHAPE: unsupported CNN padding or empty output");
    return (input + before + after - effective) / stride + 1;
}
void storage(const std::vector<const Tensor*>& inputs, const Tensor& output) {
    require(output.values.size() == output.shape.elements(), "E_VALUE: CNN output storage mismatch");
    for (const auto* input : inputs) {
        require(input && input != &output, "E_BUFFER: CNN missing or aliased input");
        require(input->values.size() == input->shape.elements(), "E_VALUE: CNN input storage mismatch");
        for (auto x : input->values) require(std::isfinite(x), "E_VALUE: CNN nonfinite input");
    }
}
// Accumulation precision differs between functional kernels and the internal
// reference. The independent semantic oracle is the original model in ORT.
template<class Acc> void evaluate(Op op, const std::vector<const Tensor*>& inputs,
                                 Tensor& output, const std::vector<std::size_t>& p,
                                 const std::atomic<bool>* token) {
    std::vector<Shape> shapes;
    for (const auto* t : inputs) { require(t != nullptr, "E_BUFFER: null CNN input"); shapes.push_back(t->shape); }
    require(cnn_shape(op, shapes, p) == output.shape, "E_SHAPE: CNN output shape mismatch");
    storage(inputs, output);
    const auto& x = inputs[0]->values;
    if (op == Op::Reshape) {
        for (std::size_t i = 0; i < x.size(); ++i) { if (i % 256 == 0) cancelled(token); output.values[i] = x[i]; }
        return;
    }
    const auto n = p[0], channels = p[1], h = p[2], w = p[3];
    const auto out_channels = op == Op::Conv2D ? p[4] : channels;
    const auto kh = op == Op::Conv2D ? p[5] : op == Op::GlobalAveragePool ? h : p[4];
    const auto kw = op == Op::Conv2D ? p[6] : op == Op::GlobalAveragePool ? w : p[5];
    const auto sh = op == Op::Conv2D ? p[7] : op == Op::GlobalAveragePool ? h : p[6];
    const auto sw = op == Op::Conv2D ? p[8] : op == Op::GlobalAveragePool ? w : p[7];
    const auto pt = op == Op::GlobalAveragePool ? 0 : p[op == Op::Conv2D ? 9 : 8];
    const auto pl = op == Op::GlobalAveragePool ? 0 : p[op == Op::Conv2D ? 10 : 9];
    const auto pb = op == Op::GlobalAveragePool ? 0 : p[op == Op::Conv2D ? 11 : 10];
    const auto pr = op == Op::GlobalAveragePool ? 0 : p[op == Op::Conv2D ? 12 : 11];
    const auto dh = op == Op::Conv2D ? p[14] : 1, dw = op == Op::Conv2D ? p[15] : 1;
    const auto oh = extent(h, kh, dh, pt, pb, sh), ow = extent(w, kw, dw, pl, pr, sw);
    const auto groups = op == Op::Conv2D ? p[13] : channels;
    const auto group_channels = channels / groups;
    for (std::size_t batch = 0; batch < n; ++batch)
        for (std::size_t oc = 0; oc < out_channels; ++oc)
            for (std::size_t y = 0; y < oh; ++y)
                for (std::size_t z = 0; z < ow; ++z) {
                    cancelled(token);
                    Acc value = op == Op::MaxPool2D ? -std::numeric_limits<Acc>::infinity() : Acc{0};
                    std::size_t valid_count = 0;
                    const auto first_channel = (oc / (out_channels / groups)) * group_channels;
                    const auto count = op == Op::Conv2D ? group_channels : 1;
                    for (std::size_t c = 0; c < count; ++c)
                        for (std::size_t ky = 0; ky < kh; ++ky) {
                            cancelled(token);
                            for (std::size_t kx = 0; kx < kw; ++kx) {
                                const auto iy = static_cast<long long>(y * sh + ky * dh) - static_cast<long long>(pt);
                                const auto ix = static_cast<long long>(z * sw + kx * dw) - static_cast<long long>(pl);
                                if (iy < 0 || ix < 0 || iy >= static_cast<long long>(h) || ix >= static_cast<long long>(w)) continue;
                                const auto ic = op == Op::Conv2D ? first_channel + c : oc;
                                const auto index = ((batch * channels + ic) * h + static_cast<std::size_t>(iy)) * w + static_cast<std::size_t>(ix);
                                const Acc element = static_cast<Acc>(x[index]);
                                if (op == Op::Conv2D) {
                                    const auto weight = ((oc * group_channels + c) * kh + ky) * kw + kx;
                                    value += element * static_cast<Acc>(inputs[1]->values[weight]);
                                } else if (op == Op::MaxPool2D) value = std::max(value, element);
                                else value += element;
                                ++valid_count;
                            }
                        }
                    if (op == Op::AveragePool2D || op == Op::GlobalAveragePool) {
                        const auto denominator = op == Op::AveragePool2D && p[12] ? kh * kw : valid_count;
                        require(denominator > 0, "E_SHAPE: empty average-pooling window");
                        value /= static_cast<Acc>(denominator);
                    }
                    const auto result = static_cast<float>(value);
                    require(std::isfinite(result), "E_NUMERIC: CNN computation produced nonfinite output");
                    output.values[((batch * out_channels + oc) * oh + y) * ow + z] = result;
                }
    cancelled(token);
}
}

bool cnn_op(Op op) {
    return op == Op::Conv2D || op == Op::MaxPool2D || op == Op::AveragePool2D ||
           op == Op::GlobalAveragePool || op == Op::Reshape;
}
bool unary_op(Op op) { return op == Op::Relu || op == Op::RowSum || (cnn_op(op) && op != Op::Conv2D); }
Shape cnn_shape(Op op, const std::vector<Shape>& inputs, const std::vector<std::size_t>& p) {
    require(cnn_op(op), "E_OPCODE: not a CNN/storage operation");
    require(inputs.size() == (op == Op::Conv2D ? 2U : 1U), "E_ARITY: CNN operand count");
    for (auto v : p) require(v <= max_elements, "E_SHAPE: CNN parameter exceeds bound");
    const auto count = op == Op::Conv2D ? 16U : op == Op::GlobalAveragePool ? 4U : op == Op::Reshape ? 2U : 13U;
    require(p.size() == count, "E_ATTRIBUTE: invalid CNN parameter count");
    if (op == Op::Reshape) {
        Shape shape{p[0], p[1]};
        require(shape.elements() == inputs[0].elements(), "E_SHAPE: reshape changes element count");
        return shape;
    }
    const auto n = p[0], c = p[1], h = p[2], w = p[3];
    require(inputs[0] == Shape{n, product({c,h,w})}, "E_SHAPE: packed NCHW input mismatch");
    (void)inputs[0].elements();
    if (op == Op::GlobalAveragePool) { Shape shape{n,c}; (void)shape.elements(); return shape; }
    const auto kh = p[op == Op::Conv2D ? 5 : 4], kw = p[op == Op::Conv2D ? 6 : 5];
    (void)product({kh,kw});
    const auto sh = p[op == Op::Conv2D ? 7 : 6], sw = p[op == Op::Conv2D ? 8 : 7];
    const auto pt = p[op == Op::Conv2D ? 9 : 8], pl = p[op == Op::Conv2D ? 10 : 9];
    const auto pb = p[op == Op::Conv2D ? 11 : 10], pr = p[op == Op::Conv2D ? 12 : 11];
    const auto dh = op == Op::Conv2D ? p[14] : 1, dw = op == Op::Conv2D ? p[15] : 1;
    std::size_t out_channels = c;
    if (op == Op::Conv2D) {
        const auto groups = p[13]; out_channels = p[4];
        require(groups > 0 && c % groups == 0 && out_channels > 0 && out_channels % groups == 0,
                "E_SHAPE: invalid convolution groups");
        require(inputs[1] == Shape{out_channels, product({c/groups,kh,kw})}, "E_SHAPE: packed convolution weights mismatch");
        (void)inputs[1].elements();
    } else require(p[12] <= 1 && (op != Op::MaxPool2D || p[12] == 0), "E_ATTRIBUTE: invalid pool count_include_pad");
    const auto oh = extent(h,kh,dh,pt,pb,sh), ow = extent(w,kw,dw,pl,pr,sw);
    Shape shape{n, product({out_channels,oh,ow})};
    const auto output_elements = shape.elements();
    const auto visits = op == Op::Conv2D ? inputs[1].cols : product({kh,kw});
    require(visits <= 64000000 / output_elements, "E_SHAPE: CNN exceeds 64 million kernel visits");
    return shape;
}
void compute_cnn(Op op, const std::vector<const Tensor*>& inputs, Tensor& output,
                 const std::vector<std::size_t>& parameters, const std::atomic<bool>* cancel) {
    evaluate<float>(op, inputs, output, parameters, cancel);
}
Tensor reference_cnn(Op op, const std::vector<const Tensor*>& inputs,
                     const std::vector<std::size_t>& parameters, const std::atomic<bool>* cancel) {
    std::vector<Shape> shapes;
    for (const auto* input : inputs) { require(input != nullptr, "E_BUFFER: null reference input"); shapes.push_back(input->shape); }
    Tensor output{cnn_shape(op, shapes, parameters), {}};
    output.values.resize(output.shape.elements());
    evaluate<double>(op, inputs, output, parameters, cancel);
    return output;
}
}
