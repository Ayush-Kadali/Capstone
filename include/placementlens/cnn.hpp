#pragma once
#include "placementlens/core.hpp"

namespace pl {
// Storage is contiguous rank-2 [N, C*H*W]; parameters retain NCHW semantics.
// Conv2D: N,C,H,W,M,KH,KW,SH,SW,PT,PL,PB,PR,GROUP,DH,DW.
// Pool2D: N,C,H,W,KH,KW,SH,SW,PT,PL,PB,PR,COUNT_INCLUDE_PAD.
// GlobalAveragePool: N,C,H,W. Reshape: output rows, output columns.
bool cnn_op(Op op);
bool unary_op(Op op);
Shape cnn_shape(Op op, const std::vector<Shape>& inputs, const std::vector<std::size_t>& parameters);
void compute_cnn(Op op, const std::vector<const Tensor*>& inputs, Tensor& output,
                 const std::vector<std::size_t>& parameters,
                 const std::atomic<bool>* cancel = nullptr);
Tensor reference_cnn(Op op, const std::vector<const Tensor*>& inputs,
                     const std::vector<std::size_t>& parameters,
                     const std::atomic<bool>* cancel = nullptr);
}
