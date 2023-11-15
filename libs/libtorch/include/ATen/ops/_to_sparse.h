#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_to_sparse_ops.h>

namespace at {


// aten::_to_sparse.sparse_dim_out(Tensor self, int sparse_dim, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _to_sparse_out(at::Tensor & out, const at::Tensor & self, int64_t sparse_dim) {
    return at::_ops::_to_sparse_sparse_dim_out::call(self, sparse_dim, out);
}
// aten::_to_sparse.sparse_dim_out(Tensor self, int sparse_dim, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _to_sparse_outf(const at::Tensor & self, int64_t sparse_dim, at::Tensor & out) {
    return at::_ops::_to_sparse_sparse_dim_out::call(self, sparse_dim, out);
}

// aten::_to_sparse.out(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _to_sparse_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::Layout> layout=c10::nullopt, at::OptionalIntArrayRef blocksize=c10::nullopt, c10::optional<int64_t> dense_dim=c10::nullopt) {
    return at::_ops::_to_sparse_out::call(self, layout, blocksize, dense_dim, out);
}
// aten::_to_sparse.out(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _to_sparse_outf(const at::Tensor & self, c10::optional<at::Layout> layout, at::OptionalIntArrayRef blocksize, c10::optional<int64_t> dense_dim, at::Tensor & out) {
    return at::_ops::_to_sparse_out::call(self, layout, blocksize, dense_dim, out);
}

}