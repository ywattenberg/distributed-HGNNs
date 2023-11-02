#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API unbind_copy_int {
  using schema = ::std::vector<at::Tensor> (const at::Tensor &, int64_t);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::unbind_copy")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "int")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "unbind_copy.int(Tensor self, int dim=0) -> Tensor[]")
  static ::std::vector<at::Tensor> call(const at::Tensor & self, int64_t dim);
  static ::std::vector<at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim);
};

struct TORCH_API unbind_copy_int_out {
  using schema = void (const at::Tensor &, int64_t, at::TensorList);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::unbind_copy")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "int_out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "unbind_copy.int_out(Tensor self, int dim=0, *, Tensor(a!)[] out) -> ()")
  static void call(const at::Tensor & self, int64_t dim, at::TensorList out);
  static void redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::TensorList out);
};

}} // namespace at::_ops