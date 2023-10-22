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



#include <ATen/ops/randint_ops.h>

namespace at {


// aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t high, at::IntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint::call(high, c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t high, at::IntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint::call(high, c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint::call(high, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint::call(high, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
  }
}

// aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint::call(high, size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint::call(high, size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint::call(high, size, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint::call(high, size, dtype, layout, device, pin_memory);
  }
}

// aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_generator::call(high, c10::fromIntArrayRefSlow(size), generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_generator::call(high, c10::fromIntArrayRefSlow(size), generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_generator::call(high, c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_generator::call(high, c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
  }
}

// aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_generator::call(high, size, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_generator::call(high, size, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_generator::call(high, size, generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_generator::call(high, size, generator, dtype, layout, device, pin_memory);
  }
}

// aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low::call(low, high, c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low::call(low, high, c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low::call(low, high, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low::call(low, high, c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
  }
}

// aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low::call(low, high, size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low::call(low, high, size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low::call(low, high, size, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low::call(low, high, size, dtype, layout, device, pin_memory);
  }
}

// aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low_generator::call(low, high, c10::fromIntArrayRefSlow(size), generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low_generator::call(low, high, c10::fromIntArrayRefSlow(size), generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low_generator::call(low, high, c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor randint(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low_generator::call(low, high, c10::fromIntArrayRefSlow(size), generator, dtype, layout, device, pin_memory);
  }
}

// aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low_generator::call(low, high, size, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options=at::kLong) {
    return at::_ops::randint_low_generator::call(low, high, size, generator, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor randint_symint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low_generator::call(low, high, size, generator, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor randint(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::randint_low_generator::call(low, high, size, generator, dtype, layout, device, pin_memory);
  }
}

// aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_out(at::Tensor & out, int64_t high, at::IntArrayRef size) {
    return at::_ops::randint_out::call(high, c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_out(at::Tensor & out, int64_t high, at::IntArrayRef size) {
    return at::_ops::randint_out::call(high, c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_outf(int64_t high, at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_out::call(high, c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_outf(int64_t high, at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_out::call(high, c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_out(at::Tensor & out, c10::SymInt high, c10::SymIntArrayRef size) {
    return at::_ops::randint_out::call(high, size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_out(at::Tensor & out, c10::SymInt high, c10::SymIntArrayRef size) {
    return at::_ops::randint_out::call(high, size, out);
  }
}

// aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_outf(c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_out::call(high, size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_outf(c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_out::call(high, size, out);
  }
}

// aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_out(at::Tensor & out, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_generator_out::call(high, c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_out(at::Tensor & out, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_generator_out::call(high, c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_outf(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_generator_out::call(high, c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_outf(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_generator_out::call(high, c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_out(at::Tensor & out, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_generator_out::call(high, size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_out(at::Tensor & out, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_generator_out::call(high, size, generator, out);
  }
}

// aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_outf(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_generator_out::call(high, size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_outf(c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_generator_out::call(high, size, generator, out);
  }
}

// aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_out(at::Tensor & out, int64_t low, int64_t high, at::IntArrayRef size) {
    return at::_ops::randint_low_out::call(low, high, c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_out(at::Tensor & out, int64_t low, int64_t high, at::IntArrayRef size) {
    return at::_ops::randint_low_out::call(low, high, c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_outf(int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_low_out::call(low, high, c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_outf(int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_low_out::call(low, high, c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_out(at::Tensor & out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size) {
    return at::_ops::randint_low_out::call(low, high, size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_out(at::Tensor & out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size) {
    return at::_ops::randint_low_out::call(low, high, size, out);
  }
}

// aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_outf(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_low_out::call(low, high, size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_outf(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::randint_low_out::call(low, high, size, out);
  }
}

// aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_out(at::Tensor & out, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_low_generator_out::call(low, high, c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_out(at::Tensor & out, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_low_generator_out::call(low, high, c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_outf(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_low_generator_out::call(low, high, c10::fromIntArrayRefSlow(size), generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & randint_outf(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_low_generator_out::call(low, high, c10::fromIntArrayRefSlow(size), generator, out);
  }
}

// aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_out(at::Tensor & out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_low_generator_out::call(low, high, size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_out(at::Tensor & out, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator) {
    return at::_ops::randint_low_generator_out::call(low, high, size, generator, out);
  }
}

// aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & randint_symint_outf(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_low_generator_out::call(low, high, size, generator, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & randint_outf(c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::randint_low_generator_out::call(low, high, size, generator, out);
  }
}

}
