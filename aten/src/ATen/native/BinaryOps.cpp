#include <type_traits>
#include <ATen/native/BinaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(tanh_backward_stub);

Tensor& add_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  alpha_check(iter.dtype(), alpha);
  add_stub(iter.device_type(), iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor add(const Tensor& self, const Tensor& other, Scalar alpha) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  add_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::add_out(self, self, other, alpha);
}

Tensor& div_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  div_stub(iter.device_type(), iter);
  return result;
}

Tensor div(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& div_(Tensor& self, const Tensor& other) {
  return native::div_out(self, self, other);
}

Tensor truncate(const Tensor& tensor) {
  if (tensor.is_floating_point()) {
    return tensor.trunc();
  }
  return tensor;
}

Tensor floor_divide(const Tensor& input, const Tensor& other) {
  Tensor out = input / other;
  return truncate(out);
}

Tensor floor_divide(const Tensor& input, Scalar other) {
  Tensor out = input / other;
  return truncate(out);
}

Tensor& mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  mul_stub(iter.device_type(), iter);
  return result;
}

Tensor mul(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  mul_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& mul_(Tensor& self, const Tensor& other) {
  return native::mul_out(self, self, other);
}

Tensor& sub_out(Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == iter.output().dtype());
  return result;
}

Tensor sub(const Tensor& self, const Tensor& other, Scalar alpha) {
  sub_check(self, other);
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  alpha_check(iter.dtype(), alpha);
  sub_stub(iter.device_type(), iter, alpha);
  return iter.output();
}

Tensor& sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub_out(self, self, other, alpha);
}

Tensor& sigmoid_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  sigmoid_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& tanh_backward_out(Tensor& result, const Tensor& grad_output, const Tensor& output) {
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return result;
}

Tensor tanh_backward(const Tensor& grad_output, const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  tanh_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor rsub(const Tensor& self, const Tensor& other, Scalar alpha) {
  return native::sub(other, self, alpha);
}

Tensor& atan2_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other);
  atan2_stub(iter.device_type(), iter);
  return result;
}

Tensor atan2(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  return native::atan2_out(result, self, other);
}

Tensor& atan2_(Tensor& self, const Tensor& other) {
  return native::atan2_out(self, self, other);
}

static Tensor wrapped_scalar_tensor(Scalar scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

static void check_convert(Scalar scalar, ScalarType scalarType) {
  // Validate that is possible to convert scalar to tensor dtype without overflow
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, scalarType, "check_convert", [&]{
    scalar.to<scalar_t>();
  });
}

static Tensor wrapped_scalar_tensor_and_check_convert(Scalar scalar, Tensor tensor) {
  check_convert(scalar, tensor.scalar_type());
  return wrapped_scalar_tensor(scalar);
}

Tensor add(const Tensor& self, Scalar other, Scalar alpha) {
  return native::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor add(Scalar other, const Tensor& self, Scalar alpha) {
  return native::add(wrapped_scalar_tensor(other), self, alpha);
}

Tensor& add_(Tensor& self, Scalar other, Scalar alpha) {
  return native::add_(self, wrapped_scalar_tensor(other), alpha);
}

// WARNING: There doesn't appear to be any testing for this function
// with sparse self input.
Tensor div(const Tensor& self, Scalar other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor div(Scalar other, const Tensor& self) {
  return wrapped_scalar_tensor(other).div(self); // redispatch!
}

// WARNING: This function, with a sparse self, is currently only
// exercised by DistributedDataParallelTest.test_sparse_gradients
Tensor& div_(Tensor& self, Scalar other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor mul(const Tensor& self, Scalar other) {
  return native::mul(self, wrapped_scalar_tensor(other));
}

Tensor mul(Scalar other, const Tensor& self) {
  return native::mul(self, other);
}

Tensor& mul_(Tensor& self, Scalar other) {
  return native::mul_(self, wrapped_scalar_tensor(other));
}

Tensor sub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::sub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor sub(Scalar other, const Tensor& self, Scalar alpha) {
  return native::sub(wrapped_scalar_tensor(other), self, alpha);
}

Tensor& sub_(Tensor& self, Scalar other, Scalar alpha) {
  return native::sub_(self, wrapped_scalar_tensor(other), alpha);
}

Tensor rsub(const Tensor& self, Scalar other, Scalar alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_and_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_and(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_and_out(result, self, other);
  return result;
}

Tensor& bitwise_and_(Tensor& self, const Tensor& other) {
  return at::bitwise_and_out(self, self, other);
}

Tensor& bitwise_and_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_and_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_and(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_and_out(result, self, other);
}

Tensor& bitwise_and_(Tensor& self, Scalar other) {
  return at::bitwise_and_out(self, self, other);
}

// Legacy and interfaces. They are aliased to bitwise_and* functions
Tensor __and__(const Tensor& self, const Tensor& other) {
  return at::bitwise_and(self, other);
}

Tensor __and__(const Tensor& self, Scalar other) {
  return at::bitwise_and(self, other);
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  return self.bitwise_and_(other);
}

Tensor& __iand__(Tensor& self, Scalar other) {
  return self.bitwise_and_(other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_or_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_or(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_or_out(result, self, other);
  return result;
}

Tensor& bitwise_or_(Tensor& self, const Tensor& other) {
  return at::bitwise_or_out(self, self, other);
}

Tensor& bitwise_or_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_or_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_or(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_or_out(result, self, other);
}

Tensor& bitwise_or_(Tensor& self, Scalar other) {
  return at::bitwise_or_out(self, self, other);
}

// Legacy or interfaces. They are aliased to bitwise_or* functions
Tensor __or__(const Tensor& self, const Tensor& other) {
  return at::bitwise_or(self, other);
}

Tensor __or__(const Tensor& self, Scalar other) {
  return at::bitwise_or(self, other);
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  return self.bitwise_or_(other);
}

Tensor& __ior__(Tensor& self, Scalar other) {
  return self.bitwise_or_(other);
}

Tensor rsub(Scalar other, const Tensor& self, Scalar alpha) {
  return native::rsub(wrapped_scalar_tensor(other), self, alpha);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(result, self, other,
    /*check_mem_overlap=*/true);
  bitwise_xor_stub(iter.device_type(), iter);
  return result;
}

Tensor bitwise_xor(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  at::bitwise_xor_out(result, self, other);
  return result;
}

Tensor& bitwise_xor_(Tensor& self, const Tensor& other) {
  return at::bitwise_xor_out(self, self, other);
}

Tensor& bitwise_xor_out(Tensor& result, const Tensor& self, Scalar other) {
  return at::bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_xor(const Tensor& self, Scalar other) {
  Tensor result = at::empty({0}, self.options());
  return at::bitwise_xor_out(result, self, other);
}

Tensor& bitwise_xor_(Tensor& self, Scalar other) {
  return at::bitwise_xor_out(self, self, other);
}

// Legacy xor interfaces. They are aliased to bitwise_xor* functions
Tensor __xor__(const Tensor& self, const Tensor& other) {
  return at::bitwise_xor(self, other);
}

Tensor __xor__(const Tensor& self, Scalar other) {
  return at::bitwise_xor(self, other);
}

Tensor& __ixor__(Tensor& self, const Tensor& other) {
  return self.bitwise_xor_(other);
}

Tensor& __ixor__(Tensor& self, Scalar other) {
  return self.bitwise_xor_(other);
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __lshift__(const Tensor& self, Scalar other) { 
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __ilshift__(Tensor& self, Scalar other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __rshift__(const Tensor& self, Scalar other) { 
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  rshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __irshift__(Tensor& self, Scalar other) {
  auto wrapper = wrapped_scalar_tensor(other).toType(self.scalar_type());
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return self;
}

template <typename Stub>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  // Validate that is possible to convert zero-dim tensor's dtype to other dtype without overflow
  if (self.scalar_type() != other.scalar_type()) {
    if (self.dim() != 0 && other.dim() == 0) {
      check_convert(other.item(), self.scalar_type());
    } else if (self.dim() == 0 && other.dim() != 0) {
      check_convert(self.item(), other.scalar_type());
    }
  }
  auto iter = TensorIterator::comparison_op(result, self, other, /*check_mem_overlap=*/true);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

// To avoid overflow during type promotion we will check that both dtypes of self and other are same
template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  TORCH_CHECK(self.dtype() == other.dtype(),
              "Expected object of scalar type ", self.dtype(), " but got scalar type ",
              other.dtype(), " for argument 'other'");
  return out_impl(self, self, other);
}

// validates that is possible to convert Scalar other to self's dtype without overflow.
// This behavior is unique to comparison ops; arithmetic operations don't do this.
// In the future, we should reconsider this inconsistency and decide if we want to add the same check to arithmetic ops.
template <typename OutImpl>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, Scalar other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, Scalar other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor_and_check_convert(other, self), out_impl);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, Scalar other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor_and_check_convert(other, self));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguious.
using OutFunc = std::add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;

Tensor& lt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, lt_stub); }
Tensor lt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor lt(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::lt_out)); }
Tensor& lt_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::lt_out)); }

Tensor& le_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, le_stub); }
Tensor le(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::le_out)); }
Tensor le(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::le_out)); }
Tensor& le_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::le_out)); }

Tensor& gt_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, gt_stub); }
Tensor gt(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor gt(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::gt_out)); }
Tensor& gt_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::gt_out)); }

Tensor& ge_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ge_stub); }
Tensor ge(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor ge(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::ge_out)); }
Tensor& ge_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ge_out)); }

Tensor& eq_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, eq_stub); }
Tensor eq(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor eq(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::eq_out)); }
Tensor& eq_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::eq_out)); }

Tensor& ne_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, ne_stub); }
Tensor ne(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor ne(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::ne_out)); }
Tensor& ne_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::ne_out)); }

Tensor& logical_and_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_and_stub); }
Tensor logical_and(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor logical_and(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }

Tensor& logical_or_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_or_stub); }
Tensor logical_or(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor logical_or(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }

Tensor& logical_xor_out(Tensor& result, const Tensor& self, const Tensor& other) { return comparison_op_out(result, self, other, logical_xor_stub); }
Tensor logical_xor(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_out(Tensor& result, const Tensor& self, Scalar other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor logical_xor(const Tensor& self, Scalar other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, Scalar other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
}
}  // namespace at
