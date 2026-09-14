#include <torch/extension.h>
#include <cuda_runtime_api.h>

// CUDA can map these allocations even when Tensor.is_pinned() reports false
// on this deployment. A real alias is essential: UVA buffers are updated later.
torch::Tensor mapped_view(torch::Tensor input) {
  TORCH_CHECK(input.device().is_cpu(), "Expected a CPU tensor");
  void* device_ptr = nullptr;
  auto error = cudaHostGetDevicePointer(&device_ptr, input.data_ptr(), 0);
  TORCH_CHECK(error == cudaSuccess, "CPU allocation is not CUDA mapped: ",
              cudaGetErrorString(error));
  int device = 0;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "Cannot query CUDA device");
  return torch::from_blob(device_ptr, input.sizes(), input.strides(),
                         [owner = input](void*) {},
                         input.options().device(torch::Device(torch::kCUDA, device)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mapped_view", &mapped_view);
}
