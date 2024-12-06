#include <torch/extension.h>
#include "causal_conv1d.h"

void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                      const c10::optional<at::Tensor> &bias,
                      const c10::optional<at::Tensor> &conv_states,
                      const c10::optional<at::Tensor> &query_start_loc,
                      const c10::optional<at::Tensor> &cache_indices,
                      const c10::optional<at::Tensor> &has_initial_state,
                      bool silu_activation,
                      int64_t pad_slot_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv1d_fwd", &causal_conv1d_fwd,
          "Causal 1D convolution forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("conv_states") = py::none(),
          py::arg("query_start_loc") = py::none(),
          py::arg("cache_indices") = py::none(),
          py::arg("has_initial_state") = py::none(),
          py::arg("silu_activation") = false,
          py::arg("pad_slot_id") = -1);
} 