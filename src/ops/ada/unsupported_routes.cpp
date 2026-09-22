#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error("This SM120-only operator route is unavailable in the SM89 build");
}
} // namespace

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                  Tensor&, Nvfp4W4a4Workspace, cudaStream_t) { unavailable(); }
void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) { unavailable(); }
void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace,
                                cudaStream_t) { unavailable(); }
void launch_nvfp4_w4a4(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                       cudaStream_t) { unavailable(); }
void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) { unavailable(); }
void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) { unavailable(); }

void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) { unavailable(); }
void w8_attn_input_splitk_mma_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                     cudaStream_t) { unavailable(); }
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable(); }
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable(); }
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable(); }
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t) { unavailable(); }
void w8_linear_add_splitk_mma_launch(const Tensor&, const Weight&, Tensor&,
                                     cudaStream_t) { unavailable(); }
void w8_linear_add_medium_splitk_launch(const Tensor&, const Weight&, Tensor&,
                                        cudaStream_t) { unavailable(); }
void w8_pair_splitk_exact_t_launch(const Tensor&, const Weight&, const Weight&, Tensor&,
                                   Tensor&, cudaStream_t) { unavailable(); }
void w8_pair_splitk_medium_launch(W8PairScheduleId, const Tensor&, const Weight&, const Weight&,
                                  Tensor&, Tensor&, cudaStream_t) { unavailable(); }
void w8_dynamic_grouped_conv_add_materialized_launch(W8DynamicConvAddSchedule, const Tensor&,
                                                     const Weight&, const Tensor&, const Tensor&,
                                                     Tensor&, Tensor&, cudaStream_t) { unavailable(); }
} // namespace ninfer::ops::detail
