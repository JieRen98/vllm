
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#include "quantization/gptq_marlin/marlin.cuh"
#include "quantization/gptq_marlin/marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define MARLIN_KERNEL_PARAMS                                          \
  const int4 *__restrict__ A, const int4 *__restrict__ B,             \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                 \
      const int4 *__restrict__ scales_ptr,                            \
      const uint16_t *__restrict__ scale2_ptr,                        \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, \
      const int32_t *__restrict__ sorted_token_ids_ptr,               \
      const int32_t *__restrict__ expert_ids_ptr,                     \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,         \
      const float *__restrict__ topk_weights_ptr, int top_k,          \
      bool mul_topk_weights, bool is_ep, int num_groups, int prob_m,  \
      int prob_n, int prob_k, int *locks, bool use_atomic_add,        \
      bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}  // namespace MARLIN_NAMESPACE_NAME

#define MARLIN_INT2_KERNEL_PARAMS                                    \
  const int4 *__restrict__ A, const int2 *__restrict__ B,            \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                \
      const int4 *__restrict__ zp_ptr,                               \
      const float *__restrict__ code_scale_ptr,                      \
      const float *__restrict__ code_zp_ptr,                         \
      const void *__restrict__ super_scale_ptr,                      \
      const int32_t *__restrict__ sorted_token_ids_ptr,              \
      const int32_t *__restrict__ expert_ids_ptr,                    \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,        \
      const float *__restrict__ topk_weights_ptr, int top_k,         \
      bool mul_topk_weights, bool is_ep, int num_groups, int prob_m, \
      int prob_n, int prob_k, int *locks, bool use_atomic_add,       \
      int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void MarlinInt2(MARLIN_INT2_KERNEL_PARAMS);

}
