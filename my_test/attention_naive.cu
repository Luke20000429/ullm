/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#ifdef ENABLE_FP8_E5M2
#include "../quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#endif

#include <algorithm>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace ullm {

// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  typename scalar_t,
  typename cache_t,
  int HEAD_SIZE,
  int BLOCK_SIZE, // NOTE: still using BLOCK_SIZE to control job distribution
  int NUM_THREADS,
  bool IS_FP8_E5M2_KV_CACHE,
  int PARTITION_SIZE = 0> // Zero means no partitioning.
__device__ void uvm_attention_kernel(
  float* __restrict__ exp_sums,           // [num_seqs, num_heads]
  float* __restrict__ max_logits,         // [num_seqs, num_heads]
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const cache_t** __restrict__ k_caches,    // num_seqs x [num_kv_heads, num_tokens (context len), head_size]
  const cache_t** __restrict__ v_caches,    // num_seqs x [num_kv_heads, head_size, num_tokens (context len)]
  const int num_kv_heads,                 // [num_heads]
  const float scale,
  // const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq] // no longer need this
  const int* __restrict__ context_lens,   // [num_seqs]
  // const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride, // next query stride
  // const int kv_block_stride,
  const int kv_head_stride) { // next head stride
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  // NOTE: as vLLM, we assume not using partitioning for now
  // const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  // const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  // const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  // const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  // const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  // const int start_token_idx = start_block_idx * BLOCK_SIZE; // -> 0
  // const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len); // -> context_len
  // const int num_tokens = end_token_idx - start_token_idx; // -> context_len

  // BLOCK size now is the processing granularity, a.k.a. number of tokens to process in one go by a warp
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1); // number of threads to process one token
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE); // number of tokens of a thread group in a block
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE; // which warp
  const int lane = thread_idx % WARP_SIZE; // which thread within a warp

  const int head_idx = blockIdx.x; // which head
  const int num_heads = gridDim.x; // num of heads
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // NOTE: k_cache and v_cache are now stored in continuous memory.
  // each blockIdx.y takes a sequence's kv_cache
  const cache_t* k_cache = k_caches[seq_idx] + kv_head_idx * kv_head_stride;
  const cache_t* v_cache = v_caches[seq_idx] + kv_head_idx * kv_head_stride;

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1); // 128 bit cache line
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
#ifdef ENABLE_FP8_E5M2
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;
#endif

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE; // index of a thread group, normally thread_idx, because group_size is one
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE; // offset inside the thread group, normally 0, because group_size is one

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  // NOTE: all threads in a warp load a query collaboratively
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // if the memory is continuous, we can use a loop to load the key
  // every thread group take care of one token, total num of thread groups = NUM_THREAD_GROUPS*NUM_WARPS
  for (int token_id = thread_group_idx; token_id < context_len; token_id += NUM_THREAD_GROUPS*NUM_WARPS) { // work within a block
    K_vec k_vecs[NUM_VECS_PER_THREAD];
    const cache_t* k_ptr = k_cache + token_id*HEAD_SIZE; // ptr to the first element of the token
#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) { // work within a thread group, load a token
      const int offset = thread_group_offset + j * THREAD_GROUP_SIZE; // offset of the element to load
      if constexpr (IS_FP8_E5M2_KV_CACHE) {
#ifdef ENABLE_FP8_E5M2
        Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + offset);
        // Vector conversion from Quant_vec to K_vec.
        k_vecs[j] = fp8_e5m2_unscaled::vec_conversion<K_vec, Quant_vec>(k_vec_quant);
#else
        assert(false);
#endif
      } else {
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset);
      }
    }
    // Compute dot product.
    // This includes a reduction across the threads in the same thread group.
    // so partial q times the full k and sum them up by reduction
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
    // Add the ALiBi bias if slopes are given.
    qk += (alibi_slope != 0) ? alibi_slope * (token_id - context_len + 1) : 0;
    if (thread_group_offset == 0) {
      // Store the partial reductions to shared memory.
      const bool mask = token_id >= context_len;
      logits[token_id] = mask ? 0.f : qk;
      // Update the max value.
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // NOTE: partition is disabled by default
  // If partitioning is enabled, store the max logit and exp_sum.
  // if (USE_PARTITIONING && thread_idx == 0) {
  //   float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
  //                                      + head_idx * max_num_partitions
  //                                      + partition_idx;
  //   *max_logits_ptr = qk_max;
  //   float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
  //                                  + head_idx * max_num_partitions
  //                                  + partition_idx;
  //   *exp_sums_ptr = exp_sum;
  // }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
#ifdef ENABLE_FP8_E5M2
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
#endif
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  // NOTE: row dim = context len = num tokens, col dim = head size
  // keep BLOCK_SIZE as processing granularity, but not necessary actually
  // vLLM didn't use thread group here as the block size is usually smaller than head size
  // NOTE: each warp works on a BLOCK size of tokens
  // NOTE: if warp_size > BLOCK_SIZE, later thread will process another row
  // TODO: avoid iteration by block, do one row per warp, consider the head size is normally larger than 32 (max number of warps per block)
  // which will only requires warp level reduction, save time
  // also save time when context len is short
  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE; // divide each row (context_len) into several v_vecs
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW; // rows can be processed by a warp in one iter
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER); // because each thread processes a row per iter
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  // NOTE: v_cache of a head = head_size (n_row) x num_tokens (context len)
  // NOTE: compute logits@v by BLOCK_SIZE of tokens as the context_len is too large, need a reduction across blocks/warps
  for (int block_idx = warp_idx; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
    const int row_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE; // offset within a block of a row
    const int token_idx = block_idx * BLOCK_SIZE + row_offset; // idx of beginning token on a row
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx)); // load the required part of logits
    // because not using partition, start_token_idx is ignored
    const cache_t* v_ptr = v_cache; // ptr to the first element of the block
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) { // distribute workload within a warp
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER; // later threads will process another row
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + row_offset; // each row is of dimension block size
        V_vec v_vec;
        if constexpr (IS_FP8_E5M2_KV_CACHE) {
#ifdef ENABLE_FP8_E5M2
          V_quant_vec v_quant_vec = *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8_e5m2_unscaled::vec_conversion<V_vec, V_quant_vec>(v_quant_vec);
#else
          assert(false);
#endif
        } else {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        }
        if (block_idx == num_context_blocks - 1) { // last block
          // When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + head_idx * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template<
  typename scalar_t,
  typename cache_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS,
  bool IS_FP8_E5M2_KV_CACHE>
__global__ void uvm_attention_v1_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const cache_t** __restrict__ k_caches,    // num_seqs x [num_kv_heads, num_tokens (context len), head_size]
  const cache_t** __restrict__ v_caches,    // num_seqs x [num_kv_heads, head_size, num_tokens (context len)]
  const int num_kv_heads,                 // [num_heads]
  const float scale,
  // const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  // const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  // const int kv_block_stride,
  const int kv_head_stride) {
  uvm_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, IS_FP8_E5M2_KV_CACHE>(
    /* exp_sums */ nullptr, /* max_logits */ nullptr,
    out, q, k_caches, v_caches, num_kv_heads, scale, context_lens, 
    alibi_slopes, q_stride, kv_head_stride);
  }

} // namespace ullm

#define LAUNCH_UVM_ATTENTION_V1(HEAD_SIZE)                                                  \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                                       \
    ((void*)ullm::uvm_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,   \
      IS_FP8_E5M2_KV_CACHE>), shared_mem_size);                                               \
  ullm::uvm_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,             \
  IS_FP8_E5M2_KV_CACHE><<<grid, block, shared_mem_size, stream>>>(                            \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    num_kv_heads,                                                                             \
    scale,                                                                                    \
    context_lens_ptr,                                                                         \
    alibi_slopes_ptr,                                                                         \
    q_stride,                                                                                 \
    kv_head_stride);


// make a test code for the kernel
template<
  typename T,
  typename CACHE_T,
  int BLOCK_SIZE,
  bool IS_FP8_E5M2_KV_CACHE,
  int NUM_THREADS = 128>
void uvm_attention_v1_launcher(
  T* out_ptr,
  T* query_ptr,
  CACHE_T* key_cache_ptr,
  CACHE_T* value_cache_ptr,
  int num_seqs,
  int num_heads,
  int head_size,
  int q_stride,
  int kv_head_stride,
  int num_kv_heads,
  float scale,
  int* context_lens_ptr,
  int max_context_len,
  const float* alibi_slopes_ptr=nullptr) {

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

int main(int argc, char* argv[]) {
  // test the kernel
  // setting test configurations from input args
  
  // 1. prepare the input arrays
  // 2. launch the kernel
  // 3. check the output
  // 4. compare with the reference implementation
  // 5. print the performance

  return 0;
}

// NOTE: skip implement v2 as it requires another round of reduction

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
