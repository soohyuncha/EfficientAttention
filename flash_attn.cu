#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <int NUM_THREADS, int HEAD_SIZE, int Br, int Bc>
__global__ void flash_attn_kernel_fp16_v1(
    const __half* query, const __half* key, const __half* value,
    const int Tc, const int Tr, const int q_len, const int kv_len, const float softmax_scale,
    __half* out, float* row_max_global, float* row_exp_sum_global, float* S_global,
    bool debug
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int q_blk_idx = blockIdx.z;
    int b = gridDim.x;
    int h = gridDim.y;
    int tid = threadIdx.x;

    int kv_b_offset = b_idx * h * kv_len * HEAD_SIZE;
    int kv_h_offset = h_idx * kv_len * HEAD_SIZE;
    int q_b_offset = b_idx * h * q_len * HEAD_SIZE;
    int q_h_offset = h_idx * q_len * HEAD_SIZE;
    int q_blk_offset = q_blk_idx * Br * HEAD_SIZE;

    // Shared Mem. for Q, K, V, S
    extern __shared__ __half sram[];
    __half* Q = sram;                          // [Br, dim]
    __half* Kj = &sram[Br * HEAD_SIZE];               // [Bc, dim]
    __half* Vj = &sram[Br * HEAD_SIZE + Bc * HEAD_SIZE];    // [Bc, dim]
    float* S = reinterpret_cast<float*>(&sram[Br * HEAD_SIZE + 2 * Bc * HEAD_SIZE]);        // [Br, Bc]

    // Intermediate variables for online softmax
    float row_max_tile, row_max_new, row_max_prev;
    float row_exp_sum_tile, row_exp_sum_new, row_exp_sum_prev;
    float out_tile, out_new, out_prev;

    // Load Q to on-chip SRAM
#pragma unroll
    for (int i = 0; i < Br; i++) {
        for (int d = 0; d < HEAD_SIZE / NUM_THREADS; d++) {
            int dim_idx = d * NUM_THREADS + tid;
            Q[i * HEAD_SIZE + dim_idx] = query[q_b_offset + q_h_offset + q_blk_offset + i * HEAD_SIZE + dim_idx];
        }
    }
    __syncthreads();

    // Iterate tiles over KV dimension
    for (int j = 0; j < Tc; j++) {
        const int kv_blk_offset = j * Bc * HEAD_SIZE;
        // 1) Load K_j, V_j to on-chip SRAM
#pragma unroll
        for (int jj = 0; jj < Bc; jj++) {
            for (int d = 0; d < HEAD_SIZE / NUM_THREADS; d++) {
                int dim_idx = d * NUM_THREADS + tid;
                Kj[jj * HEAD_SIZE + dim_idx] = key[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + dim_idx];
                Vj[jj * HEAD_SIZE + dim_idx] = value[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + dim_idx];
            }
        }
        __syncthreads();
        
        // 2) Compute S = Q * K^T && maximum elements of j-th tile
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_max_tile = -FLT_MAX;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                float sum;
                // 1) Upper triangle -> -inf
                if (q_blk_idx * Br + tid < j * Bc + jj) {
                    sum = -FLT_MAX;
                }
                // 2) Lower triangle & diagonal -> q * k^T
                else {
                    sum = 0;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        sum = __fmaf_rn(__half2float(Q[tid * HEAD_SIZE + d]), __half2float(Kj[jj * HEAD_SIZE + d]), sum);
                    }
                    sum *= softmax_scale;
                }
                
                row_max_tile = fmaxf(row_max_tile, sum);
                S[tid * Bc + jj] = sum;
                if (debug) {
                    S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + tid) * kv_len) + (j * Bc + jj)] = sum;
                }
            }
        }
        
        // 3) Compute e^(S - row_max)
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_exp_sum_tile = 0;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                S[tid * Bc + jj] = __expf(S[tid * Bc + jj] - row_max_tile);
                row_exp_sum_tile += S[tid * Bc + jj];
            }
        }

        // 4) Compute O = S * V
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_max_prev = row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)];
            row_max_new = fmaxf(row_max_prev, row_max_tile);

            row_exp_sum_prev = row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)];     
            row_exp_sum_new = row_exp_sum_prev * expf(row_max_prev - row_max_new) + row_exp_sum_tile * expf(row_max_tile - row_max_new);

            for (int d = 0; d < HEAD_SIZE; d++) {
                out_tile = 0;
                for (int jj = 0; jj < Bc; jj++) {
                    if (j * Bc + jj >= kv_len) {
                        continue;
                    }
                    out_tile = __fmaf_rn(S[tid * Bc + jj], __half2float(Vj[jj * HEAD_SIZE + d]), out_tile);
                }
                
                out_prev = __half2float(out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + d]);
                out_new = (1 / row_exp_sum_new) * (out_prev * row_exp_sum_prev * __expf(row_max_prev - row_max_new) + out_tile * __expf(row_max_tile - row_max_new));
                out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + d] = __float2half(out_new);
            }

            row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_max_new;
            row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_exp_sum_new;
        }
        __syncthreads();
    }
}


template <int NUM_THREADS, int HEAD_SIZE, int Br, int Bc>
__global__ void flash_attn_kernel_fp16_v2(
    const __half* query, const __half* key, const __half* value,
    const int Tc, const int Tr, const int q_len, const int kv_len, const float softmax_scale,
    __half* out, float* row_max_global, float* row_exp_sum_global, float* S_global,
    bool debug
) {
    const int b_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int q_blk_idx = blockIdx.z;
    const int b = gridDim.x;
    const int h = gridDim.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int kv_b_offset = b_idx * h * kv_len * HEAD_SIZE;
    const int kv_h_offset = h_idx * kv_len * HEAD_SIZE;
    const int q_b_offset = b_idx * h * q_len * HEAD_SIZE;
    const int q_h_offset = h_idx * q_len * HEAD_SIZE;
    const int q_blk_offset = q_blk_idx * Br * HEAD_SIZE;

    // Shared Mem. for Q, K, V, S
    extern __shared__ __half sram[];
    __half* Q = sram;                           // [Br, dim]
    __half* Kj = Q + (Br * HEAD_SIZE);        // [Bc, dim]
    __half* Vj = Kj + (Bc * HEAD_SIZE);    // [Bc, dim]
    float* S = reinterpret_cast<float*>(Vj + Bc * HEAD_SIZE);        // [Br, Bc]
    __half* S_fp16 = reinterpret_cast<__half*>(S + Br * Bc);        // [Br, WMMA_K]
    float* O = S;     // [Br, WMMA_N]; Reuse S_FP32

    // Intermediate variables for online softmax
    float row_max_tile, row_max_new, row_max_prev;
    float row_exp_sum_tile, row_exp_sum_new, row_exp_sum_prev;
    float out_tile, out_new, out_prev;

    // Load Q to on-chip SRAM
#pragma unroll
    for (int i = 0; i < Br; i++) {
        for (int d = 0; d < HEAD_SIZE / NUM_THREADS; d++) {
            int dim_idx = d * NUM_THREADS + tid;
            Q[i * HEAD_SIZE + dim_idx] = query[q_b_offset + q_h_offset + q_blk_offset + i * HEAD_SIZE + dim_idx];
        }
    }
    __syncthreads();

    // Iterate tiles over KV dimension
    for (int j = 0; j < Tc; j++) {
        const int kv_blk_offset = j * Bc * HEAD_SIZE;
        // 1) Load K_j, V_j to on-chip SRAM
#pragma unroll
        for (int jj = 0; jj < Bc; jj++) {
            for (int d = 0; d < HEAD_SIZE / NUM_THREADS; d++) {
                int dim_idx = d * NUM_THREADS + tid;
                Kj[jj * HEAD_SIZE + dim_idx] = key[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + dim_idx];
                Vj[jj * HEAD_SIZE + dim_idx] = value[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + dim_idx];
            }
        }
        __syncthreads();
        
        // 2) Compute S = Q * K^T && maximum elements of j-th tile
        for (int jj = 0; jj < Bc; jj += WMMA_N) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            
            for (int k = 0; k < HEAD_SIZE; k += WMMA_K) {
                wmma::load_matrix_sync(q_frag, Q + (warp_id * WMMA_M) * HEAD_SIZE + k, HEAD_SIZE);
                wmma::load_matrix_sync(k_frag, Kj + jj * HEAD_SIZE + k, HEAD_SIZE);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(S + (warp_id * WMMA_M) * Bc + jj, s_frag, Bc, wmma::mem_row_major);
        }
        __syncthreads();
        
        // 3) Each thread obtains "row_max" of j-th tile
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_max_tile = -FLT_MAX;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                float sum;
                if (q_blk_idx * Br + tid < j * Bc + jj) {
                    sum = -FLT_MAX;
                }
                else {
                    sum = S[tid * Bc + jj] * softmax_scale;
                }
                row_max_tile = fmaxf(row_max_tile, sum);
                S[tid * Bc + jj] = sum;
                if (debug) {
                    S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + tid) * kv_len) + (j * Bc + jj)] = sum;
                }
            }
        }
        
        // 4) Compute e^(S - row_max)
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_exp_sum_tile = 0;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                S_fp16[tid * Bc + jj] = __float2half(__expf(S[tid * Bc + jj] - row_max_tile));
                row_exp_sum_tile += __half2float(S_fp16[tid * Bc + jj]);
            }
        }
        __syncthreads();

        // 5) Compute O = S * V 
        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_max_prev = row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)];
            row_max_new = fmaxf(row_max_prev, row_max_tile);

            row_exp_sum_prev = row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)]; 
            row_exp_sum_new = row_exp_sum_prev * expf(row_max_prev - row_max_new) + row_exp_sum_tile * expf(row_max_tile - row_max_new);
        }

        for (int d = 0; d < HEAD_SIZE; d += WMMA_N) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> s_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);
        
            for (int jj = 0; jj < Bc; jj += WMMA_K) {
                wmma::load_matrix_sync(s_frag, S_fp16 + (warp_id * WMMA_M) * Bc + jj, Bc);
                wmma::load_matrix_sync(v_frag, Vj + jj * HEAD_SIZE + d, HEAD_SIZE);
                wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(O + (warp_id * WMMA_M) * WMMA_N, o_frag, WMMA_N, wmma::mem_row_major);
            __syncthreads();

            for (int dd = 0; dd < WMMA_N; dd += 1) {
                if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
                    out_prev = __half2float(out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + (d + dd)]);
                    out_new = (1 / row_exp_sum_new) * (out_prev * row_exp_sum_prev * __expf(row_max_prev - row_max_new) + O[tid * WMMA_N + dd] * __expf(row_max_tile - row_max_new));
                    out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + (d + dd)] = __float2half(out_new);
                }
            }
        }

        __syncthreads();

        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_max_new;
            row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_exp_sum_new;
        }
        __syncthreads();
    }
}



template <int NUM_THREADS, int HEAD_SIZE, int Br, int Bc>
__global__ void flash_attn_kernel_fp16(
    const __half* query,
    const __half* key,
    const __half* value,
    const int Tc, const int Tr, const int q_len, const int kv_len, const float softmax_scale,
    __half* out, float* row_max_global, float* row_exp_sum_global, float* S_global,
    bool debug
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int q_blk_idx = blockIdx.z;
    int b = gridDim.x;
    int h = gridDim.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    const int dim = HEAD_SIZE;

    int kv_b_offset = b_idx * h * kv_len * HEAD_SIZE;
    int kv_h_offset = h_idx * kv_len * HEAD_SIZE;
    int q_b_offset = b_idx * h * q_len * HEAD_SIZE;
    int q_h_offset = h_idx * q_len * HEAD_SIZE;
    int q_blk_offset = q_blk_idx * Br * HEAD_SIZE;

    const int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    // SRAM for Q, K, V, S
    extern __shared__ __half sram[];
    __half* Q = sram;                          // [Br, dim]
    __half* Kj = &sram[Br * dim];               // [Bc, dim]
    __half* Vj = &sram[Br * dim + Bc * dim];    // [Bc, dim]
    float* S = reinterpret_cast<float*>(&sram[Br * dim + 2 * Bc * dim]);        // [Br, Bc]

    // Intermediate variables
    __shared__ float row_max_tile[Br];
    __shared__ float row_exp_sum_tile[Br];
//    __shared__ float out_i[HEAD_SIZE];
    float row_max_new, row_max_prev;
    float row_exp_sum_new, row_exp_sum_prev;
    float out_i, out_prev, out_new;
    

    // Load Q to on-chip SRAM
#pragma unroll
    for (int i = 0; i < Br; i++) {
        Q[i * HEAD_SIZE + tid] = query[q_b_offset + q_h_offset + q_blk_offset + i * HEAD_SIZE + tid];
    }
    __syncthreads();

    // Iterate tiles over KV dimension
    for (int j = 0; j < Tc; j++) {
        const int kv_blk_offset = j * Bc * HEAD_SIZE;
        // Load K_j, V_j to on-chip SRAM
#pragma unroll
        for (int jj = 0; jj < Bc; jj++) {
            Kj[jj * HEAD_SIZE + tid] = key[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + tid];
            Vj[jj * HEAD_SIZE + tid] = value[kv_b_offset + kv_h_offset + kv_blk_offset + jj * HEAD_SIZE + tid];
        }
        __syncthreads();


        // Compute S = Q * K^T ==> Naive GeMM... No need for shared memory for Q...
//        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
//            row_max_tile = -FLT_MAX;
//            for (int jj = 0; jj < Bc; jj++) {
//                if (j * Bc + jj >= kv_len) {
//                    continue;
//                }
//                float sum;
//                // 1) Upper triangle -> -inf
//                if (q_blk_idx * Br + tid < j * Bc + jj) {
//                    sum = -FLT_MAX;
//                }
//                // 2) Lower triangle & diagonal -> q * k^T
//                else {
//                    sum = 0;
//                    for (int d = 0; d < HEAD_SIZE; d++) {
//                        sum += __half2float(__hmul(Q[tid * HEAD_SIZE + d], Kj[jj * HEAD_SIZE + d]));
//                    }
//                    sum *= softmax_scale;
//                }
//                
//                row_max_tile = fmaxf(row_max_tile, sum);
//                S[tid * Bc + jj] = sum;
//                if (debug) {
//                    S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + tid) * kv_len) + (j * Bc + jj)] = sum;
//                }
//            }
//        }

        // Compute S = Q * K^T
        for (int i = 0; i < Br; i++) {
            for (int jj = 0; jj < Bc / NUM_WARPS; jj++) {
                int kv_idx = jj * NUM_WARPS + warp_id;
                float thread_local_sum;
                // Causal condition
                if (q_blk_idx * Br + i >= j * Bc + kv_idx) {
                    thread_local_sum = 0;
                    // Thread computation
                    for (int d = 0; d < HEAD_SIZE / WARP_SIZE; d++) {
                        thread_local_sum = __fmaf_rn(__half2float(Q[i * HEAD_SIZE + d * WARP_SIZE + lane_id]), __half2float(Kj[kv_idx * HEAD_SIZE + d * WARP_SIZE + lane_id]), thread_local_sum);
                    }
                    // Reduction
                    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                        thread_local_sum += __shfl_down_sync(0xffffffff, thread_local_sum, offset);
                    }
                }
                else {
                    thread_local_sum = -FLT_MAX;
                }         
                // Write to shared mem.
                if (lane_id == 0) {
                    S[i * Bc + kv_idx] = thread_local_sum * softmax_scale;
                    if (debug) {
                        S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + i) * kv_len) + (j * Bc + kv_idx)] = S[i * Bc + kv_idx];
                    }
                }
            }
        }
        __syncthreads();
        

//        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
//            // Compute exp(S - row_max)
//            row_exp_sum_tile = 0;
//            for (int jj = 0; jj < Bc; jj++) {
//                if (j * Bc + jj >= kv_len) {
//                    continue;
//                }
//                S[tid * Bc + jj] = __expf(S[tid * Bc + jj] - row_max_tile);
//                row_exp_sum_tile += S[tid * Bc + jj];
//            }
//        }

        // Get "row_max" for each row of j-th tile
        for (int i = 0; i < Br / NUM_WARPS; i++) {
            int row_idx = i * NUM_WARPS + warp_id;

            // Get row_max for j-th tile
            float thread_local_row_max = -FLT_MAX;
            // Per-thread
            for (int jj = 0; jj < Bc / WARP_SIZE; jj++) {
                int kv_idx = jj * WARP_SIZE + lane_id;
                thread_local_row_max = fmaxf(thread_local_row_max, S[row_idx * Bc + kv_idx]); 
            }
            // Reduction
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                thread_local_row_max = fmaxf(thread_local_row_max, __shfl_down_sync(0xffffffff, thread_local_row_max, offset));
            }
            if (lane_id == 0) {
                row_max_tile[row_idx] = thread_local_row_max;
            }
        }
        __syncthreads();

        // Get "row_exp_sum" for each row of j-th tile
        for (int i = 0; i < Br / NUM_WARPS; i++) {
            int row_idx = i * NUM_WARPS + warp_id;

            // Compute exp(S - row_max) && Get row_exp_sum for j-th tile
            float thread_local_row_exp_sum = 0;
            // Per-thread
            for (int jj = 0; jj < Bc / WARP_SIZE; jj++) {
                int kv_idx = jj * WARP_SIZE + lane_id;
                S[row_idx * Bc + kv_idx] = expf(S[row_idx * Bc + kv_idx] - row_max_tile[row_idx]);
                thread_local_row_exp_sum += S[row_idx * Bc + kv_idx]; 
            }
            // Reduction
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                thread_local_row_exp_sum +=  __shfl_down_sync(0xffffffff, thread_local_row_exp_sum, offset);
            }
            if (lane_id == 0) {
                row_exp_sum_tile[row_idx] = thread_local_row_exp_sum;
            }
        }
        __syncthreads();

        if ((tid < Br) && (q_blk_idx * Br + tid) < q_len) {
            // Compute O = S * V
            row_max_prev = row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)];
            row_max_new = fmaxf(row_max_prev, row_max_tile[tid]);

            row_exp_sum_prev = row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)];     
            row_exp_sum_new = row_exp_sum_prev * expf(row_max_prev - row_max_new) + row_exp_sum_tile[tid] * expf(row_max_tile[tid] - row_max_new);

            for (int d = 0; d < HEAD_SIZE; d++) {
                out_i = 0;
                for (int jj = 0; jj < Bc; jj++) {
                    if (j * Bc + jj >= kv_len) {
                        continue;
                    }
                    out_i = __fmaf_rn(S[tid * Bc + jj], __half2float(Vj[jj * HEAD_SIZE + d]), out_i);
                }
                
                out_prev = __half2float(out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + d]);
                out_new = (1 / row_exp_sum_new) * (out_prev * row_exp_sum_prev * __expf(row_max_prev - row_max_new) + out_i * __expf(row_max_tile[tid] - row_max_new));
                out[q_b_offset + q_h_offset + q_blk_offset + tid * HEAD_SIZE + d] = __float2half(out_new);
            }

            row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_max_new;
            row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + tid)] = row_exp_sum_new;
        }
        __syncthreads();
        
//        for (int i = 0; i < Br / NUM_WARPS; i++) {
//            int row_idx = i * NUM_WARPS + warp_id;
//
//            if (lane_id == 0) {
//                row_max_prev = row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + row_idx)];
//                row_max_new = fmaxf(row_max_prev, row_max_tile[row_idx]);
//
//                row_exp_sum_prev = row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + row_idx)];     
//                row_exp_sum_new = row_exp_sum_prev * expf(row_max_prev - row_max_new) + row_exp_sum_tile[row_idx] * expf(row_max_tile[row_idx] - row_max_new);
//            }
//            
//            // Compute O = S  * V
//            for (int d = 0; d < HEAD_SIZE; d++) {
//                float thread_local_sum = 0;
//                // Per-thread
//                for (int jj = 0; jj < Bc / WARP_SIZE; jj++) {
//                    int kv_idx = jj * WARP_SIZE + lane_id;
//                    thread_local_sum = __fmaf_rn(S[row_idx * Bc + kv_idx], __half2float(Vj[kv_idx * HEAD_SIZE + d]), thread_local_sum);
//                }
//                // Reduction
//                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
//                    thread_local_sum += __shfl_down_sync(0xffffffff, thread_local_sum, offset);
//                }
//
//                if (lane_id == 0) {
//                    out_i = thread_local_sum;
//                    out_prev = __half2float(out[q_b_offset + q_h_offset + q_blk_offset + row_idx * HEAD_SIZE + d]);
//                    out_new = (1 / row_exp_sum_new) * (out_prev * row_exp_sum_prev * __expf(row_max_prev - row_max_new) + out_i * __expf(row_max_tile[row_idx] - row_max_new));
//                    out[q_b_offset + q_h_offset + q_blk_offset + row_idx * HEAD_SIZE + d] = __float2half(out_new);
//                }
//            }
//
//            if (lane_id == 0) {
//                row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + row_idx)] = row_max_new;
//                row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + row_idx)] = row_exp_sum_new;
//            }
//        }
//        __syncthreads();
    }

    return;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> flash_attn_fp16(
        torch::Tensor query,        // [b, h, n, d]
        torch::Tensor key,          // [b, h, n, d]
        torch::Tensor value,        // [b, h, n, d]
        bool print_info,
        bool debug,
        int version
) {
    // Get dimension info.
    const int b = query.size(0);
    const int h = query.size(1);
    const int q_len = query.size(2);
    const int kv_len = key.size(2);
    const int dim = query.size(3);
    const float softmax_scale = 1.0 / sqrt(dim);
    
    // Tiling dimension
    
    if (version == 1) {
        const int Bc = 32;
        const int Br = 64;
        const int Tc = ceil((float) kv_len / Bc);
        const int Tr = ceil((float) q_len / Br);

        // Declare intermediate & output tensors
        auto out = torch::zeros_like(query);        // [b, h, n, d]
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
        auto row_max = torch::full({b, h, Tr, Br}, -std::numeric_limits<float>::max(), options);    // [b, h, Tr, Br]; Initialize as -inf
        auto row_exp_sum = torch::zeros({b, h, Tr, Br}, options);      // [b, h, Tr, Br]; Initialize as zero
        torch::Tensor attn_score = torch::empty({b, h, q_len, kv_len}, options);

        int max_sram_size, sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        if (print_info) {
            printf("Max shared memory: %d KB \n", max_sram_size / 1024);
        }

        sram_size = Br * dim * sizeof(at::Half) + 2 * Bc * dim * sizeof(at::Half) + Br * Bc * sizeof(float);

        if (print_info) {
            printf("Required shared memory: %d KB \n", sram_size / 1024);
            printf("Tile dimension Bc=%d Br=%d Tc=%d Tr=%d \n", Bc, Br, Tc, Tr);
        }
        dim3 grid_dim(b, h, Tr);

        // Launch kernel
        const int NUM_THREADS = 64;
        dim3 block_dim(NUM_THREADS);
        flash_attn_kernel_fp16_v1<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
                reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                Tc, Tr, q_len, kv_len, softmax_scale,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                reinterpret_cast<float*>(row_max.data_ptr<float>()),
                reinterpret_cast<float*>(row_exp_sum.data_ptr<float>()),
                reinterpret_cast<float*>(attn_score.data_ptr<float>()),
                debug
        );

        return std::make_tuple(out, row_max, row_exp_sum, attn_score);
    }

    else if (version == 2) {
        const int Bc = 32;
        const int Br = 64;
        const int Tc = ceil((float) kv_len / Bc);
        const int Tr = ceil((float) q_len / Br);

        // Declare intermediate & output tensors
        auto out = torch::zeros_like(query);        // [b, h, n, d]
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
        auto row_max = torch::full({b, h, Tr, Br}, -std::numeric_limits<float>::max(), options);    // [b, h, Tr, Br]; Initialize as -inf
        auto row_exp_sum = torch::zeros({b, h, Tr, Br}, options);      // [b, h, Tr, Br]; Initialize as zero
        torch::Tensor attn_score = torch::empty({b, h, q_len, kv_len}, options);

        int max_sram_size, sram_size;
        cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        if (print_info) {
            printf("Max shared memory: %d KB \n", max_sram_size / 1024);
        }

        sram_size = Br * dim * sizeof(at::Half) + 2 * Bc * dim * sizeof(at::Half) + Br * Bc * sizeof(float) + Br * Bc * sizeof(__half);

        if (print_info) {
            printf("Required shared memory: %d KB \n", sram_size / 1024);
            printf("Tile dimension Bc=%d Br=%d Tc=%d Tr=%d \n", Bc, Br, Tc, Tr);
        }
        dim3 grid_dim(b, h, Tr);

        // Launch kernel
        const int NUM_THREADS = 128;
        dim3 block_dim(NUM_THREADS);
        flash_attn_kernel_fp16_v2<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
                reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
                Tc, Tr, q_len, kv_len, softmax_scale,
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                reinterpret_cast<float*>(row_max.data_ptr<float>()),
                reinterpret_cast<float*>(row_exp_sum.data_ptr<float>()),
                reinterpret_cast<float*>(attn_score.data_ptr<float>()),
                debug
        );

        return std::make_tuple(out, row_max, row_exp_sum, attn_score);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fp16", &flash_attn_fp16);
}
