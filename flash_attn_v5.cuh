using namespace nvcuda;

#define WARP_SIZE 32
#define VEC_SIZE 8

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


template <int NUM_THREADS, int HEAD_SIZE, int Br, int Bc>
__global__ void flash_attn_kernel_fp16_v5(
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
    float out_tmp[HEAD_SIZE / 2];
    
    const int THREAD_GROUP_SIZE = NUM_THREADS / Br;     // 2
    const int thread_group_id = tid / THREAD_GROUP_SIZE;    // [0, 63]
    const int thread_group_offset = tid % THREAD_GROUP_SIZE;    // [0, 1]


    // Load Q to on-chip SRAM (vectorized load w/ 16B granularity)
#pragma unroll
    for (int vec_offset = 0; vec_offset < Br * HEAD_SIZE; vec_offset += NUM_THREADS * VEC_SIZE) {
        const int vec_idx = vec_offset + tid * VEC_SIZE;
        const float4* query_fp4_ptr = reinterpret_cast<const float4*>(&query[q_b_offset + q_h_offset + q_blk_offset + vec_idx]);
        float4* Q_fp4_ptr = reinterpret_cast<float4*>(&Q[vec_idx]);
        *Q_fp4_ptr = *query_fp4_ptr;
    }

    // Init. private variables
    row_max_prev = -FLT_MAX;
    row_exp_sum_prev = 0;
    for (int i = 0; i < HEAD_SIZE / 2; i++) {
        out_tmp[i] = 0;
    }

    // Iterate tiles over KV dimension
    for (int j = 0; j < Tc; j++) {
        const int kv_blk_offset = j * Bc * HEAD_SIZE;
        // 1) Load K_j, V_j to on-chip SRAM (vectorized load w/ 16B granularity)
#pragma unroll
        for (int vec_offset = 0; vec_offset < Bc * HEAD_SIZE; vec_offset += NUM_THREADS * VEC_SIZE) {
            const int vec_idx = vec_offset + tid * VEC_SIZE;
            const float4* key_fp4_ptr = reinterpret_cast<const float4*>(&key[kv_b_offset + kv_h_offset + kv_blk_offset + vec_idx]);
            float4* Kj_fp4_ptr = reinterpret_cast<float4*>(&Kj[vec_idx]);
            *Kj_fp4_ptr = *key_fp4_ptr;
            const float4* value_fp4_ptr = reinterpret_cast<const float4*>(&value[kv_b_offset + kv_h_offset + kv_blk_offset + vec_idx]);
            float4* Vj_fp4_ptr = reinterpret_cast<float4*>(&Vj[vec_idx]);
            *Vj_fp4_ptr = *value_fp4_ptr;
        }
        __syncthreads();
        
        // 2) Compute S = Q * K^T && maximum elements of j-th tile
        for (int jj = 0; jj < Bc; jj += WMMA_N) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll 
            for (int k = 0; k < HEAD_SIZE; k += WMMA_K) {
                wmma::load_matrix_sync(q_frag, Q + (warp_id * WMMA_M) * HEAD_SIZE + k, HEAD_SIZE);
                wmma::load_matrix_sync(k_frag, Kj + jj * HEAD_SIZE + k, HEAD_SIZE);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(S + (warp_id * WMMA_M) * Bc + jj, s_frag, Bc, wmma::mem_row_major);
        }
        
        for (int i = warp_id * WMMA_M; i < (warp_id + 1) * WMMA_M; i++) {
            // 3) Store scaled q*k^T into shared mem. of S
            float s_ij = (q_blk_idx * Br + i < j * Bc + lane_id) ? -FLT_MAX : S[i * Bc + lane_id] * softmax_scale;
            S[i * Bc + lane_id] = s_ij;
            if (debug) {
                S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + i) * kv_len) + (j * Bc + lane_id)] = s_ij;
            }
            // 4) Get "row_max" using warp primitive
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                s_ij = fmaxf(s_ij, __shfl_xor_sync(0xFFFFFFFF, s_ij, offset));
            }
            // Assign thread-private variables
            row_max_tile = s_ij;
            float row_max_prev_i = __shfl_sync(0xFFFFFFFF, row_max_prev, (i * THREAD_GROUP_SIZE) % WARP_SIZE);
            float row_max_new_i = fmaxf(row_max_prev_i, row_max_tile);
            row_max_new = (thread_group_id == i) ? row_max_new_i : row_max_new;

            // 5) Compute e^(S - row_max)
            s_ij = __expf(S[i * Bc + lane_id] - row_max_new_i);
            S_fp16[i * Bc + lane_id] = __float2half(s_ij);

            // 6) Compute "row_exp_sum" using warp primitive
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                s_ij += __shfl_xor_sync(0xFFFFFFFF, s_ij, offset);
            }
            row_exp_sum_new = (thread_group_id == i) ? row_exp_sum_prev * expf(row_max_prev - row_max_new) + s_ij : row_exp_sum_new;
        }
        
        __syncthreads();
        
        // 7) Compute O = S * V 
        for (int d = 0; d < HEAD_SIZE; d += WMMA_N) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> s_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);
#pragma unroll
            for (int jj = 0; jj < Bc; jj += WMMA_K) {
                wmma::load_matrix_sync(s_frag, S_fp16 + (warp_id * WMMA_M) * Bc + jj, Bc);
                wmma::load_matrix_sync(v_frag, Vj + jj * HEAD_SIZE + d, HEAD_SIZE);
                wmma::mma_sync(o_frag, s_frag, v_frag, o_frag);
            }
            wmma::store_matrix_sync(O + (warp_id * WMMA_M) * WMMA_N, o_frag, WMMA_N, wmma::mem_row_major);
            __syncthreads();

            for (int dd = 0; dd < WMMA_N / THREAD_GROUP_SIZE; dd += 1) {
                out_tmp[d / THREAD_GROUP_SIZE + dd] = out_tmp[d / THREAD_GROUP_SIZE + dd] * __expf(row_max_prev - row_max_new) + O[thread_group_id * WMMA_N + (WMMA_N / THREAD_GROUP_SIZE) * thread_group_offset + dd];
            }
            __syncthreads();
        }

        // 8) Update final output
        
        if (debug) {
            row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + thread_group_id)] = row_max_new;
            row_exp_sum_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + thread_group_id)] = row_exp_sum_new;
        }
        row_max_prev = row_max_new;
        row_exp_sum_prev = row_exp_sum_new;
        
        // Write scaled output into global memory only at the last j-th tile (vectorized store w/ 16B granularity)
        if (j == Tc - 1) {
#pragma unroll
            for (int d = 0; d < HEAD_SIZE; d += VEC_SIZE * THREAD_GROUP_SIZE) {
                float4* out_fp4_ptr = reinterpret_cast<float4*>(&out[q_b_offset + q_h_offset + q_blk_offset + thread_group_id * HEAD_SIZE + d + VEC_SIZE * thread_group_offset]);
                __half out_tmp_fp16[VEC_SIZE];
#pragma unroll
                for (int dd = 0; dd < VEC_SIZE; dd++) {
                    out_tmp_fp16[dd] = __float2half((1 / row_exp_sum_new) * out_tmp[d / THREAD_GROUP_SIZE + dd]);
                }
                const float4* out_tmp_fp4_ptr = reinterpret_cast<const float4*>(&out_tmp_fp16);
                *out_fp4_ptr = *out_tmp_fp4_ptr;
            }
        }  
    }
}
