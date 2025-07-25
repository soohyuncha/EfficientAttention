using namespace nvcuda;

#define WARP_SIZE 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


template <int NUM_THREADS, int HEAD_SIZE, int Br, int Bc>
__global__ void flash_attn_kernel_fp16_v3(
    const __half* query, const __half* key, const __half* value,
    const int Tc, const int Tr, const int q_len, const int kv_len, const float softmax_scale,
    __half* out, float* row_max_global, float* S_global,
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
    float out_tmp[HEAD_SIZE];

    // Load Q to on-chip SRAM
#pragma unroll
    for (int i = 0; i < Br; i++) {
        for (int d = 0; d < HEAD_SIZE / NUM_THREADS; d++) {
            int dim_idx = d * NUM_THREADS + tid;
            Q[i * HEAD_SIZE + dim_idx] = query[q_b_offset + q_h_offset + q_blk_offset + i * HEAD_SIZE + dim_idx];
        }
    }
    __syncthreads();
    
    // Init. private variables
    row_max_prev = -FLT_MAX;
    row_exp_sum_prev = 0;
    for (int i = 0; i < HEAD_SIZE; i++) {
        out_tmp[i] = 0;
    }

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
#pragma unroll 
            for (int k = 0; k < HEAD_SIZE; k += WMMA_K) {
                wmma::load_matrix_sync(q_frag, Q + (warp_id * WMMA_M) * HEAD_SIZE + k, HEAD_SIZE);
                wmma::load_matrix_sync(k_frag, Kj + jj * HEAD_SIZE + k, HEAD_SIZE);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }
            wmma::store_matrix_sync(S + (warp_id * WMMA_M) * Bc + jj, s_frag, Bc, wmma::mem_row_major);
        }
        

        if ((lane_id < WMMA_M) && (q_blk_idx * Br + (warp_id * WMMA_M + lane_id) < q_len)) {
            // 3) Each thread obtains "row_max" of j-th tile
            int row_idx = warp_id * WMMA_M + lane_id;
            row_max_tile = -FLT_MAX;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                float sum;
                if (q_blk_idx * Br + row_idx < j * Bc + jj) {
                    sum = -FLT_MAX;
                }
                else {
                    sum = S[row_idx * Bc + jj] * softmax_scale;
                }
                row_max_tile = fmaxf(row_max_tile, sum);
                S[row_idx * Bc + jj] = sum;
                if (debug) {
                    S_global[(b_idx * h * q_len * kv_len) + (h_idx * q_len * kv_len) + ((q_blk_idx * Br + row_idx) * kv_len) + (j * Bc + jj)] = sum;
                }
            }

            // 4) Update "row_max" of j-th tile
            row_max_new = fmaxf(row_max_prev, row_max_tile);

            // 5) Compute e^(S - row_max)
            row_exp_sum_tile = 0;
            for (int jj = 0; jj < Bc; jj++) {
                if (j * Bc + jj >= kv_len) {
                    continue;
                }
                S_fp16[row_idx * Bc + jj] = __float2half(__expf(S[row_idx * Bc + jj] - row_max_new));
                row_exp_sum_tile += __half2float(S_fp16[row_idx * Bc + jj]);
            }

            // 6) Update "row_exp_sum" of j-th tile
            row_exp_sum_new = row_exp_sum_prev * expf(row_max_prev - row_max_new) + row_exp_sum_tile;
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

            for (int dd = 0; dd < WMMA_N; dd += 1) {
                if ((lane_id < WMMA_M) && (q_blk_idx * Br + (warp_id * WMMA_M + lane_id) < q_len)) {
                    int row_idx = warp_id * WMMA_M + lane_id;
                    out_tmp[d + dd] = out_tmp[d + dd] * __expf(row_max_prev - row_max_new) + O[row_idx * WMMA_N + dd];
                }
            }
            __syncthreads();
        }

        // 8) Update final output
        if ((lane_id < WMMA_M) && (q_blk_idx * Br + (warp_id * WMMA_M + lane_id) < q_len)) {
            int row_idx = warp_id * WMMA_M + lane_id;
            if (debug) {
                row_max_global[(b_idx * h * q_len) + (h_idx * q_len) + (q_blk_idx * Br + row_idx)] = row_max_new;
            }
            row_max_prev = row_max_new;
            row_exp_sum_prev = row_exp_sum_new;
            
            // Update output with scale factor only at the last tile
            if (j == Tc - 1) {
#pragma unroll
                for (int d = 0; d < HEAD_SIZE; d++) { 
                    out[q_b_offset + q_h_offset + q_blk_offset + row_idx * HEAD_SIZE + d] = __float2half((1 / row_exp_sum_new) * out_tmp[d]); 
                }
            }
        }
        __syncthreads();
    }
}
