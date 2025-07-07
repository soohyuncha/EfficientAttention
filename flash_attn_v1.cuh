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
