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

#include "flash_attn_v1.cuh"
#include "flash_attn_v2.cuh"
#include "flash_attn_v3.cuh"
#include "flash_attn_v4.cuh"
#include "flash_attn_v5.cuh"
#include "flash_attn_v6.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> flash_attn_fp16(
        torch::Tensor query,        // [b, h, n, d]
        torch::Tensor key,          // [b, h, n, d]
        torch::Tensor value,        // [b, h, n, d]
        bool print_info,
        bool debug,
        float version
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

    else if (version >= 3 && version <= 6) {
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
        if (version == 3) {
            flash_attn_kernel_fp16_v3<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
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
        }
        else if (version == 4) {
            flash_attn_kernel_fp16_v4<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
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
        }
        else if (version == 5) {
            flash_attn_kernel_fp16_v5<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
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
        }
        else if (version == 6) {
            flash_attn_kernel_fp16_v6<NUM_THREADS, 128, Br, Bc><<<grid_dim, block_dim, sram_size>>> (
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
        }

        return std::make_tuple(out, row_max, row_exp_sum, attn_score);
    }

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fp16", &flash_attn_fp16);
}
