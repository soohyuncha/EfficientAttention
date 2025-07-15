import torch
import flash_attn
import custom_flash_attn
from triton_src.triton_prefill_attn import triton_flash_attn_prefill

import os
import math
import argparse
import time

import triton
import triton.language as tl

import numpy as np

DTYPE=torch.float16
DEVICE="cuda:0"

def bench_flash_attention(query, key, value, BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize=False, NUM_TESTS=10):
    def _get_causal_mask(n):
        causal_mask = torch.tril(torch.ones((n, n), dtype=DTYPE, device=query.device))
        causal_mask = torch.where(causal_mask == 0, -torch.inf, causal_mask)
        causal_mask = torch.where(causal_mask == 1, 0, causal_mask)
        return causal_mask.unsqueeze(0).unsqueeze(0)
    
    # 1) Torch
    causal_mask = _get_causal_mask(N_CTX)
    p = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM))
    p = p + causal_mask
    p = torch.softmax(p.float(), dim = -1).to(torch.float16)
    ref_out_torch = torch.matmul(p, value)

    # 2) FA-2
    ref_out_fa2 = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)
    
    func_check = torch.allclose(ref_out_torch, ref_out_fa2, rtol=0, atol=1e-2)
    if func_check:
        print(f'> FA-2 functionality correct!')
    else:
        print(f'> FA-2 functionality wrong...')
        print(torch.topk(abs(out_ref_torch.reshape(-1) - out_ref_fa2.reshape(-1)), k=20, dim=0))
        exit()
    print()
    
    # 3) CUDA-based custom
    ver = 7
    out_custom_cuda, _, _ = custom_flash_attn.flash_attn_fp16(query, key, value, True, True, ver)

    func_check = torch.allclose(ref_out_torch, out_custom_cuda, rtol=0, atol=1e-2)
    if func_check:
        print(f'> Custom CUDA ver {ver} functionality correct!')
    else:
        print(f'> Custom CUDA ver {ver} functionality wrong...')
        print(torch.topk(abs(ref_out_torch.reshape(-1) - out_custom_cuda.reshape(-1)), k=20, dim=0))
        exit()
    print()

    # 4) Triton-based
    out_triton, s_triton, m_triton = triton_flash_attn_prefill(query, key, value, causal, warp_specialize, True)
    s_torch = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM)) + causal_mask
    s_torch = s_torch.float()
    s_torch = torch.where(s_torch < -1e5, 0, s_torch)
    s_triton = torch.where(s_triton < -1e5, 0, s_triton)
    row_max_torch = torch.max(s_torch, dim=-1).values

#    func_check = torch.allclose(s_torch, s_triton, rtol=0, atol=1e-2)
#    if func_check:
#        print(f'> Triton functionality for S correct!')
#    else:
#        print(f'> Triton functionality for S wrong...')
#        print(torch.topk(abs(s_torch.reshape(-1) - s_triton.reshape(-1)), k=20, dim=0))
#        exit()
#
#    func_check = torch.allclose(row_max_torch, m_triton, rtol=0, atol=1e-2)
#    if func_check:
#        print(f'> Triton functionality for M correct!')
#    else:
#        print(f'> Triton functionality for M wrong...')
#        print(torch.topk(abs(row_max_torch.reshape(-1) - m_triton.reshape(-1)), k=20, dim=0))
#        exit()

    func_check = torch.allclose(ref_out_torch, out_triton, rtol=0, atol=1e-2)
    if func_check:
        print(f'> Triton functionality correct!')
    else:
        print(f'> Triton functionality wrong...')
        print(torch.topk(abs(ref_out_torch.reshape(-1) - out_triton.reshape(-1)), k=20, dim=0))
        exit()
    print()
    

    # Compute FLOPs
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    total_tflops = total_flops / (10**12)

    # Profile starts!
    NUM_WARMUP = 3
    # 1) Torch
    torch_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        p = torch.matmul(query, key.transpose(2, 3)) * (1.0 / math.sqrt(HEAD_DIM))
        p = p + causal_mask
        p = torch.softmax(p.float(), dim = -1).to(torch.float16)
        ref_out_torch = torch.matmul(p, value)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            torch_latency.append((e - s) * (10**3))
    
    # 2) FA-2
    fa2_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        ref_out_fa2 = flash_attn.flash_attn_func(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True).transpose(1, 2)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            fa2_latency.append((e - s) * (10**3))

    # 3) Custom CUDA-based
    custom_cuda_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        out_custom_cuda, _, _ = custom_flash_attn.flash_attn_fp16(query, key, value, False, False, ver)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            custom_cuda_latency.append((e - s) * (10**3))

    # 4) Triton-based
    triton_latency = []
    for i in range(NUM_WARMUP + NUM_TESTS):
        torch.cuda.synchronize()
        s = time.time()

        out_triton, s_triton, m_triton = triton_flash_attn_prefill(query, key, value, causal, warp_specialize, False)

        torch.cuda.synchronize()
        e = time.time()
        if i >= NUM_WARMUP:
            triton_latency.append((e - s) * (10**3))


    print("=========================")
    print("Profile result")
    print("=========================")
    print(f"> Ideal: {(total_tflops / 165.2) * 1000: .2f} ms \t 165.2 TFLOPS")
    print(f"> Mean")
    print(f"\t > Torch: {np.mean(torch_latency): .3f} ms \t {total_tflops / (np.mean(torch_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > FA-2 (CUDA): {np.mean(fa2_latency): .3f} ms \t {total_tflops / (np.mean(fa2_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > Custom (CUDA): {np.mean(custom_cuda_latency): .3f} ms \t {total_tflops / (np.mean(custom_cuda_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > Triton: {np.mean(triton_latency): .3f} ms \t {total_tflops / (np.mean(triton_latency) / (10**3)): .2f} TFLOPS")

    print(f"> Median")
    print(f"\t > Torch: {np.median(torch_latency): .3f} ms \t {total_tflops / (np.median(torch_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > FA-2 (CUDA): {np.median(fa2_latency): .3f} ms \t {total_tflops / (np.median(fa2_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > Custom (CUDA): {np.median(custom_cuda_latency): .3f} ms \t {total_tflops / (np.median(custom_cuda_latency) / (10**3)): .2f} TFLOPS")
    print(f"\t > Triton: {np.median(triton_latency): .3f} ms \t {total_tflops / (np.median(triton_latency) / (10**3)): .2f} TFLOPS")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--b", type=int, default=8, help="Batch size",)
    parser.add_argument("--h", type=int, default=32, help="#. of heads",)
    parser.add_argument("--n", type=int, default=1024, help="Query length",)
    parser.add_argument("--dim", type=int, default=128, help="Head dim",)
    parser.add_argument("--num_tests", type=int, default=10, help="Iteration number",)

    args = parser.parse_args()

    b, h, n, dim = args.b, args.h, args.n, args.dim

    torch.manual_seed(1234)

    query = torch.rand((b, h, n, dim), dtype=DTYPE, device=DEVICE)
    key = torch.rand((b, h, n, dim), dtype=DTYPE, device=DEVICE)
    value = torch.rand((b, h, n, dim), dtype=DTYPE, device=DEVICE)

    bench_flash_attention(query, key, value, b, h, n, dim, causal=True, NUM_TESTS=args.num_tests)
